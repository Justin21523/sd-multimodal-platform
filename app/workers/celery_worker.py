# app/workers/celery_worker.py
import asyncio
import logging
import time
import traceback
from datetime import datetime
from typing import Dict, Any, Optional

import torch
from celery import Celery, Task
from celery.signals import worker_init, worker_shutdown, task_prerun, task_postrun
from celery.exceptions import SoftTimeLimitExceeded, TimeLimitExceeded
from PIL import Image
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from app.config import settings
from app.shared_cache import shared_cache  # noqa: F401  (side-effect: set cache env vars)
from app.core.queue_manager import TaskStatus, TaskInfo, QueueManager, get_queue_manager

from services.postprocess.upscale_service import UpscaleService
from services.postprocess.face_restore_service import FaceRestoreService

logger = logging.getLogger(__name__)

# =====================================
# Celery App Configuration
# =====================================


def create_celery_app() -> Celery:
    """Create and configure Celery application"""
    redis_url = settings.get_redis_url()

    celery_app = Celery(
        "sd_platform_worker",
        broker=redis_url,
        backend=redis_url,
        include=["app.workers.celery_worker"],
    )

    # Celery configuration
    celery_app.conf.update(
        # Task routing and execution
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        # Concurrency and performance
        worker_concurrency=settings.CELERY_WORKER_CONCURRENCY,
        worker_prefetch_multiplier=1,  # Important for GPU tasks
        task_acks_late=True,
        worker_disable_rate_limits=True,
        # Task timeouts and retries
        task_soft_time_limit=1800,  # 30 minutes soft limit
        task_time_limit=2400,  # 40 minutes hard limit
        task_reject_on_worker_lost=True,
        # Memory and resource management
        worker_max_tasks_per_child=50,  # Restart worker after N tasks
        worker_max_memory_per_child=8 * 1024 * 1024,  # 8GB memory limit
        # Queue routing
        task_routes={
            "app.workers.celery_worker.process_txt2img": {"queue": "generation"},
            "app.workers.celery_worker.process_img2img": {"queue": "generation"},
            "app.workers.celery_worker.process_inpaint": {"queue": "generation"},
            "app.workers.celery_worker.process_upscale": {"queue": "postprocess"},
            "app.workers.celery_worker.process_face_restore": {"queue": "postprocess"},
            "app.workers.celery_worker.cleanup_expired_tasks": {"queue": "maintenance"},
        },
        # Result backend settings
        result_expires=3600,  # Results expire after 1 hour
        result_backend_transport_options={
            "master_name": "mymaster",
            "visibility_timeout": 3600,
        },
    )

    return celery_app


celery_app = create_celery_app()

# =====================================
# Worker State Management
# =====================================


class WorkerState:
    """Track worker state and resources"""

    def __init__(self):
        self.is_initialized = False
        self.gpu_memory_baseline = 0.0
        self.active_tasks = set()
        self.task_start_times = {}

        # Service instances (shared across tasks)
        self.upscale_service: Optional[UpscaleService] = None
        self.face_restore_service: Optional[FaceRestoreService] = None

        # Queue manager for status updates
        self.queue_manager: Optional[QueueManager] = None


worker_state = WorkerState()

# =====================================
# Worker Lifecycle Signals
# =====================================


@worker_init.connect
def init_worker(sender=None, **kwargs):
    """Initialize worker with AI models and services"""
    try:
        logger.info("Initializing Celery worker...")

        # Initialize GPU baseline memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            worker_state.gpu_memory_baseline = torch.cuda.memory_allocated() / 1024**3
            logger.info(
                f"GPU baseline memory: {worker_state.gpu_memory_baseline:.2f}GB"
            )

        # Initialize services (this will be done lazily per task type)
        worker_state.is_initialized = True
        logger.info("Celery worker initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize worker: {e}")
        raise


@worker_shutdown.connect
def shutdown_worker(sender=None, **kwargs):
    """Clean shutdown of worker resources"""
    try:
        logger.info("Shutting down Celery worker...")

        # Cleanup AI models
        try:
            from services.models.sd_models import get_model_manager

            model_manager = get_model_manager()
            if getattr(model_manager, "is_initialized", False):
                asyncio.run(model_manager.cleanup())
        except Exception as e:
            logger.error(f"Error cleaning up ModelManager: {e}")

        if worker_state.upscale_service:
            asyncio.run(worker_state.upscale_service.cleanup())
        if worker_state.face_restore_service:
            asyncio.run(worker_state.face_restore_service.cleanup())

        # Final GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Worker shutdown completed")

    except Exception as e:
        logger.error(f"Error during worker shutdown: {e}")


@task_prerun.connect
def task_prerun_handler(
    sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds
):
    """Pre-task execution setup"""
    try:
        worker_state.active_tasks.add(task_id)
        worker_state.task_start_times[task_id] = time.time()

        # Update task status to RUNNING
        if len(args) > 0:  # type: ignore  First arg should be task_id from our queue
            queue_task_id = args[0]  # type: ignore
            asyncio.run(_update_task_status(queue_task_id, TaskStatus.RUNNING))

        logger.info(
            f"Task {task_id} started, active tasks: {len(worker_state.active_tasks)}"
        )

    except Exception as e:
        logger.error(f"Error in task prerun for {task_id}: {e}")


@task_postrun.connect
def task_postrun_handler(
    sender=None,
    task_id=None,
    task=None,
    args=None,
    kwargs=None,
    retval=None,
    state=None,
    **kwds,
):
    """Post-task execution cleanup"""
    try:
        worker_state.active_tasks.discard(task_id)
        start_time = worker_state.task_start_times.pop(task_id, None)

        duration = time.time() - start_time if start_time else 0

        # GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(
            f"Task {task_id} completed in {duration:.2f}s, active tasks: {len(worker_state.active_tasks)}"
        )

    except Exception as e:
        logger.error(f"Error in task postrun for {task_id}: {e}")


# =====================================
# Custom Task Base Class
# =====================================


class GPUTask(Task):
    """Custom task class with GPU resource management"""

    def retry(
        self,
        args=None,
        kwargs=None,
        exc=None,
        throw=True,
        eta=None,
        countdown=None,
        max_retries=None,
        **options,
    ):
        """Override retry with GPU cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return super().retry(
            args, kwargs, exc, throw, eta, countdown, max_retries, **options
        )

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure with proper cleanup"""
        logger.error(f"Task {task_id} failed: {exc}")

        # Update queue task status if applicable
        if args and len(args) > 0:
            queue_task_id = args[0]
            error_info = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            asyncio.run(
                _update_task_status(
                    queue_task_id, TaskStatus.FAILED, error_info=error_info
                )
            )

        # GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =====================================
# Service Initialization Helpers
# =====================================


async def _get_upscale_service() -> UpscaleService:
    """Get or initialize upscale service"""
    if worker_state.upscale_service is None:
        worker_state.upscale_service = UpscaleService()
        await worker_state.upscale_service.initialize()  # type: ignore
    return worker_state.upscale_service


async def _get_face_restore_service() -> FaceRestoreService:
    """Get or initialize face restoration service"""
    if worker_state.face_restore_service is None:
        worker_state.face_restore_service = FaceRestoreService()
        await worker_state.face_restore_service.initialize()  # type: ignore
    return worker_state.face_restore_service


async def _get_queue_manager() -> QueueManager:
    """Get queue manager for status updates"""
    if worker_state.queue_manager is None:
        worker_state.queue_manager = await get_queue_manager()
    return worker_state.queue_manager


async def _update_task_status(task_id: str, status: TaskStatus, **updates):
    """Update task status in queue"""
    try:
        queue_manager = await _get_queue_manager()
        current = await queue_manager.get_task_status(task_id)
        if current and current.status in {
            TaskStatus.CANCELLED,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.TIMEOUT,
        }:
            if status != current.status:
                return
        await queue_manager.task_store.update_task_status(task_id, status, **updates)  # type: ignore
    except Exception as e:
        logger.error(f"Failed to update task status for {task_id}: {e}")


def _build_error_info(exc: BaseException) -> Dict[str, Any]:
    return {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _get_retry_config(params: Dict[str, Any]) -> tuple[int, int]:
    max_retries = params.get("max_retries", getattr(settings, "MAX_RETRIES", 3))
    backoff_sec = params.get("retry_backoff_sec", getattr(settings, "RETRY_BACKOFF_SEC", 5))
    try:
        max_retries_int = int(max_retries)
    except Exception:
        max_retries_int = 3
    try:
        backoff_int = int(backoff_sec)
    except Exception:
        backoff_int = 5
    return max(0, max_retries_int), max(1, backoff_int)


def _is_retryable_exception(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):  # type: ignore[attr-defined]
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


class TaskCancelledError(RuntimeError):
    pass


class TaskTimeoutError(RuntimeError):
    pass


async def _raise_if_task_terminal(task_id: str) -> None:
    """
    Best-effort cooperative cancellation/timeout check.

    If Redis/queue is unavailable, this becomes a no-op (worker continues).
    """
    try:
        queue_manager = await _get_queue_manager()
        task_info = await queue_manager.get_task_status(task_id)
    except Exception:
        return

    if not task_info:
        return

    if task_info.status == TaskStatus.CANCELLED:
        raise TaskCancelledError("Task was cancelled")
    if task_info.status == TaskStatus.TIMEOUT:
        raise TaskTimeoutError("Task timed out")


def _threadsafe_raise_if_task_terminal(
    task_id: str, loop: asyncio.AbstractEventLoop
) -> None:
    """
    Run the cooperative cancellation/timeout check from a worker thread.

    This is used inside model callbacks/loops to stop long-running work quickly.
    """
    try:
        asyncio.run_coroutine_threadsafe(_raise_if_task_terminal(task_id), loop).result(
            timeout=1.0
        )
    except (TaskCancelledError, TaskTimeoutError):
        raise
    except Exception:
        # Best-effort only: if the loop/Redis is unavailable, keep running.
        return


def _resolve_allowed_image_path(path_str: str, *, allowed_roots: Optional[list[Path]] = None) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        raise ValueError("image_path must be an absolute path")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Image path does not exist: {path}")

    resolved = path.resolve()
    roots = allowed_roots or [
        Path(settings.ASSETS_PATH).expanduser().resolve(),
        Path(str(settings.OUTPUT_PATH)).expanduser().resolve(),
    ]
    for root in roots:
        root_resolved = Path(root).expanduser().resolve()
        try:
            if resolved.is_relative_to(root_resolved):
                return resolved
        except AttributeError:  # pragma: no cover (py<3.9)
            if str(resolved).startswith(str(root_resolved) + "/"):
                return resolved
    raise ValueError(
        f"image_path must be under allowed roots: {[str(r) for r in roots]}"
    )


def _open_image_file(path: Path) -> Image.Image:
    img = Image.open(path)
    img.load()
    return img


async def _load_image_from_asset_id(asset_id: str) -> Image.Image:
    from services.assets.asset_manager import get_asset_manager

    asset_manager = get_asset_manager()
    asset = await asset_manager.get_asset(asset_id)
    if not asset:
        raise ValueError(f"Asset not found: {asset_id}")
    file_path = asset.get("file_path")
    if not isinstance(file_path, str) or not file_path:
        raise ValueError(f"Asset {asset_id} missing file_path")
    resolved = _resolve_allowed_image_path(
        file_path, allowed_roots=[Path(settings.ASSETS_PATH)]
    )
    return _open_image_file(resolved)


async def _load_image_from_sources(
    *,
    base64_data: Optional[str],
    asset_id: Optional[str],
    image_path: Optional[str],
    label: str,
) -> Image.Image:
    provided = [base64_data is not None, asset_id is not None, image_path is not None]
    if sum(provided) != 1:
        raise ValueError(
            f"{label} requires exactly one of: base64, asset_id, image_path"
        )
    if base64_data is not None:
        from utils.image_utils import base64_to_pil_image

        return base64_to_pil_image(base64_data)
    if asset_id is not None:
        return await _load_image_from_asset_id(asset_id)
    assert image_path is not None
    resolved = _resolve_allowed_image_path(image_path)
    return _open_image_file(resolved)


async def _get_task_user_id(task_id: str) -> Optional[str]:
    try:
        queue_manager = await _get_queue_manager()
        task_info = await queue_manager.get_task_status(task_id)
        if task_info and isinstance(task_info.user_id, str) and task_info.user_id:
            return task_info.user_id
    except Exception:
        return None
    return None


def _record_history_completion(
    *,
    history_id: str,
    task_type: str,
    input_params: Dict[str, Any],
    result_data: Dict[str, Any],
) -> None:
    try:
        from services.history import get_history_store

        user_id = input_params.get("user_id")
        if not (isinstance(user_id, str) and user_id):
            try:
                user_id = asyncio.run(_get_task_user_id(history_id))
            except Exception:
                user_id = None

        store = get_history_store()
        store.record_completion(
            history_id=history_id,
            task_type=task_type,
            run_mode="async",
            user_id=user_id if isinstance(user_id, str) and user_id else None,
            input_params=input_params,
            result_data=result_data,
        )
    except Exception as e:
        logger.warning(f"Failed to write history record for {history_id}: {e}")


# =====================================
# Task Implementation Functions
# =====================================


@celery_app.task(bind=True, base=GPUTask, name="process_txt2img")
def process_txt2img(self, task_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Process text-to-image generation task"""
    try:
        logger.info(f"Processing txt2img task: {task_id}")

        # Run async service call
        result = asyncio.run(_process_txt2img_async(task_id, params))

        # Update task status on success
        asyncio.run(
            _update_task_status(
                task_id, TaskStatus.COMPLETED, result_data=result, progress_percent=100
            )
        )
        _record_history_completion(
            history_id=task_id,
            task_type="txt2img",
            input_params=params,
            result_data=result,
        )

        return result

    except TaskCancelledError as e:
        logger.info(f"txt2img task cancelled: {task_id}")
        error_info = {"error_type": "Cancelled", "error_message": str(e)}
        asyncio.run(
            _update_task_status(
                task_id,
                TaskStatus.CANCELLED,
                error_info=error_info,
                current_step="cancelled",
            )
        )
        return {
            "success": False,
            "task_id": task_id,
            "task_type": "txt2img",
            "error": error_info,
        }

    except TaskTimeoutError as e:
        logger.info(f"txt2img task timed out (cooperative): {task_id}")
        error_info = {"error_type": "Timeout", "error_message": str(e)}
        asyncio.run(
            _update_task_status(
                task_id,
                TaskStatus.TIMEOUT,
                error_info=error_info,
                current_step="timeout",
            )
        )
        raise

    except (SoftTimeLimitExceeded, TimeLimitExceeded) as e:
        logger.error(f"Timeout processing txt2img task {task_id}: {e}")
        error_info = _build_error_info(e)
        asyncio.run(
            _update_task_status(task_id, TaskStatus.TIMEOUT, error_info=error_info)
        )
        raise

    except Exception as e:
        logger.error(f"Error processing txt2img task {task_id}: {e}")

        if _is_retryable_exception(e):
            max_retries, backoff_sec = _get_retry_config(params)
            if getattr(self.request, "retries", 0) < max_retries:
                retry_index = getattr(self.request, "retries", 0)
                countdown = min(300, backoff_sec * (2**retry_index))
                error_info = _build_error_info(e)
                asyncio.run(
                    _update_task_status(
                        task_id,
                        TaskStatus.RETRYING,
                        retry_count=retry_index + 1,
                        error_info=error_info,
                        current_step="retrying",
                    )
                )
                raise self.retry(exc=e, countdown=countdown, max_retries=max_retries)

        # Update task status on failure
        error_info = _build_error_info(e)
        asyncio.run(_update_task_status(task_id, TaskStatus.FAILED, error_info=error_info))

        raise


async def _process_txt2img_async(
    task_id: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Async implementation of txt2img processing"""
    from services.models.sd_models import get_model_manager
    from utils.file_utils import get_output_url, save_generation_output

    await _raise_if_task_terminal(task_id)

    # Update progress
    await _update_task_status(
        task_id, TaskStatus.RUNNING, progress_percent=10, current_step="loading_model"
    )
    await _raise_if_task_terminal(task_id)

    # Generate image
    await _update_task_status(
        task_id,
        TaskStatus.RUNNING,
        progress_percent=25,
        current_step="generating",
        total_steps=num_inference_steps,
    )
    await _raise_if_task_terminal(task_id)
    await _raise_if_task_terminal(task_id)

    prompt = params.get("prompt", "")
    negative_prompt = params.get("negative_prompt", "")
    num_inference_steps = params.get(
        "num_inference_steps", params.get("steps", settings.DEFAULT_STEPS)
    )
    guidance_scale = params.get(
        "guidance_scale", params.get("cfg_scale", settings.DEFAULT_CFG)
    )
    seed = params.get("seed")
    num_images = params.get("num_images", 1)

    try:
        num_inference_steps = int(num_inference_steps)  # type: ignore[arg-type]
    except Exception:
        num_inference_steps = settings.DEFAULT_STEPS

    try:
        guidance_scale = float(guidance_scale)  # type: ignore[arg-type]
    except Exception:
        guidance_scale = settings.DEFAULT_CFG

    if seed is not None:
        try:
            seed = int(seed)  # type: ignore[arg-type]
        except Exception:
            seed = None
    if seed == -1:
        seed = None

    try:
        num_images = int(num_images)  # type: ignore[arg-type]
    except Exception:
        num_images = 1
    num_images = max(1, min(num_images, settings.MAX_BATCH_SIZE))

    try:
        width = int(params.get("width"))  # type: ignore[arg-type]
    except Exception:
        width = settings.DEFAULT_WIDTH
    try:
        height = int(params.get("height"))  # type: ignore[arg-type]
    except Exception:
        height = settings.DEFAULT_HEIGHT

    model_manager = get_model_manager()
    target_model = params.get("model_id") or model_manager.auto_select_model(
        prompt, task_type="txt2img"
    )
    if not model_manager.is_initialized:
        ok = await model_manager.initialize(target_model)
        if not ok:
            raise RuntimeError(f"Failed to initialize model: {target_model}")
    elif model_manager.current_model != target_model:
        ok = await model_manager.switch_model(target_model)
        if not ok:
            raise RuntimeError(f"Failed to switch to model: {target_model}")

    await _update_task_status(
        task_id,
        TaskStatus.RUNNING,
        progress_percent=25,
        current_step="generating",
        total_steps=num_inference_steps,
    )
    await _raise_if_task_terminal(task_id)

    loop = asyncio.get_running_loop()
    abort_check = lambda: _threadsafe_raise_if_task_terminal(task_id, loop)
    last_step = {"value": -1}
    stride = max(1, int(num_inference_steps // 50))

    def _on_step(step: int, total_steps: int) -> None:
        _threadsafe_raise_if_task_terminal(task_id, loop)
        if step <= last_step["value"]:
            return
        last_step["value"] = step
        denom = max(1, int(total_steps))
        pct = 25 + int(((step + 1) / denom) * 60)
        pct = max(0, min(pct, 89))
        try:
            asyncio.run_coroutine_threadsafe(
                _update_task_status(
                    task_id,
                    TaskStatus.RUNNING,
                    progress_percent=pct,
                    current_step=f"step {step + 1}/{denom}",
                    total_steps=denom,
                ),
                loop,
            )
        except Exception:
            return

    generation_result = await asyncio.to_thread(
        lambda: asyncio.run(
            model_manager.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                num_images=num_images,
                progress_callback=_on_step,
                callback_steps=stride,
            )
        )
    )

    await _raise_if_task_terminal(task_id)

    await _update_task_status(
        task_id, TaskStatus.RUNNING, progress_percent=90, current_step="saving_results"
    )
    await _raise_if_task_terminal(task_id)

    images = generation_result.get("images") or []
    image_items = []
    for i, image in enumerate(images):
        await _raise_if_task_terminal(task_id)
        image_path = await save_generation_output(
            image, task_id=f"{task_id}_{i}", subfolder="txt2img"
        )
        image_items.append(
            {
                "image_path": str(image_path),
                "image_url": get_output_url(image_path),
                "width": image.width,
                "height": image.height,
            }
        )

    return {
        "success": True,
        "task_id": task_id,
        "task_type": "txt2img",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "parameters": {
            "model_id": target_model,
            "width": width,
            "height": height,
            "steps": num_inference_steps,
            "cfg_scale": guidance_scale,
            "seed": seed,
            "num_images": num_images,
        },
        "result": {
            "images": image_items,
            "image_count": len(image_items),
            "model_used": target_model,
        },
    }


@celery_app.task(bind=True, base=GPUTask, name="process_img2img")
def process_img2img(self, task_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Process image-to-image generation task"""
    try:
        logger.info(f"Processing img2img task: {task_id}")

        result = asyncio.run(_process_img2img_async(task_id, params))

        asyncio.run(
            _update_task_status(
                task_id, TaskStatus.COMPLETED, result_data=result, progress_percent=100
            )
        )
        _record_history_completion(
            history_id=task_id,
            task_type="img2img",
            input_params=params,
            result_data=result,
        )

        return result

    except TaskCancelledError as e:
        logger.info(f"img2img task cancelled: {task_id}")
        error_info = {"error_type": "Cancelled", "error_message": str(e)}
        asyncio.run(
            _update_task_status(
                task_id,
                TaskStatus.CANCELLED,
                error_info=error_info,
                current_step="cancelled",
            )
        )
        return {
            "success": False,
            "task_id": task_id,
            "task_type": "img2img",
            "error": error_info,
        }

    except TaskTimeoutError as e:
        logger.info(f"img2img task timed out (cooperative): {task_id}")
        error_info = {"error_type": "Timeout", "error_message": str(e)}
        asyncio.run(
            _update_task_status(
                task_id,
                TaskStatus.TIMEOUT,
                error_info=error_info,
                current_step="timeout",
            )
        )
        raise

    except (SoftTimeLimitExceeded, TimeLimitExceeded) as e:
        logger.error(f"Timeout processing img2img task {task_id}: {e}")
        error_info = _build_error_info(e)
        asyncio.run(
            _update_task_status(task_id, TaskStatus.TIMEOUT, error_info=error_info)
        )
        raise

    except Exception as e:
        logger.error(f"Error processing img2img task {task_id}: {e}")

        if _is_retryable_exception(e):
            max_retries, backoff_sec = _get_retry_config(params)
            if getattr(self.request, "retries", 0) < max_retries:
                retry_index = getattr(self.request, "retries", 0)
                countdown = min(300, backoff_sec * (2**retry_index))
                error_info = _build_error_info(e)
                asyncio.run(
                    _update_task_status(
                        task_id,
                        TaskStatus.RETRYING,
                        retry_count=retry_index + 1,
                        error_info=error_info,
                        current_step="retrying",
                    )
                )
                raise self.retry(exc=e, countdown=countdown, max_retries=max_retries)

        error_info = _build_error_info(e)
        asyncio.run(_update_task_status(task_id, TaskStatus.FAILED, error_info=error_info))

        raise


async def _process_img2img_async(
    task_id: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Async implementation of img2img processing"""
    await _raise_if_task_terminal(task_id)
    await _update_task_status(
        task_id, TaskStatus.RUNNING, progress_percent=10, current_step="loading_model"
    )
    await _raise_if_task_terminal(task_id)

    # Load input image
    input_image = await _load_image_from_sources(
        base64_data=params.get("init_image") or params.get("image"),
        asset_id=params.get("init_asset_id"),
        image_path=params.get("image_path") or params.get("init_image_path"),
        label="img2img init_image",
    )

    await _raise_if_task_terminal(task_id)

    prompt = params.get("prompt", "")
    negative_prompt = params.get("negative_prompt", "")
    strength = params.get("strength", 0.8)
    num_inference_steps = params.get(
        "num_inference_steps", params.get("steps", settings.DEFAULT_STEPS)
    )
    guidance_scale = params.get(
        "guidance_scale", params.get("cfg_scale", settings.DEFAULT_CFG)
    )
    seed = params.get("seed")
    width = params.get("width")
    height = params.get("height")

    try:
        strength = float(strength)  # type: ignore[arg-type]
    except Exception:
        strength = 0.8

    try:
        num_inference_steps = int(num_inference_steps)  # type: ignore[arg-type]
    except Exception:
        num_inference_steps = settings.DEFAULT_STEPS

    try:
        guidance_scale = float(guidance_scale)  # type: ignore[arg-type]
    except Exception:
        guidance_scale = settings.DEFAULT_CFG

    if seed is not None:
        try:
            seed = int(seed)  # type: ignore[arg-type]
        except Exception:
            seed = None
    if seed == -1:
        seed = None

    if width is not None:
        try:
            width = int(width)  # type: ignore[arg-type]
        except Exception:
            width = None

    if height is not None:
        try:
            height = int(height)  # type: ignore[arg-type]
        except Exception:
            height = None

    controlnet_cfg = params.get("controlnet")
    if isinstance(controlnet_cfg, dict) and controlnet_cfg:
        await _update_task_status(
            task_id,
            TaskStatus.RUNNING,
            progress_percent=25,
            current_step="loading_controlnet",
        )
        await _raise_if_task_terminal(task_id)

        from services.models.sd_models import ModelRegistry
        from services.processors.controlnet_service import get_controlnet_manager
        from utils.image_utils import prepare_img2img_image
        from utils.file_utils import get_output_url, save_generation_output

        controlnet_type = controlnet_cfg.get("type")
        if not controlnet_type:
            raise ValueError("ControlNet requires `controlnet.type`")

        control_image = await _load_image_from_sources(
            base64_data=controlnet_cfg.get("image"),
            asset_id=controlnet_cfg.get("asset_id") or params.get("control_asset_id"),
            image_path=controlnet_cfg.get("image_path")
            or params.get("control_image_path"),
            label="controlnet image",
        )

        # Resolve base model path (local only; must follow ~/Desktop/data_model_structure.md)
        requested_model_id = params.get("model_id") or settings.PRIMARY_MODEL
        if requested_model_id not in ModelRegistry.AVAILABLE_MODELS:
            requested_model_id = settings.PRIMARY_MODEL
        local_rel = ModelRegistry.AVAILABLE_MODELS.get(requested_model_id, {}).get(
            "local_path"
        )
        if not isinstance(local_rel, str) or not local_rel:
            raise ValueError(f"Model {requested_model_id} is missing local_path metadata")
        base_model_path = str((Path(settings.MODELS_PATH) / local_rel).resolve())
        if not Path(base_model_path).exists():
            raise FileNotFoundError(
                f"Base model not found at: {base_model_path}. Install models under {settings.MODELS_PATH}."
            )

        await _raise_if_task_terminal(task_id)

        # Prepare images (ensure sizes match & are SD-compatible)
        init_image = prepare_img2img_image(
            input_image,
            target_width=width if isinstance(width, int) else None,
            target_height=height if isinstance(height, int) else None,
        )
        control_image = prepare_img2img_image(
            control_image, target_width=init_image.width, target_height=init_image.height
        )

        await _update_task_status(
            task_id,
            TaskStatus.RUNNING,
            progress_percent=40,
            current_step="generating_controlnet",
            total_steps=num_inference_steps,
        )
        await _raise_if_task_terminal(task_id)

        controlnet_manager = get_controlnet_manager()
        ok = await controlnet_manager.create_pipeline(
            base_model_path, controlnet_type, pipeline_mode="img2img"
        )
        if not ok:
            raise RuntimeError(f"Failed to setup ControlNet pipeline: {controlnet_type}")

        controlnet_params = {"preprocess": bool(controlnet_cfg.get("preprocess", True))}

        loop = asyncio.get_running_loop()
        last_step = {"value": -1}
        stride = max(1, int(num_inference_steps // 50))

        def _on_step(step: int, total_steps: int) -> None:
            _threadsafe_raise_if_task_terminal(task_id, loop)
            if step <= last_step["value"]:
                return
            last_step["value"] = step
            denom = max(1, int(total_steps))
            pct = 40 + int(((step + 1) / denom) * 49)
            pct = max(0, min(pct, 89))
            try:
                asyncio.run_coroutine_threadsafe(
                    _update_task_status(
                        task_id,
                        TaskStatus.RUNNING,
                        progress_percent=pct,
                        current_step=f"step {step + 1}/{denom}",
                        total_steps=denom,
                    ),
                    loop,
                )
            except Exception:
                return

        cn_result = await asyncio.to_thread(
            lambda: asyncio.run(
                controlnet_manager.generate_with_controlnet(
                    prompt=prompt,
                    init_image=init_image,
                    control_image=control_image,
                    controlnet_type=controlnet_type,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    controlnet_strength=controlnet_cfg.get("strength", 1.0),
                    guidance_start=controlnet_cfg.get("guidance_start", 0.0),
                    guidance_end=controlnet_cfg.get("guidance_end", 1.0),
                    seed=seed,
                    controlnet_params=controlnet_params,
                    width=init_image.width,
                    height=init_image.height,
                    progress_callback=_on_step,
                    callback_steps=stride,
                )
            )
        )

        await _raise_if_task_terminal(task_id)

        await _update_task_status(
            task_id, TaskStatus.RUNNING, progress_percent=90, current_step="saving_results"
        )
        await _raise_if_task_terminal(task_id)

        generated_images = cn_result.get("images") or []
        image_items = []
        for i, image in enumerate(generated_images):
            await _raise_if_task_terminal(task_id)
            image_path = await save_generation_output(
                image, task_id=f"{task_id}_{i}", subfolder="img2img"
            )
            image_items.append(
                {
                    "image_path": str(image_path),
                    "image_url": get_output_url(image_path),
                    "width": image.width,
                    "height": image.height,
                }
            )

        result = {
            "success": True,
            "task_id": task_id,
            "task_type": "img2img",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "strength": strength,
            "parameters": {
                "model_id": requested_model_id,
                "steps": num_inference_steps,
                "cfg_scale": guidance_scale,
                "seed": seed,
            },
            "result": {
                "images": image_items,
                "image_count": len(image_items),
                "model_used": requested_model_id,
            },
            "controlnet_info": {
                "type": controlnet_type,
                "strength": controlnet_cfg.get("strength", 1.0),
                "preprocess": bool(controlnet_cfg.get("preprocess", True)),
                "guidance_start": controlnet_cfg.get("guidance_start", 0.0),
                "guidance_end": controlnet_cfg.get("guidance_end", 1.0),
            },
        }
        return result

    from services.models.sd_models import get_model_manager
    from utils.file_utils import get_output_url, save_generation_output
    from utils.image_utils import prepare_img2img_image

    await _update_task_status(
        task_id, TaskStatus.RUNNING, progress_percent=30, current_step="generating"
    )

    num_images = params.get("num_images", 1)
    try:
        num_images = int(num_images)  # type: ignore[arg-type]
    except Exception:
        num_images = 1
    num_images = max(1, min(num_images, settings.MAX_BATCH_SIZE))

    init_image = prepare_img2img_image(
        input_image,
        target_width=width if isinstance(width, int) else None,
        target_height=height if isinstance(height, int) else None,
    )

    model_manager = get_model_manager()
    target_model = params.get("model_id") or model_manager.auto_select_model(
        prompt, task_type="img2img"
    )
    if not model_manager.is_initialized:
        ok = await model_manager.initialize(target_model)
        if not ok:
            raise RuntimeError(f"Failed to initialize model: {target_model}")
    elif model_manager.current_model != target_model:
        ok = await model_manager.switch_model(target_model)
        if not ok:
            raise RuntimeError(f"Failed to switch to model: {target_model}")

    await _raise_if_task_terminal(task_id)

    loop = asyncio.get_running_loop()
    last_step = {"value": -1}
    stride = max(1, int(num_inference_steps // 50))

    def _on_step(step: int, total_steps: int) -> None:
        _threadsafe_raise_if_task_terminal(task_id, loop)
        if step <= last_step["value"]:
            return
        last_step["value"] = step
        denom = max(1, int(total_steps))
        pct = 25 + int(((step + 1) / denom) * 60)
        pct = max(0, min(pct, 89))
        try:
            asyncio.run_coroutine_threadsafe(
                _update_task_status(
                    task_id,
                    TaskStatus.RUNNING,
                    progress_percent=pct,
                    current_step=f"step {step + 1}/{denom}",
                    total_steps=denom,
                ),
                loop,
            )
        except Exception:
            return

    generation_result = await asyncio.to_thread(
        lambda: asyncio.run(
            model_manager.generate_img2img(
                prompt=prompt,
                image=init_image,
                negative_prompt=negative_prompt,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                width=init_image.width,
                height=init_image.height,
                num_images=num_images,
                progress_callback=_on_step,
                callback_steps=stride,
            )
        )
    )

    await _raise_if_task_terminal(task_id)

    await _update_task_status(
        task_id, TaskStatus.RUNNING, progress_percent=90, current_step="saving_results"
    )
    await _raise_if_task_terminal(task_id)

    images = generation_result.get("images") or []
    image_items = []
    for i, image in enumerate(images):
        await _raise_if_task_terminal(task_id)
        image_path = await save_generation_output(
            image, task_id=f"{task_id}_{i}", subfolder="img2img"
        )
        image_items.append(
            {
                "image_path": str(image_path),
                "image_url": get_output_url(image_path),
                "width": image.width,
                "height": image.height,
            }
        )

    return {
        "success": True,
        "task_id": task_id,
        "task_type": "img2img",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "input_image_size": f"{init_image.width}x{init_image.height}",
        "strength": strength,
        "parameters": {
            "model_id": target_model,
            "steps": num_inference_steps,
            "cfg_scale": guidance_scale,
            "seed": seed,
            "num_images": num_images,
        },
        "result": {
            "images": image_items,
            "image_count": len(image_items),
            "model_used": target_model,
        },
    }


@celery_app.task(bind=True, base=GPUTask, name="process_inpaint")
def process_inpaint(self, task_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Process inpainting generation task"""
    try:
        logger.info(f"Processing inpaint task: {task_id}")

        result = asyncio.run(_process_inpaint_async(task_id, params))

        asyncio.run(
            _update_task_status(
                task_id, TaskStatus.COMPLETED, result_data=result, progress_percent=100
            )
        )
        _record_history_completion(
            history_id=task_id,
            task_type="inpaint",
            input_params=params,
            result_data=result,
        )

        return result

    except TaskCancelledError as e:
        logger.info(f"inpaint task cancelled: {task_id}")
        error_info = {"error_type": "Cancelled", "error_message": str(e)}
        asyncio.run(
            _update_task_status(
                task_id,
                TaskStatus.CANCELLED,
                error_info=error_info,
                current_step="cancelled",
            )
        )
        return {
            "success": False,
            "task_id": task_id,
            "task_type": "inpaint",
            "error": error_info,
        }

    except TaskTimeoutError as e:
        logger.info(f"inpaint task timed out (cooperative): {task_id}")
        error_info = {"error_type": "Timeout", "error_message": str(e)}
        asyncio.run(
            _update_task_status(
                task_id,
                TaskStatus.TIMEOUT,
                error_info=error_info,
                current_step="timeout",
            )
        )
        raise

    except (SoftTimeLimitExceeded, TimeLimitExceeded) as e:
        logger.error(f"Timeout processing inpaint task {task_id}: {e}")
        error_info = _build_error_info(e)
        asyncio.run(
            _update_task_status(task_id, TaskStatus.TIMEOUT, error_info=error_info)
        )
        raise

    except Exception as e:
        logger.error(f"Error processing inpaint task {task_id}: {e}")

        if _is_retryable_exception(e):
            max_retries, backoff_sec = _get_retry_config(params)
            if getattr(self.request, "retries", 0) < max_retries:
                retry_index = getattr(self.request, "retries", 0)
                countdown = min(300, backoff_sec * (2**retry_index))
                error_info = _build_error_info(e)
                asyncio.run(
                    _update_task_status(
                        task_id,
                        TaskStatus.RETRYING,
                        retry_count=retry_index + 1,
                        error_info=error_info,
                        current_step="retrying",
                    )
                )
                raise self.retry(exc=e, countdown=countdown, max_retries=max_retries)

        error_info = _build_error_info(e)
        asyncio.run(_update_task_status(task_id, TaskStatus.FAILED, error_info=error_info))

        raise


async def _process_inpaint_async(task_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Async implementation of inpaint processing"""
    from services.models.sd_models import get_model_manager
    from utils.file_utils import get_output_url, save_generation_output
    from utils.image_utils import prepare_inpaint_mask

    await _raise_if_task_terminal(task_id)

    await _update_task_status(
        task_id, TaskStatus.RUNNING, progress_percent=10, current_step="loading_model"
    )
    await _raise_if_task_terminal(task_id)

    prompt = params.get("prompt", "")
    negative_prompt = params.get("negative_prompt", "")
    strength = params.get("strength", 0.75)
    num_inference_steps = params.get("num_inference_steps", params.get("steps", 25))
    guidance_scale = params.get("guidance_scale", params.get("cfg_scale", 7.5))
    seed = params.get("seed")
    width = params.get("width")
    height = params.get("height")
    mask_blur = params.get("mask_blur", 4)

    try:
        strength = float(strength)  # type: ignore[arg-type]
    except Exception:
        strength = 0.75
    try:
        num_inference_steps = int(num_inference_steps)  # type: ignore[arg-type]
    except Exception:
        num_inference_steps = 25
    try:
        guidance_scale = float(guidance_scale)  # type: ignore[arg-type]
    except Exception:
        guidance_scale = 7.5
    if seed is not None:
        try:
            seed = int(seed)  # type: ignore[arg-type]
        except Exception:
            seed = None
    if seed == -1:
        seed = None

    init_image = await _load_image_from_sources(
        base64_data=params.get("init_image"),
        asset_id=params.get("init_asset_id"),
        image_path=params.get("image_path") or params.get("init_image_path"),
        label="inpaint init_image",
    )
    mask_image = await _load_image_from_sources(
        base64_data=params.get("mask_image"),
        asset_id=params.get("mask_asset_id"),
        image_path=params.get("mask_path") or params.get("mask_image_path"),
        label="inpaint mask_image",
    )
    init_image, mask_image = prepare_inpaint_mask(
        init_image,
        mask_image,
        blur_radius=int(mask_blur) if isinstance(mask_blur, (int, float, str)) else 4,
        target_width=width if isinstance(width, int) else None,
        target_height=height if isinstance(height, int) else None,
    )

    await _update_task_status(
        task_id,
        TaskStatus.RUNNING,
        progress_percent=35,
        current_step="generating",
        total_steps=num_inference_steps,
    )
    await _raise_if_task_terminal(task_id)

    model_manager = get_model_manager()
    target_model = params.get("model_id") or model_manager.auto_select_model(
        prompt, task_type="inpaint"
    )
    if model_manager.current_model != target_model:
        ok = await model_manager.switch_model(target_model)
        if not ok:
            raise RuntimeError(f"Failed to switch to model: {target_model}")

    await _raise_if_task_terminal(task_id)

    loop = asyncio.get_running_loop()
    last_step = {"value": -1}
    stride = max(1, int(num_inference_steps // 50))

    def _on_step(step: int, total_steps: int) -> None:
        _threadsafe_raise_if_task_terminal(task_id, loop)
        if step <= last_step["value"]:
            return
        last_step["value"] = step
        denom = max(1, int(total_steps))
        pct = 35 + int(((step + 1) / denom) * 54)
        pct = max(0, min(pct, 89))
        try:
            asyncio.run_coroutine_threadsafe(
                _update_task_status(
                    task_id,
                    TaskStatus.RUNNING,
                    progress_percent=pct,
                    current_step=f"step {step + 1}/{denom}",
                    total_steps=denom,
                ),
                loop,
            )
        except Exception:
            return

    generation_result = await asyncio.to_thread(
        lambda: asyncio.run(
            model_manager.generate_inpaint(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                mask_image=mask_image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=init_image.width,
                height=init_image.height,
                seed=seed,
                progress_callback=_on_step,
                callback_steps=stride,
            )
        )
    )

    await _raise_if_task_terminal(task_id)

    await _update_task_status(
        task_id, TaskStatus.RUNNING, progress_percent=90, current_step="saving_results"
    )
    await _raise_if_task_terminal(task_id)

    images = generation_result.get("images") or []
    image_items = []
    for i, image in enumerate(images):
        await _raise_if_task_terminal(task_id)
        image_path = await save_generation_output(
            image, task_id=f"{task_id}_{i}", subfolder="inpaint"
        )
        image_items.append(
            {
                "image_path": str(image_path),
                "image_url": get_output_url(image_path),
                "width": image.width,
                "height": image.height,
            }
        )

    return {
        "success": True,
        "task_id": task_id,
        "task_type": "inpaint",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "strength": strength,
        "inpaint_info": {
            "mask_blur": mask_blur,
            "fill_method": params.get("inpainting_fill", "original"),
        },
        "parameters": {
            "model_id": target_model,
            "steps": num_inference_steps,
            "cfg_scale": guidance_scale,
            "seed": seed,
        },
        "result": {
            "images": image_items,
            "image_count": len(image_items),
            "model_used": target_model,
        },
    }


@celery_app.task(bind=True, base=GPUTask, name="process_upscale")
def process_upscale(self, task_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Process image upscaling task"""
    try:
        logger.info(f"Processing upscale task: {task_id}")

        result = asyncio.run(_process_upscale_async(task_id, params))

        asyncio.run(
            _update_task_status(
                task_id, TaskStatus.COMPLETED, result_data=result, progress_percent=100
            )
        )
        _record_history_completion(
            history_id=task_id,
            task_type="upscale",
            input_params=params,
            result_data=result,
        )

        return result

    except TaskCancelledError as e:
        logger.info(f"upscale task cancelled: {task_id}")
        error_info = {"error_type": "Cancelled", "error_message": str(e)}
        asyncio.run(
            _update_task_status(
                task_id,
                TaskStatus.CANCELLED,
                error_info=error_info,
                current_step="cancelled",
            )
        )
        return {
            "success": False,
            "task_id": task_id,
            "task_type": "upscale",
            "error": error_info,
        }

    except TaskTimeoutError as e:
        logger.info(f"upscale task timed out (cooperative): {task_id}")
        error_info = {"error_type": "Timeout", "error_message": str(e)}
        asyncio.run(
            _update_task_status(
                task_id,
                TaskStatus.TIMEOUT,
                error_info=error_info,
                current_step="timeout",
            )
        )
        raise

    except (SoftTimeLimitExceeded, TimeLimitExceeded) as e:
        logger.error(f"Timeout processing upscale task {task_id}: {e}")
        error_info = _build_error_info(e)
        asyncio.run(
            _update_task_status(task_id, TaskStatus.TIMEOUT, error_info=error_info)
        )
        raise

    except Exception as e:
        logger.error(f"Error processing upscale task {task_id}: {e}")

        if _is_retryable_exception(e):
            max_retries, backoff_sec = _get_retry_config(params)
            if getattr(self.request, "retries", 0) < max_retries:
                retry_index = getattr(self.request, "retries", 0)
                countdown = min(300, backoff_sec * (2**retry_index))
                error_info = _build_error_info(e)
                asyncio.run(
                    _update_task_status(
                        task_id,
                        TaskStatus.RETRYING,
                        retry_count=retry_index + 1,
                        error_info=error_info,
                        current_step="retrying",
                    )
                )
                raise self.retry(exc=e, countdown=countdown, max_retries=max_retries)

        error_info = _build_error_info(e)
        asyncio.run(_update_task_status(task_id, TaskStatus.FAILED, error_info=error_info))

        raise


async def _process_upscale_async(
    task_id: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Async implementation of upscaling"""
    service = await _get_upscale_service()

    await _raise_if_task_terminal(task_id)

    await _update_task_status(
        task_id, TaskStatus.RUNNING, progress_percent=10, current_step="loading_model"
    )
    await _raise_if_task_terminal(task_id)

    input_image = await _load_image_from_sources(
        base64_data=params.get("image"),
        asset_id=params.get("image_asset_id") or params.get("asset_id"),
        image_path=params.get("image_path"),
        label="upscale image",
    )

    await _raise_if_task_terminal(task_id)

    await _update_task_status(
        task_id, TaskStatus.RUNNING, progress_percent=30, current_step="upscaling"
    )
    await _raise_if_task_terminal(task_id)

    raw_scale = params.get("scale", 4)
    try:
        scale = int(raw_scale)  # type: ignore[arg-type]
    except Exception:
        scale = 4
    scale = max(1, min(scale, 8))

    model_name = params.get("model", "RealESRGAN_x4plus")
    if not isinstance(model_name, str) or not model_name.strip():
        model_name = "RealESRGAN_x4plus"

    tile_size = params.get("tile_size")
    if tile_size is not None:
        try:
            tile_size = int(tile_size)  # type: ignore[arg-type]
        except Exception:
            tile_size = None

    await _raise_if_task_terminal(task_id)

    loop = asyncio.get_running_loop()
    last_step = {"value": -1}

    def _on_tile(step: int, total_steps: int) -> None:
        if step <= last_step["value"]:
            return
        last_step["value"] = step
        denom = max(1, int(total_steps))
        pct = 30 + int(((step + 1) / denom) * 59)
        pct = max(0, min(pct, 89))
        try:
            asyncio.run_coroutine_threadsafe(
                _update_task_status(
                    task_id,
                    TaskStatus.RUNNING,
                    progress_percent=pct,
                    current_step=f"tile {step + 1}/{denom}",
                    total_steps=denom,
                ),
                loop,
            )
        except Exception:
            return

    service_result = await asyncio.to_thread(
        lambda: asyncio.run(
            service.upscale_image(
                image=input_image,
                scale=scale,
                model_name=model_name,
                tile_size=tile_size,
                user_id=params.get("user_id"),
                progress_callback=_on_tile,
                abort_check=abort_check,
            )
        )
    )

    await _raise_if_task_terminal(task_id)

    await _update_task_status(
        task_id, TaskStatus.RUNNING, progress_percent=90, current_step="saving_results"
    )
    await _raise_if_task_terminal(task_id)

    if not isinstance(service_result, dict):
        raise TypeError(f"Unexpected upscale result type: {type(service_result)}")

    inner = service_result.get("result") if isinstance(service_result, dict) else None
    if not isinstance(inner, dict):
        inner = {}

    image_path = inner.get("image_path")
    image_url = inner.get("image_url")
    metadata_path = inner.get("metadata_path")
    width = inner.get("upscaled_width")
    height = inner.get("upscaled_height")

    try:
        width_int = int(width)  # type: ignore[arg-type]
    except Exception:
        width_int = None
    try:
        height_int = int(height)  # type: ignore[arg-type]
    except Exception:
        height_int = None

    return {
        "success": True,
        "task_id": task_id,
        "task_type": "upscale",
        "service_task_id": service_result.get("task_id"),
        "parameters": {
            "model": model_name,
            "scale": scale,
            "tile_size": tile_size,
        },
        "result": {
            "images": [
                {
                    "image_path": str(image_path) if image_path else "",
                    "image_url": str(image_url) if image_url else "",
                    "metadata_path": str(metadata_path) if metadata_path else "",
                    "width": width_int,
                    "height": height_int,
                }
            ]
            if image_url or image_path
            else [],
            "image_count": 1 if image_url or image_path else 0,
            "model_used": inner.get("model_used") or model_name,
            "processing_time": inner.get("processing_time"),
            "details": {
                "original_width": inner.get("original_width"),
                "original_height": inner.get("original_height"),
                "upscaled_width": inner.get("upscaled_width"),
                "upscaled_height": inner.get("upscaled_height"),
                "scale_factor": inner.get("scale_factor") or scale,
                "vram_used": inner.get("vram_used"),
                "device": inner.get("device"),
            },
        },
    }


@celery_app.task(bind=True, base=GPUTask, name="process_face_restore")
def process_face_restore(self, task_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Process face restoration task"""
    try:
        logger.info(f"Processing face restore task: {task_id}")

        result = asyncio.run(_process_face_restore_async(task_id, params))

        asyncio.run(
            _update_task_status(
                task_id, TaskStatus.COMPLETED, result_data=result, progress_percent=100
            )
        )
        _record_history_completion(
            history_id=task_id,
            task_type="face_restore",
            input_params=params,
            result_data=result,
        )

        return result

    except TaskCancelledError as e:
        logger.info(f"face_restore task cancelled: {task_id}")
        error_info = {"error_type": "Cancelled", "error_message": str(e)}
        asyncio.run(
            _update_task_status(
                task_id,
                TaskStatus.CANCELLED,
                error_info=error_info,
                current_step="cancelled",
            )
        )
        return {
            "success": False,
            "task_id": task_id,
            "task_type": "face_restore",
            "error": error_info,
        }

    except TaskTimeoutError as e:
        logger.info(f"face_restore task timed out (cooperative): {task_id}")
        error_info = {"error_type": "Timeout", "error_message": str(e)}
        asyncio.run(
            _update_task_status(
                task_id,
                TaskStatus.TIMEOUT,
                error_info=error_info,
                current_step="timeout",
            )
        )
        raise

    except (SoftTimeLimitExceeded, TimeLimitExceeded) as e:
        logger.error(f"Timeout processing face_restore task {task_id}: {e}")
        error_info = _build_error_info(e)
        asyncio.run(
            _update_task_status(task_id, TaskStatus.TIMEOUT, error_info=error_info)
        )
        raise

    except Exception as e:
        logger.error(f"Error processing face restore task {task_id}: {e}")

        if _is_retryable_exception(e):
            max_retries, backoff_sec = _get_retry_config(params)
            if getattr(self.request, "retries", 0) < max_retries:
                retry_index = getattr(self.request, "retries", 0)
                countdown = min(300, backoff_sec * (2**retry_index))
                error_info = _build_error_info(e)
                asyncio.run(
                    _update_task_status(
                        task_id,
                        TaskStatus.RETRYING,
                        retry_count=retry_index + 1,
                        error_info=error_info,
                        current_step="retrying",
                    )
                )
                raise self.retry(exc=e, countdown=countdown, max_retries=max_retries)

        error_info = _build_error_info(e)
        asyncio.run(_update_task_status(task_id, TaskStatus.FAILED, error_info=error_info))

        raise


async def _process_face_restore_async(
    task_id: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Async implementation of face restoration"""
    service = await _get_face_restore_service()

    await _raise_if_task_terminal(task_id)

    await _update_task_status(
        task_id, TaskStatus.RUNNING, progress_percent=10, current_step="loading_model"
    )
    await _raise_if_task_terminal(task_id)

    input_image = await _load_image_from_sources(
        base64_data=params.get("image"),
        asset_id=params.get("image_asset_id") or params.get("asset_id"),
        image_path=params.get("image_path"),
        label="face_restore image",
    )

    await _raise_if_task_terminal(task_id)

    await _update_task_status(
        task_id, TaskStatus.RUNNING, progress_percent=30, current_step="restoring_faces"
    )
    await _raise_if_task_terminal(task_id)

    model_name = params.get("model") or "GFPGAN_v1.4"
    if not isinstance(model_name, str) or not model_name.strip():
        model_name = "GFPGAN_v1.4"

    raw_upscale = params.get("upscale", 2)
    try:
        upscale = int(raw_upscale)  # type: ignore[arg-type]
    except Exception:
        upscale = 2
    upscale = max(1, min(upscale, 8))

    only_center_face = bool(params.get("only_center_face", False))
    has_aligned = bool(params.get("has_aligned", False))
    paste_back = bool(params.get("paste_back", True))
    raw_weight = params.get("weight", 0.5)
    try:
        weight = float(raw_weight)  # type: ignore[arg-type]
    except Exception:
        weight = 0.5
    weight = max(0.0, min(weight, 1.0))

    await _raise_if_task_terminal(task_id)

    loop = asyncio.get_running_loop()
    abort_check = lambda: _threadsafe_raise_if_task_terminal(task_id, loop)
    last_step = {"value": -1}

    def _on_face(step: int, total_steps: int) -> None:
        if step <= last_step["value"]:
            return
        last_step["value"] = step
        denom = max(1, int(total_steps))
        pct = 30 + int(((step + 1) / denom) * 59)
        pct = max(0, min(pct, 89))
        try:
            asyncio.run_coroutine_threadsafe(
                _update_task_status(
                    task_id,
                    TaskStatus.RUNNING,
                    progress_percent=pct,
                    current_step=f"face {step + 1}/{denom}",
                    total_steps=denom,
                ),
                loop,
            )
        except Exception:
            return

    service_result = await asyncio.to_thread(
        lambda: asyncio.run(
            service.restore_faces(
                image=input_image,
                model_name=model_name,
                upscale=upscale,
                only_center_face=only_center_face,
                has_aligned=has_aligned,
                paste_back=paste_back,
                weight=weight,
                user_id=params.get("user_id"),
                progress_callback=_on_face,
                abort_check=abort_check,
            )
        )
    )

    await _raise_if_task_terminal(task_id)

    await _update_task_status(
        task_id, TaskStatus.RUNNING, progress_percent=90, current_step="saving_results"
    )
    await _raise_if_task_terminal(task_id)

    if not isinstance(service_result, dict):
        raise TypeError(
            f"Unexpected face_restore result type: {type(service_result)}"
        )

    inner = service_result.get("result") if isinstance(service_result, dict) else None
    if not isinstance(inner, dict):
        inner = {}

    image_path = inner.get("image_path")
    image_url = inner.get("image_url")
    metadata_path = inner.get("metadata_path")
    width = inner.get("restored_width")
    height = inner.get("restored_height")

    try:
        width_int = int(width)  # type: ignore[arg-type]
    except Exception:
        width_int = None
    try:
        height_int = int(height)  # type: ignore[arg-type]
    except Exception:
        height_int = None

    return {
        "success": True,
        "task_id": task_id,
        "task_type": "face_restore",
        "service_task_id": service_result.get("task_id"),
        "parameters": {
            "model": model_name,
            "upscale": upscale,
        },
        "result": {
            "images": [
                {
                    "image_path": str(image_path) if image_path else "",
                    "image_url": str(image_url) if image_url else "",
                    "metadata_path": str(metadata_path) if metadata_path else "",
                    "width": width_int,
                    "height": height_int,
                }
            ]
            if image_url or image_path
            else [],
            "image_count": 1 if image_url or image_path else 0,
            "model_used": inner.get("model_used") or model_name,
            "processing_time": inner.get("processing_time"),
            "details": {
                "original_width": inner.get("original_width"),
                "original_height": inner.get("original_height"),
                "restored_width": inner.get("restored_width"),
                "restored_height": inner.get("restored_height"),
                "faces_detected": inner.get("faces_detected"),
                "faces_restored": inner.get("faces_restored"),
                "individual_faces": inner.get("individual_faces"),
                "vram_used": inner.get("vram_used"),
                "device": inner.get("device"),
            },
        },
    }


# =====================================
# Maintenance Tasks
# =====================================


@celery_app.task(name="cleanup_expired_tasks")
def cleanup_expired_tasks():
    """Clean up expired tasks and results"""
    try:
        logger.info("Running task cleanup...")

        # This would typically clean up old task results, temp files, etc.
        # Implementation depends on your specific cleanup requirements

        logger.info("Task cleanup completed")
        return {"status": "success", "message": "Cleanup completed"}

    except Exception as e:
        logger.error(f"Error during task cleanup: {e}")
        return {"status": "error", "message": str(e)}


# =====================================
# Periodic Task Scheduler
# =====================================

from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    "cleanup-expired-tasks": {
        "task": "cleanup_expired_tasks",
        "schedule": crontab(minute=0, hour="*/4"),  # Every 4 hours
    },
}

if __name__ == "__main__":
    # For debugging - run worker directly
    celery_app.worker_main()
