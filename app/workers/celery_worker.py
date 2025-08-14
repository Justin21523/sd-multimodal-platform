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
from PIL import Image
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import settings
from app.core.queue_manager import TaskStatus, TaskInfo, QueueManager, get_queue_manager

from app.services.generation.img2img_service import Img2ImgService
from app.services.postprocess.upscale_service import UpscaleService
from app.services.postprocess.face_restore_service import FaceRestoreService
from app.services.generation.txt2img_service import Txt2ImgService

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
        self.txt2img_service: Optional[Txt2ImgService] = None
        self.img2img_service: Optional[Img2ImgService] = None
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
        if worker_state.txt2img_service:
            asyncio.run(worker_state.txt2img_service.cleanup())
        if worker_state.img2img_service:
            asyncio.run(worker_state.img2img_service.cleanup())
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


async def _get_txt2img_service() -> Txt2ImgService:
    """Get or initialize txt2img service"""
    if worker_state.txt2img_service is None:
        worker_state.txt2img_service = Txt2ImgService()
        await worker_state.txt2img_service.initialize()  # type: ignore
    return worker_state.txt2img_service


async def _get_img2img_service() -> Img2ImgService:
    """Get or initialize img2img service"""
    if worker_state.img2img_service is None:
        worker_state.img2img_service = Img2ImgService()
        await worker_state.img2img_service.initialize()  # type: ignore
    return worker_state.img2img_service


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
        await queue_manager.task_store.update_task_status(task_id, status, **updates)  # type: ignore
    except Exception as e:
        logger.error(f"Failed to update task status for {task_id}: {e}")


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

        return result

    except Exception as e:
        logger.error(f"Error processing txt2img task {task_id}: {e}")

        # Update task status on failure
        error_info = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        }
        asyncio.run(
            _update_task_status(task_id, TaskStatus.FAILED, error_info=error_info)
        )

        raise


async def _process_txt2img_async(
    task_id: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Async implementation of txt2img processing"""
    service = await _get_txt2img_service()

    # Update progress
    await _update_task_status(
        task_id, TaskStatus.RUNNING, progress_percent=10, current_step="loading_model"
    )

    # Generate image
    await _update_task_status(
        task_id, TaskStatus.RUNNING, progress_percent=30, current_step="generating"
    )

    result = await service.generate_image(
        prompt=params.get("prompt", ""),
        negative_prompt=params.get("negative_prompt", ""),
        width=params.get("width", 1024),
        height=params.get("height", 1024),
        num_inference_steps=params.get("steps", 25),
        guidance_scale=params.get("cfg_scale", 7.5),
        seed=params.get("seed"),
    )

    await _update_task_status(
        task_id, TaskStatus.RUNNING, progress_percent=90, current_step="saving_results"
    )

    return result


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

        return result

    except Exception as e:
        logger.error(f"Error processing img2img task {task_id}: {e}")

        error_info = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        }
        asyncio.run(
            _update_task_status(task_id, TaskStatus.FAILED, error_info=error_info)
        )

        raise


async def _process_img2img_async(
    task_id: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Async implementation of img2img processing"""
    service = await _get_img2img_service()

    await _update_task_status(
        task_id, TaskStatus.RUNNING, progress_percent=10, current_step="loading_model"
    )

    # Load input image
    input_image_path = params.get("image_path")
    if not input_image_path:
        raise ValueError("Input image path is required for img2img")

    input_image = Image.open(input_image_path)

    await _update_task_status(
        task_id, TaskStatus.RUNNING, progress_percent=30, current_step="generating"
    )

    result = await service.generate_image(
        prompt=params.get("prompt", ""),
        image=input_image,
        strength=params.get("strength", 0.8),
        negative_prompt=params.get("negative_prompt", ""),
        num_inference_steps=params.get("steps", 25),
        guidance_scale=params.get("cfg_scale", 7.5),
        seed=params.get("seed"),
    )

    await _update_task_status(
        task_id, TaskStatus.RUNNING, progress_percent=90, current_step="saving_results"
    )

    return result


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

        return result

    except Exception as e:
        logger.error(f"Error processing upscale task {task_id}: {e}")

        error_info = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        }
        asyncio.run(
            _update_task_status(task_id, TaskStatus.FAILED, error_info=error_info)
        )

        raise


async def _process_upscale_async(
    task_id: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Async implementation of upscaling"""
    service = await _get_upscale_service()

    await _update_task_status(
        task_id, TaskStatus.RUNNING, progress_percent=10, current_step="loading_model"
    )

    # Load input image
    input_image_path = params.get("image_path")
    if not input_image_path:
        raise ValueError("Input image path is required for upscaling")

    input_image = Image.open(input_image_path)

    await _update_task_status(
        task_id, TaskStatus.RUNNING, progress_percent=30, current_step="upscaling"
    )

    result = await service.upscale_image(
        image=input_image,
        scale=params.get("scale", 4),
        model_name=params.get("model", "RealESRGAN_x4plus"),
    )

    await _update_task_status(
        task_id, TaskStatus.RUNNING, progress_percent=90, current_step="saving_results"
    )

    return result


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

        return result

    except Exception as e:
        logger.error(f"Error processing face restore task {task_id}: {e}")

        error_info = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        }
        asyncio.run(
            _update_task_status(task_id, TaskStatus.FAILED, error_info=error_info)
        )

        raise


async def _process_face_restore_async(
    task_id: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Async implementation of face restoration"""
    service = await _get_face_restore_service()

    await _update_task_status(
        task_id, TaskStatus.RUNNING, progress_percent=10, current_step="loading_model"
    )

    # Load input image
    input_image_path = params.get("image_path")
    if not input_image_path:
        raise ValueError("Input image path is required for face restoration")

    input_image = Image.open(input_image_path)

    await _update_task_status(
        task_id, TaskStatus.RUNNING, progress_percent=30, current_step="restoring_faces"
    )

    result = await service.restore_faces(
        image=input_image,
        model_name=params.get("model", "GFPGAN"),
        upscale=params.get("upscale", 2),
    )

    await _update_task_status(
        task_id, TaskStatus.RUNNING, progress_percent=90, current_step="saving_results"
    )

    return result


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
