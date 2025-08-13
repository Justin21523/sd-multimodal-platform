# services/queue/task_manager.py
"""
Task Queue Manager with Celery + Redis
Handles asynchronous generation and post-processing tasks
"""

import time
import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta


# services/queue/task_manager.py (修復版本)
"""
Task Queue Manager with Celery + Redis - Fixed version
Handles asynchronous generation and post-processing tasks
"""

import time
import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

try:
    import redis
    from celery import Celery, Task
    from celery.result import AsyncResult
    from celery.exceptions import Retry, WorkerLostError

    REDIS_AVAILABLE = True
except ImportError:
    print("⚠️  Redis/Celery not installed - queue functionality disabled")
    REDIS_AVAILABLE = False

from app.config import settings
from utils.logging_utils import get_generation_logger
from utils.file_utils import save_generation_output
from utils.metadata_utils import save_generation_metadata


# Task status enumeration
class TaskStatus(str, Enum):
    PENDING = "pending"
    STARTED = "started"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILURE = "failure"
    REVOKED = "revoked"
    RETRY = "retry"


@dataclass
class TaskInfo:
    """Task information structure"""

    task_id: str
    task_type: str  # 'txt2img', 'img2img', 'postprocess', 'batch'
    status: TaskStatus
    progress: float  # 0.0 - 1.0
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key in ["created_at", "started_at", "completed_at"]:
            if data[key]:
                data[key] = data[key].isoformat()
        return data


class TaskManager:
    """Central task queue manager with Redis + Celery"""

    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True,
        )

        self.celery_app = self._create_celery_app()
        self.active_tasks: Dict[str, TaskInfo] = {}

        # Task retention settings
        self.task_retention_hours = settings.TASK_RETENTION_HOURS

    def _create_celery_app(self) -> Celery:
        """Create and configure Celery application"""
        if not self.redis_client:
            return None

        try:
            app = Celery(
                "sd_multimodal",
                broker=settings.CELERY_BROKER_URL,
                backend=settings.CELERY_RESULT_BACKEND,
                include=["services.queue.tasks"],
            )

            app.conf.update(
                task_serializer="json",
                accept_content=["json"],
                result_serializer="json",
                timezone="UTC",
                enable_utc=True,
                task_track_started=True,
                task_time_limit=settings.TASK_TIME_LIMIT,
                task_soft_time_limit=settings.TASK_SOFT_TIME_LIMIT,
                worker_max_tasks_per_child=settings.WORKER_MAX_TASKS,
                worker_prefetch_multiplier=1,
                task_acks_late=True,
                task_reject_on_worker_lost=True,
                result_expires=settings.TASK_RETENTION_HOURS * 3600,
            )

            return app

        except Exception as e:
            print(f"⚠️  Celery initialization failed: {str(e)}")
            return None

    async def submit_generation_task(
        self,
        task_type: str,
        generation_params: Dict[str, Any],
        postprocess_chain: Optional[List[str]] = None,
    ) -> str:
        """Submit image generation task to queue"""

        task_id = str(uuid.uuid4())

        # Create task info
        task_info = TaskInfo(
            task_id=task_id,
            task_type=task_type,
            status=TaskStatus.PENDING,
            progress=0.0,
            created_at=datetime.utcnow(),
            meta={"generation_params": generation_params},
        )

        # Store task info
        await self._store_task_info(task_info)

        if not self.celery_app:
            # Fallback mode - simulate task
            task_info.status = TaskStatus.FAILURE
            task_info.error_message = "Queue system not available"
            await self._store_task_info(task_info)
            return task_id

        # Submit to Celery
        try:
            if task_type == "txt2img":
                celery_task = self.celery_app.send_task(
                    "tasks.generate_txt2img",
                    args=[task_id, generation_params, postprocess_chain],
                    task_id=task_id,
                )
            elif task_type == "img2img":
                celery_task = self.celery_app.send_task(
                    "tasks.generate_img2img",
                    args=[task_id, generation_params, postprocess_chain],
                    task_id=task_id,
                )
            elif task_type == "postprocess":
                celery_task = self.celery_app.send_task(
                    "tasks.run_postprocess",
                    args=[task_id, generation_params],
                    task_id=task_id,
                )
            else:
                raise ValueError(f"Unknown task type: {task_type}")

            logger = get_generation_logger(task_type, "queue")
            logger.info(f"Submitted task {task_id} to queue")

        except Exception as e:
            # Handle Celery submission error
            task_info.status = TaskStatus.FAILURE
            task_info.error_message = f"Task submission failed: {str(e)}"
            await self._store_task_info(task_info)

        return task_id

    async def submit_batch_task(
        self,
        task_type: str,
        batch_params: List[Dict[str, Any]],
        postprocess_chain: Optional[List[str]] = None,
    ) -> str:
        """Submit batch processing task"""

        batch_id = str(uuid.uuid4())

        # Create batch task info
        batch_info = TaskInfo(
            task_id=batch_id,
            task_type="batch",
            status=TaskStatus.PENDING,
            progress=0.0,
            created_at=datetime.utcnow(),
            meta={
                "batch_type": task_type,
                "batch_size": len(batch_params),
                "subtasks": [],
            },
        )

        # Submit individual tasks
        subtask_ids = []
        for i, params in enumerate(batch_params):
            subtask_id = await self.submit_generation_task(
                task_type, params, postprocess_chain
            )
            subtask_ids.append(subtask_id)

        batch_info.meta["subtasks"] = subtask_ids  # type: ignore
        await self._store_task_info(batch_info)

        # Submit batch coordinator task if Celery available
        if self.celery_app:
            try:
                self.celery_app.send_task(
                    "tasks.coordinate_batch",
                    args=[batch_id, subtask_ids],
                    task_id=batch_id,
                )
            except Exception as e:
                batch_info.status = TaskStatus.FAILURE
                batch_info.error_message = f"Batch coordination failed: {str(e)}"
                await self._store_task_info(batch_info)

        return batch_id

    async def get_task_status(self, task_id: str) -> Optional[TaskInfo]:
        """Get current task status and progress"""

        # Check local cache first
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]

        if not self.redis_client:
            return None

        # Fetch from Redis
        try:
            task_data = self.redis_client.get(f"task:{task_id}")
            if not task_data:
                return None

            # Parse task info
            task_dict = json.loads(task_data)  # type: ignore

            # Convert datetime strings back to datetime objects
            for key in ["created_at", "started_at", "completed_at"]:
                if task_dict[key]:
                    task_dict[key] = datetime.fromisoformat(task_dict[key])

            task_info = TaskInfo(**task_dict)

            # Update from Celery if task is active
            if self.celery_app and task_info.status in [
                TaskStatus.PENDING,
                TaskStatus.STARTED,
                TaskStatus.PROCESSING,
            ]:

                try:
                    celery_result = AsyncResult(task_id, app=self.celery_app)

                    # Update status from Celery
                    if celery_result.state == "PENDING":
                        task_info.status = TaskStatus.PENDING
                    elif celery_result.state == "STARTED":
                        task_info.status = TaskStatus.STARTED
                        if not task_info.started_at:
                            task_info.started_at = datetime.utcnow()
                    elif celery_result.state == "PROGRESS":
                        task_info.status = TaskStatus.PROCESSING
                        if celery_result.info and isinstance(celery_result.info, dict):
                            task_info.progress = celery_result.info.get(
                                "progress", task_info.progress
                            )
                    elif celery_result.state == "SUCCESS":
                        task_info.status = TaskStatus.SUCCESS
                        task_info.progress = 1.0
                        task_info.completed_at = datetime.utcnow()
                        task_info.result_data = celery_result.result
                    elif celery_result.state == "FAILURE":
                        task_info.status = TaskStatus.FAILURE
                        task_info.completed_at = datetime.utcnow()
                        task_info.error_message = str(celery_result.info)
                    elif celery_result.state == "REVOKED":
                        task_info.status = TaskStatus.REVOKED
                        task_info.completed_at = datetime.utcnow()

                    # Update stored task info
                    await self._store_task_info(task_info)

                except Exception as e:
                    # Celery error - keep existing task info
                    pass

            return task_info

        except Exception as e:
            return None

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel running task"""

        task_info = await self.get_task_status(task_id)
        if not task_info:
            return False

        if task_info.status in [
            TaskStatus.SUCCESS,
            TaskStatus.FAILURE,
            TaskStatus.REVOKED,
        ]:
            return False  # Task already completed

        # Revoke Celery task if available
        if self.celery_app:
            try:
                self.celery_app.control.revoke(task_id, terminate=True)
            except Exception:
                pass

        # Update task status
        task_info.status = TaskStatus.REVOKED
        task_info.completed_at = datetime.utcnow()
        await self._store_task_info(task_info)

        logger = get_generation_logger(task_info.task_type, "queue")
        logger.info(f"Cancelled task {task_id}")

        return True

    async def list_tasks(
        self, status_filter: Optional[List[TaskStatus]] = None, limit: int = 50
    ) -> List[TaskInfo]:
        """List tasks with optional status filtering"""

        if not self.redis_client:
            # Return local tasks only
            tasks = list(self.active_tasks.values())
            if status_filter:
                tasks = [t for t in tasks if t.status in status_filter]
            return sorted(tasks, key=lambda t: t.created_at, reverse=True)[:limit]

        try:
            # Get all task keys from Redis - Fixed version
            task_keys_pattern = "task:*"
            task_keys = self.redis_client.keys(task_keys_pattern)

            # Convert to list if it's not already (Redis-py returns list, not generator)
            if hasattr(task_keys, "__iter__") and not isinstance(
                task_keys, (list, tuple)
            ):
                task_keys = list(task_keys)  # type: ignore

            tasks = []

            # Process up to limit keys
            for key in task_keys[:limit]:  # type: ignore
                try:
                    task_data = self.redis_client.get(key)
                    if task_data:
                        task_dict = json.loads(task_data)  # type: ignore

                        # Convert datetime strings
                        for dt_key in ["created_at", "started_at", "completed_at"]:
                            if task_dict.get(dt_key):
                                task_dict[dt_key] = datetime.fromisoformat(
                                    task_dict[dt_key]
                                )

                        task_info = TaskInfo(**task_dict)

                        # Apply status filter
                        if not status_filter or task_info.status in status_filter:
                            tasks.append(task_info)

                except Exception as e:
                    # Skip invalid task data
                    continue

            # Sort by creation time (newest first)
            tasks.sort(key=lambda t: t.created_at, reverse=True)

            return tasks

        except Exception as e:
            # Fallback to local tasks
            return list(self.active_tasks.values())[:limit]

    async def cleanup_old_tasks(self) -> int:
        """Clean up old completed tasks"""

        if not self.redis_client:
            return 0

        cutoff_time = datetime.utcnow() - timedelta(hours=self.task_retention_hours)
        cleaned_count = 0

        try:
            task_keys = self.redis_client.keys("task:*")

            for key in task_keys:  # type: ignore
                try:
                    task_data = self.redis_client.get(key)
                    if task_data:
                        task_dict = json.loads(task_data)  # type: ignore
                        created_at = datetime.fromisoformat(task_dict["created_at"])

                        if created_at < cutoff_time:
                            self.redis_client.delete(key)
                            cleaned_count += 1

                except Exception:
                    # Skip invalid task data
                    continue

            logger = get_generation_logger("system", "cleanup")
            logger.info(f"Cleaned up {cleaned_count} old tasks")

        except Exception as e:
            logger = get_generation_logger("system", "cleanup")
            logger.error(f"Task cleanup failed: {str(e)}")

        return cleaned_count

    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""

        base_stats = {
            "redis_connected": bool(self.redis_client),
            "celery_available": bool(self.celery_app),
            "active_workers": 0,
            "active_tasks": 0,
            "scheduled_tasks": 0,
            "reserved_tasks": 0,
            "status_distribution": {},
            "total_tasks": 0,
        }

        if not self.celery_app:
            return base_stats

        try:
            # Get Celery inspect instance
            inspect = self.celery_app.control.inspect()

            # Get active tasks
            active = inspect.active()
            scheduled = inspect.scheduled()
            reserved = inspect.reserved()

            if active:
                base_stats["active_workers"] = len(active)
                base_stats["active_tasks"] = sum(
                    len(tasks) for tasks in active.values()
                )

            if scheduled:
                base_stats["scheduled_tasks"] = sum(
                    len(tasks) for tasks in scheduled.values()
                )

            if reserved:
                base_stats["reserved_tasks"] = sum(
                    len(tasks) for tasks in reserved.values()
                )

            # Count tasks by status
            all_tasks = await self.list_tasks(limit=1000)
            status_counts = {}
            for status in TaskStatus:
                status_counts[status.value] = sum(
                    1 for t in all_tasks if t.status == status
                )

            base_stats["status_distribution"] = status_counts
            base_stats["total_tasks"] = len(all_tasks)

        except Exception as e:
            # Return basic stats if inspection fails
            pass

        return base_stats

    async def _store_task_info(self, task_info: TaskInfo):
        """Store task info in Redis or local cache"""

        if self.redis_client:
            try:
                key = f"task:{task_info.task_id}"
                data = json.dumps(task_info.to_dict())

                # Store with expiration
                self.redis_client.setex(key, self.task_retention_hours * 3600, data)
            except Exception as e:
                # Fallback to local storage
                pass

        # Cache locally for active tasks
        if task_info.status in [
            TaskStatus.PENDING,
            TaskStatus.STARTED,
            TaskStatus.PROCESSING,
        ]:
            self.active_tasks[task_info.task_id] = task_info
        elif task_info.task_id in self.active_tasks:
            # Remove from local cache when completed
            del self.active_tasks[task_info.task_id]


# Global task manager instance
_task_manager = None


def get_task_manager() -> TaskManager:
    """Get global task manager instance (singleton)"""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager
