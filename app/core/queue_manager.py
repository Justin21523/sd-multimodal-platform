# app/core/queue_manager.py
import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import uuid

from typing import Any

try:
    import redis.asyncio as redis_async

    REDIS_AVAILABLE = True
except ImportError:  # pragma: no cover
    redis_async = None  # type: ignore[assignment]
    REDIS_AVAILABLE = False
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)

# =====================================
# Task Status & Models
# =====================================


class TaskStatus(str, Enum):
    """Task status enumeration with clear lifecycle"""

    PENDING = "pending"  # Task queued, waiting for worker
    RUNNING = "running"  # Task being processed by worker
    COMPLETED = "completed"  # Task finished successfully
    FAILED = "failed"  # Task failed with error
    CANCELLED = "cancelled"  # Task cancelled by user
    RETRYING = "retrying"  # Task failed, being retried
    TIMEOUT = "timeout"  # Task exceeded time limit


class TaskPriority(str, Enum):
    """Task priority levels for queue management"""

    LOW = "low"  # 1-5 priority
    NORMAL = "normal"  # 6-7 priority
    HIGH = "high"  # 8-9 priority
    URGENT = "urgent"  # 10 priority (admin/system tasks)


@dataclass
class TaskInfo:
    """Complete task information structure"""

    task_id: str
    task_type: str  # "txt2img", "img2img", "upscale", etc.
    status: TaskStatus
    priority: TaskPriority
    user_id: Optional[str] = None

    # Timing information
    created_at: datetime = None  # type: ignore
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    estimated_duration: Optional[float] = None  # seconds

    # Progress tracking
    progress_percent: int = 0
    current_step: str = "queued"
    total_steps: Optional[int] = None

    # Task parameters and results
    input_params: Dict[str, Any] = None  # type: ignore
    result_data: Optional[Dict[str, Any]] = None
    error_info: Optional[Dict[str, Any]] = None

    # Resource usage
    gpu_memory_used: Optional[float] = None  # GB
    processing_time: Optional[float] = None  # seconds

    # Retry information
    retry_count: int = 0
    max_retries: int = 3

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.input_params is None:
            self.input_params = {}


class QueueStats(BaseModel):
    """Queue system statistics"""

    total_tasks: int = 0
    pending_tasks: int = 0
    running_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0

    average_wait_time: float = 0.0  # seconds
    average_processing_time: float = 0.0  # seconds
    queue_throughput: float = 0.0  # tasks per minute

    active_workers: int = 0
    total_workers: int = 0
    gpu_memory_usage: float = 0.0  # percentage

    # Rate limiting stats
    current_hour_requests: int = 0
    daily_requests: int = 0
    rate_limit_violations: int = 0


# =====================================
# Redis Connection & Task Storage
# =====================================


class RedisTaskStore:
    """Redis-based task storage with atomic operations"""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis_client: Optional[Any] = None
        self.task_prefix = "task:"
        self.queue_prefix = "queue:"
        self.user_tasks_prefix = "user:"
        self.user_events_suffix = ":events"
        self.stats_key = "queue:stats"

    async def _publish_user_task_event(self, task_info: TaskInfo) -> None:
        """
        Best-effort publish of a task update for SSE subscribers.

        Channel: user:{user_id}:events
        Payload: {"task_id": "..."}
        """
        if not task_info.user_id:
            return
        client = self.redis_client
        if client is None or not hasattr(client, "publish"):
            return
        try:
            channel = f"{self.user_tasks_prefix}{task_info.user_id}{self.user_events_suffix}"
            payload = json.dumps({"task_id": task_info.task_id}, ensure_ascii=False)
            await client.publish(channel, payload)  # type: ignore[attr-defined]
        except Exception:
            return

    async def connect(self):
        """Initialize Redis connection with connection pooling"""
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis client not installed. Install `redis` to enable queue features."
            )

        try:
            self.redis_client = redis_async.from_url(  # type: ignore[union-attr]
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20,
                retry_on_timeout=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )

            # Test connection
            await self.redis_client.ping()
            logger.info("Redis connection established successfully")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self):
        """Clean disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis connection closed")

    async def store_task(self, task_info: TaskInfo) -> bool:
        """Store task information with atomic operation"""
        try:
            task_key = f"{self.task_prefix}{task_info.task_id}"
            task_data = json.dumps(asdict(task_info), default=str)  # type: ignore

            # Use pipeline for atomic operation
            pipe = self.redis_client.pipeline()  # type: ignore
            pipe.hset(task_key, "data", task_data)
            pipe.expire(task_key, 86400 * 7)  # 7 days TTL

            # Add to priority queue
            priority_score = self._get_priority_score(task_info.priority)
            queue_key = f"{self.queue_prefix}{task_info.status}"
            pipe.zadd(queue_key, {task_info.task_id: priority_score})

            # Maintain a lightweight user->task index for efficient streaming/listing.
            if task_info.user_id:
                user_key = f"{self.user_tasks_prefix}{task_info.user_id}:tasks"
                try:
                    created_score = float(task_info.created_at.timestamp())
                except Exception:
                    created_score = time.time()
                pipe.zadd(user_key, {task_info.task_id: created_score})
                pipe.expire(user_key, 86400 * 7)

            await pipe.execute()
            await self._publish_user_task_event(task_info)
            logger.debug(f"Task {task_info.task_id} stored successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to store task {task_info.task_id}: {e}")
            return False

    async def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """Retrieve task information by ID"""
        try:
            task_key = f"{self.task_prefix}{task_id}"
            task_data = await self.redis_client.hget(task_key, "data")  # type: ignore

            if not task_data:
                return None

            task_dict = json.loads(task_data)

            # Normalize enum fields (stored as strings)
            if task_dict.get("status"):
                try:
                    task_dict["status"] = TaskStatus(task_dict["status"])
                except Exception:
                    pass
            if task_dict.get("priority"):
                try:
                    task_dict["priority"] = TaskPriority(task_dict["priority"])
                except Exception:
                    pass

            # Convert string dates back to datetime objects
            if task_dict.get("created_at"):
                task_dict["created_at"] = datetime.fromisoformat(
                    task_dict["created_at"]
                )
            if task_dict.get("started_at"):
                task_dict["started_at"] = datetime.fromisoformat(
                    task_dict["started_at"]
                )
            if task_dict.get("completed_at"):
                task_dict["completed_at"] = datetime.fromisoformat(
                    task_dict["completed_at"]
                )
            if task_dict.get("cancelled_at"):
                task_dict["cancelled_at"] = datetime.fromisoformat(
                    task_dict["cancelled_at"]
                )

            return TaskInfo(**task_dict)

        except Exception as e:
            logger.error(f"Failed to get task {task_id}: {e}")
            return None

    async def update_task_status(
        self, task_id: str, status: TaskStatus, **updates
    ) -> bool:
        """Update task status and other fields atomically"""
        try:
            task_info = await self.get_task(task_id)
            if not task_info:
                logger.warning(f"Task {task_id} not found for status update")
                return False

            # Update fields
            old_status = task_info.status
            terminal_statuses = {
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.TIMEOUT,
            }
            if old_status in terminal_statuses and status != old_status:
                logger.debug(
                    "Ignoring status update for terminal task",
                    extra={"task_id": task_id, "old_status": old_status, "new_status": status},
                )
                return False
            task_info.status = status

            # Set timing information based on status
            if status == TaskStatus.RUNNING and not task_info.started_at:
                task_info.started_at = datetime.now()
            elif status in [
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.TIMEOUT,
            ]:
                task_info.completed_at = datetime.now()
                if status == TaskStatus.CANCELLED and not task_info.cancelled_at:
                    task_info.cancelled_at = task_info.completed_at
                if task_info.started_at:
                    task_info.processing_time = (
                        task_info.completed_at - task_info.started_at
                    ).total_seconds()

            # Apply additional updates
            for field, value in updates.items():
                if hasattr(task_info, field):
                    setattr(task_info, field, value)

            # Store updated task
            await self.store_task(task_info)

            # Move task between queues if status changed
            if old_status != status:
                await self._move_task_between_queues(task_id, old_status, status)

            logger.debug(f"Task {task_id} status updated: {old_status} → {status}")
            return True

        except Exception as e:
            logger.error(f"Failed to update task {task_id} status: {e}")
            return False

    async def get_queue_tasks(
        self, status: TaskStatus, limit: int = 100, offset: int = 0
    ) -> List[TaskInfo]:
        """Get tasks from specific queue with pagination"""
        try:
            queue_key = f"{self.queue_prefix}{status}"

            # Get task IDs from sorted set (highest priority first)
            task_ids = await self.redis_client.zrevrange(  # type: ignore
                queue_key, offset, offset + limit - 1
            )

            # Fetch task information
            tasks = []
            for task_id in task_ids:
                task_info = await self.get_task(task_id)
                if task_info:
                    tasks.append(task_info)

            return tasks

        except Exception as e:
            logger.error(f"Failed to get queue tasks for {status}: {e}")
            return []

    async def get_user_tasks(
        self, user_id: str, limit: int = 50, offset: int = 0
    ) -> List[TaskInfo]:
        """Get all tasks for a specific user"""
        try:
            user_key = f"{self.user_tasks_prefix}{user_id}:tasks"
            task_ids = await self.redis_client.zrevrange(  # type: ignore
                user_key, offset, offset + limit - 1
            )
            tasks: List[TaskInfo] = []
            missing: List[str] = []
            for task_id in task_ids:
                task_info = await self.get_task(task_id)
                if task_info:
                    tasks.append(task_info)
                else:
                    missing.append(task_id)

            if missing:
                try:
                    pipe = self.redis_client.pipeline()  # type: ignore
                    pipe.zrem(user_key, *missing)
                    await pipe.execute()
                except Exception:
                    pass

            if tasks:
                tasks.sort(key=lambda x: x.created_at, reverse=True)
                return tasks

            # Fallback for legacy data (no user index yet).
            all_tasks: List[TaskInfo] = []
            for status in TaskStatus:
                queue_tasks = await self.get_queue_tasks(status, limit=1000)
                all_tasks.extend([t for t in queue_tasks if t.user_id == user_id])
            all_tasks.sort(key=lambda x: x.created_at, reverse=True)
            return all_tasks[offset : offset + limit]

        except Exception as e:
            logger.error(f"Failed to get user tasks for {user_id}: {e}")
            return []

    def _get_priority_score(self, priority: TaskPriority) -> float:
        """Convert priority to Redis sorted set score"""
        priority_scores = {
            TaskPriority.LOW: 1.0,
            TaskPriority.NORMAL: 5.0,
            TaskPriority.HIGH: 8.0,
            TaskPriority.URGENT: 10.0,
        }
        base_score = priority_scores.get(priority, 5.0)

        # Add timestamp component for FIFO within same priority
        timestamp_component = time.time() / 1e10  # Small contribution
        return base_score + timestamp_component

    async def _move_task_between_queues(
        self, task_id: str, old_status: TaskStatus, new_status: TaskStatus
    ):
        """Move task between different status queues"""
        try:
            old_queue = f"{self.queue_prefix}{old_status}"
            new_queue = f"{self.queue_prefix}{new_status}"

            # Get current score (priority)
            score = await self.redis_client.zscore(old_queue, task_id)  # type: ignore
            if score is None:
                score = 5.0  # Default score

            # Atomic move operation
            pipe = self.redis_client.pipeline()  # type: ignore
            pipe.zrem(old_queue, task_id)
            pipe.zadd(new_queue, {task_id: score})
            await pipe.execute()

        except Exception as e:
            logger.error(f"Failed to move task {task_id} between queues: {e}")


# =====================================
# Queue Manager - Main Controller
# =====================================


class QueueManager:
    """Main queue management system with rate limiting and concurrency control"""

    def __init__(self):
        self.task_store: Optional[RedisTaskStore] = None
        self.is_initialized = False

        # Concurrency control
        self.max_concurrent_tasks = settings.MAX_CONCURRENT_TASKS
        self.current_running_tasks: Set[str] = set()
        self.task_semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

        # Rate limiting (requests per hour per user)
        self.rate_limit_requests_per_hour = settings.RATE_LIMIT_PER_HOUR
        self.user_request_counts: Dict[str, List[datetime]] = {}

        # Statistics tracking
        self.stats = QueueStats()
        self.stats_update_interval = 60  # seconds
        self._stats_task: Optional[asyncio.Task] = None

    async def initialize(self):
        """Initialize queue manager with Redis connection"""
        try:
            redis_url = settings.get_redis_url()
            self.task_store = RedisTaskStore(redis_url)
            await self.task_store.connect()

            # Start background statistics updates
            self._stats_task = asyncio.create_task(self._update_stats_periodically())

            self.is_initialized = True
            logger.info("Queue manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize queue manager: {e}")
            raise

    async def shutdown(self):
        """Graceful shutdown with cleanup"""
        try:
            if self._stats_task:
                self._stats_task.cancel()
                try:
                    await self._stats_task
                except asyncio.CancelledError:
                    pass

            if self.task_store:
                await self.task_store.disconnect()

            logger.info("Queue manager shutdown completed")

        except Exception as e:
            logger.error(f"Error during queue manager shutdown: {e}")

    async def enqueue_task(
        self,
        task_type: str,
        input_params: Dict[str, Any],
        user_id: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> Optional[str]:
        """Enqueue a new task with rate limiting check"""
        try:
            # Rate limiting check
            if user_id and not await self._check_rate_limit(user_id):
                logger.warning(f"Rate limit exceeded for user {user_id}")
                return None

            # Create unique task ID
            task_id = f"{task_type}_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"

            # Create task info
            task_info = TaskInfo(
                task_id=task_id,
                task_type=task_type,
                status=TaskStatus.PENDING,
                priority=priority,
                user_id=user_id,
                input_params=input_params,
                estimated_duration=self._estimate_task_duration(
                    task_type, input_params
                ),
            )

            # Store task
            success = await self.task_store.store_task(task_info)  # type: ignore
            if success:
                logger.info(f"Task {task_id} enqueued successfully")
                return task_id
            else:
                logger.error(f"Failed to enqueue task {task_id}")
                return None

        except Exception as e:
            logger.error(f"Error enqueuing task: {e}")
            return None

    async def get_task_status(self, task_id: str) -> Optional[TaskInfo]:
        """Get current status of a task"""
        return await self.task_store.get_task(task_id)  # type: ignore

    async def cancel_task(self, task_id: str, user_id: Optional[str] = None) -> bool:
        """Cancel a task if it belongs to the user and is cancellable"""
        try:
            task_info = await self.task_store.get_task(task_id)  # type: ignore
            if not task_info:
                logger.warning(f"Task {task_id} not found for cancellation")
                return False

            # Authorization check
            if user_id and task_info.user_id != user_id:
                logger.warning(
                    f"User {user_id} not authorized to cancel task {task_id}"
                )
                return False

            # Check if task can be cancelled
            if task_info.status in [
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.TIMEOUT,
            ]:
                logger.warning(
                    f"Task {task_id} cannot be cancelled (status: {task_info.status})"
                )
                return False

            # Cancel task
            success = await self.task_store.update_task_status(  # type: ignore
                task_id,
                TaskStatus.CANCELLED,
                cancelled_at=datetime.now(),
                current_step="cancelled",
                error_info={"error_type": "Cancelled", "error_message": "Task cancelled"},
            )

            if success:
                # Remove from running tasks if applicable
                self.current_running_tasks.discard(task_id)
                logger.info(f"Task {task_id} cancelled successfully")

            return success

        except Exception as e:
            logger.error(f"Error cancelling task {task_id}: {e}")
            return False

    async def get_queue_status(self) -> QueueStats:
        """Get current queue statistics"""
        await self._update_queue_stats()
        return self.stats

    async def get_user_tasks(
        self, user_id: str, limit: int = 50, offset: int = 0
    ) -> List[TaskInfo]:
        """Get all tasks for a specific user"""
        return await self.task_store.get_user_tasks(user_id, limit, offset)  # type: ignore

    async def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user has exceeded rate limit"""
        try:
            current_time = datetime.now()
            hour_ago = current_time - timedelta(hours=1)

            # Initialize user request history if not exists
            if user_id not in self.user_request_counts:
                self.user_request_counts[user_id] = []

            # Clean old requests (older than 1 hour)
            user_requests = self.user_request_counts[user_id]
            user_requests[:] = [
                req_time for req_time in user_requests if req_time > hour_ago
            ]

            # Check rate limit
            if len(user_requests) >= self.rate_limit_requests_per_hour:
                self.stats.rate_limit_violations += 1
                return False

            # Add current request
            user_requests.append(current_time)
            return True

        except Exception as e:
            logger.error(f"Error checking rate limit for user {user_id}: {e}")
            return True  # Allow on error to avoid blocking legitimate users

    def _estimate_task_duration(self, task_type: str, params: Dict[str, Any]) -> float:
        """Estimate task duration based on type and parameters"""
        base_durations = {
            "txt2img": 15.0,  # seconds
            "img2img": 12.0,
            "inpaint": 18.0,
            "upscale": 8.0,
            "face_restore": 5.0,
            "video_animate": 120.0,
        }

        base_duration = base_durations.get(task_type, 20.0)

        # Adjust based on parameters
        steps = params.get("steps", params.get("num_inference_steps", 20))
        try:
            steps_int = int(steps)
        except Exception:
            steps_int = 20
        if steps_int > 30:
            base_duration *= 1.5
        try:
            width = int(params.get("width", 512) or 512)
        except Exception:
            width = 512
        try:
            height = int(params.get("height", 512) or 512)
        except Exception:
            height = 512
        if width * height > 1024 * 1024:
            base_duration *= 1.3

        return base_duration

    async def _update_stats_periodically(self):
        """Background task to update statistics"""
        while True:
            try:
                await asyncio.sleep(self.stats_update_interval)
                await self._update_queue_stats()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error updating queue stats: {e}")

    async def _update_queue_stats(self):
        """Update queue statistics"""
        try:
            # Count tasks by status
            stats = QueueStats()

            for status in TaskStatus:
                tasks = await self.task_store.get_queue_tasks(status, limit=10000)  # type: ignore
                task_count = len(tasks)

                if status == TaskStatus.PENDING:
                    stats.pending_tasks = task_count
                elif status == TaskStatus.RUNNING:
                    stats.running_tasks = task_count
                elif status == TaskStatus.COMPLETED:
                    stats.completed_tasks = task_count
                elif status == TaskStatus.FAILED:
                    stats.failed_tasks = task_count
                elif status == TaskStatus.CANCELLED:
                    stats.cancelled_tasks = task_count

                stats.total_tasks += task_count

            # Calculate timing statistics
            recent_completed = await self.task_store.get_queue_tasks(  # type: ignore
                TaskStatus.COMPLETED, limit=100
            )

            if recent_completed:
                wait_times = []
                processing_times = []

                for task in recent_completed:
                    if task.started_at and task.created_at:
                        wait_time = (task.started_at - task.created_at).total_seconds()
                        wait_times.append(wait_time)

                    if task.processing_time:
                        processing_times.append(task.processing_time)

                if wait_times:
                    stats.average_wait_time = sum(wait_times) / len(wait_times)
                if processing_times:
                    stats.average_processing_time = sum(processing_times) / len(
                        processing_times
                    )

            # Calculate throughput (tasks per minute)
            one_hour_ago = datetime.now() - timedelta(hours=1)
            recent_tasks = [
                t
                for t in recent_completed
                if t.completed_at and t.completed_at > one_hour_ago
            ]
            stats.queue_throughput = len(recent_tasks) / 60  # Convert to per minute

            # Worker information
            stats.active_workers = len(self.current_running_tasks)
            stats.total_workers = self.max_concurrent_tasks

            self.stats = stats

        except Exception as e:
            logger.error(f"Error updating queue statistics: {e}")


# =====================================
# Global Queue Manager Instance
# =====================================

_queue_manager: Optional[QueueManager] = None


async def get_queue_manager() -> QueueManager:
    """Get global queue manager instance (singleton pattern)"""
    global _queue_manager
    if _queue_manager is None:
        _queue_manager = QueueManager()
        await _queue_manager.initialize()
    return _queue_manager


async def shutdown_queue_manager():
    """Shutdown global queue manager"""
    global _queue_manager
    if _queue_manager:
        await _queue_manager.shutdown()
        _queue_manager = None
