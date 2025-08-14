# tests/test_phase6_queue_system.py
import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.queue_manager import (
    QueueManager,
    TaskInfo,
    TaskStatus,
    TaskPriority,
    QueueStats,
    RedisTaskStore,
)
from app.config import get_testing_settings


class TestRedisTaskStore:
    """Test Redis task storage functionality"""

    @pytest.fixture
    async def task_store(self):
        """Create test task store"""
        settings = get_testing_settings()
        store = RedisTaskStore(settings.get_redis_url())
        await store.connect()
        yield store
        await store.disconnect()

    @pytest.fixture
    def sample_task(self):
        """Create sample task for testing"""
        return TaskInfo(
            task_id="test_task_123",
            task_type="txt2img",
            status=TaskStatus.PENDING,
            priority=TaskPriority.NORMAL,
            user_id="test_user",
            input_params={"prompt": "test prompt", "steps": 20},
        )

    async def test_store_and_retrieve_task(self, task_store, sample_task):
        """Test task storage and retrieval"""
        # Store task
        success = await task_store.store_task(sample_task)
        assert success, "Task storage should succeed"

        # Retrieve task
        retrieved_task = await task_store.get_task(sample_task.task_id)
        assert retrieved_task is not None, "Task should be retrievable"
        assert retrieved_task.task_id == sample_task.task_id
        assert retrieved_task.task_type == sample_task.task_type
        assert retrieved_task.status == sample_task.status

    async def test_update_task_status(self, task_store, sample_task):
        """Test task status updates"""
        # Store initial task
        await task_store.store_task(sample_task)

        # Update status to RUNNING
        success = await task_store.update_task_status(
            sample_task.task_id, TaskStatus.RUNNING, progress_percent=50
        )
        assert success, "Status update should succeed"

        # Verify update
        updated_task = await task_store.get_task(sample_task.task_id)
        assert updated_task.status == TaskStatus.RUNNING
        assert updated_task.progress_percent == 50
        assert updated_task.started_at is not None

    async def test_queue_operations(self, task_store, sample_task):
        """Test queue listing functionality"""
        # Store task
        await task_store.store_task(sample_task)

        # Get pending tasks
        pending_tasks = await task_store.get_queue_tasks(TaskStatus.PENDING, limit=10)

        task_ids = [t.task_id for t in pending_tasks]
        assert sample_task.task_id in task_ids, "Task should appear in pending queue"

    async def test_user_task_filtering(self, task_store):
        """Test user-specific task retrieval"""
        # Create tasks for different users
        task1 = TaskInfo(
            task_id="user1_task",
            task_type="txt2img",
            status=TaskStatus.PENDING,
            priority=TaskPriority.NORMAL,
            user_id="user1",
        )
        task2 = TaskInfo(
            task_id="user2_task",
            task_type="img2img",
            status=TaskStatus.RUNNING,
            priority=TaskPriority.HIGH,
            user_id="user2",
        )

        await task_store.store_task(task1)
        await task_store.store_task(task2)

        # Get tasks for user1
        user1_tasks = await task_store.get_user_tasks("user1", limit=10)
        user1_task_ids = [t.task_id for t in user1_tasks]

        assert "user1_task" in user1_task_ids
        assert "user2_task" not in user1_task_ids


class TestQueueManager:
    """Test queue manager functionality"""

    @pytest.fixture
    async def queue_manager(self):
        """Create test queue manager"""
        with patch("app.core.queue_manager.RedisTaskStore") as mock_store_class:
            # Mock Redis task store
            mock_store = AsyncMock()
            mock_store_class.return_value = mock_store

            manager = QueueManager()
            manager.task_store = mock_store
            manager.is_initialized = True

            yield manager

            if manager._stats_task:
                manager._stats_task.cancel()

    async def test_enqueue_task_success(self, queue_manager):
        """Test successful task enqueuing"""
        # Mock store_task to return success
        queue_manager.task_store.store_task.return_value = True

        task_id = await queue_manager.enqueue_task(
            task_type="txt2img",
            input_params={"prompt": "test", "steps": 20},
            user_id="test_user",
            priority=TaskPriority.NORMAL,
        )

        assert task_id is not None, "Task ID should be returned"
        assert task_id.startswith("txt2img_"), "Task ID should have correct prefix"

        # Verify store_task was called
        queue_manager.task_store.store_task.assert_called_once()

    async def test_rate_limiting(self, queue_manager):
        """Test rate limiting functionality"""
        # Set low rate limit for testing
        queue_manager.rate_limit_requests_per_hour = 2

        # First request should succeed
        task_id1 = await queue_manager.enqueue_task(
            task_type="txt2img", input_params={"prompt": "test1"}, user_id="test_user"
        )
        assert task_id1 is not None

        # Second request should succeed
        task_id2 = await queue_manager.enqueue_task(
            task_type="txt2img", input_params={"prompt": "test2"}, user_id="test_user"
        )
        assert task_id2 is not None

        # Third request should fail due to rate limit
        task_id3 = await queue_manager.enqueue_task(
            task_type="txt2img", input_params={"prompt": "test3"}, user_id="test_user"
        )
        assert task_id3 is None, "Third request should be rate limited"

    async def test_cancel_task_success(self, queue_manager):
        """Test successful task cancellation"""
        # Mock task retrieval
        mock_task = TaskInfo(
            task_id="test_task",
            task_type="txt2img",
            status=TaskStatus.PENDING,
            priority=TaskPriority.NORMAL,
            user_id="test_user",
        )
        queue_manager.task_store.get_task.return_value = mock_task
        queue_manager.task_store.update_task_status.return_value = True

        success = await queue_manager.cancel_task("test_task", "test_user")
        assert success, "Cancellation should succeed"

        # Verify status update was called
        queue_manager.task_store.update_task_status.assert_called_once_with(
            "test_task",
            TaskStatus.CANCELLED,
            cancelled_at=pytest.approx(datetime.now(), abs=timedelta(seconds=5)),
        )

    async def test_cancel_task_unauthorized(self, queue_manager):
        """Test unauthorized task cancellation"""
        # Mock task with different user
        mock_task = TaskInfo(
            task_id="test_task",
            task_type="txt2img",
            status=TaskStatus.PENDING,
            priority=TaskPriority.NORMAL,
            user_id="other_user",  # Different user
        )
        queue_manager.task_store.get_task.return_value = mock_task

        success = await queue_manager.cancel_task("test_task", "test_user")
        assert not success, "Cancellation should fail for unauthorized user"

    async def test_cancel_completed_task(self, queue_manager):
        """Test cancellation of already completed task"""
        # Mock completed task
        mock_task = TaskInfo(
            task_id="test_task",
            task_type="txt2img",
            status=TaskStatus.COMPLETED,  # Already completed
            priority=TaskPriority.NORMAL,
            user_id="test_user",
        )
        queue_manager.task_store.get_task.return_value = mock_task

        success = await queue_manager.cancel_task("test_task", "test_user")
        assert not success, "Cannot cancel completed task"


class TestTaskStatusFlow:
    """Test complete task lifecycle"""

    async def test_task_lifecycle(self):
        """Test complete task status flow"""
        task = TaskInfo(
            task_id="lifecycle_test",
            task_type="txt2img",
            status=TaskStatus.PENDING,
            priority=TaskPriority.NORMAL,
        )

        # PENDING -> RUNNING
        assert task.status == TaskStatus.PENDING
        assert task.started_at is None

        # Simulate task start
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        task.progress_percent = 25

        assert task.status == TaskStatus.RUNNING
        assert task.started_at is not None

        # Simulate completion
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        task.progress_percent = 100

        if task.started_at and task.completed_at:
            task.processing_time = (task.completed_at - task.started_at).total_seconds()

        assert task.status == TaskStatus.COMPLETED
        assert task.processing_time is not None
        assert task.progress_percent == 100


class TestPerformance:
    """Performance and load testing"""

    @pytest.mark.asyncio
    async def test_concurrent_enqueue_performance(self):
        """Test performance under concurrent load"""
        with patch("app.core.queue_manager.RedisTaskStore") as mock_store_class:
            mock_store = AsyncMock()
            mock_store.store_task.return_value = True
            mock_store_class.return_value = mock_store

            manager = QueueManager()
            manager.task_store = mock_store
            manager.is_initialized = True

            # Test concurrent enqueuing
            async def enqueue_task(i):
                return await manager.enqueue_task(
                    task_type="txt2img",
                    input_params={"prompt": f"test prompt {i}"},
                    user_id=f"user_{i % 10}",  # 10 different users
                    priority=TaskPriority.NORMAL,
                )

            start_time = time.time()

            # Enqueue 100 tasks concurrently
            tasks = [enqueue_task(i) for i in range(100)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            end_time = time.time()
            duration = end_time - start_time

            # Check results
            successful_tasks = [
                r for r in results if isinstance(r, str) and r is not None
            ]
            failed_tasks = [r for r in results if r is None or isinstance(r, Exception)]

            print(f"Enqueued 100 tasks in {duration:.2f} seconds")
            print(f"Successful: {len(successful_tasks)}, Failed: {len(failed_tasks)}")

            # Performance assertions
            assert duration < 5.0, "Should complete within 5 seconds"
            assert len(successful_tasks) >= 90, "At least 90% should succeed"

    async def test_memory_usage_pattern(self):
        """Test memory usage patterns"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create many task objects
        tasks = []
        for i in range(1000):
            task = TaskInfo(
                task_id=f"memory_test_{i}",
                task_type="txt2img",
                status=TaskStatus.PENDING,
                priority=TaskPriority.NORMAL,
                input_params={"prompt": f"test {i}" * 100},  # Large params
            )
            tasks.append(task)

        peak_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Clear tasks
        tasks.clear()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        print(
            f"Memory usage: Initial: {initial_memory:.1f}MB, "
            f"Peak: {peak_memory:.1f}MB, Final: {final_memory:.1f}MB"
        )

        # Memory should not grow excessively
        memory_growth = peak_memory - initial_memory
        memory_cleanup = peak_memory - final_memory

        assert memory_growth < 500, "Memory growth should be reasonable"
        assert memory_cleanup > memory_growth * 0.5, "Memory should be cleaned up"


# =====================================
# Integration Tests
# =====================================


class TestAPIIntegration:
    """Test API integration with queue system"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        from app.main import app

        with TestClient(app) as client:
            yield client

    def test_enqueue_via_api(self, client):
        """Test task enqueuing via API"""
        with patch("app.api.v1.queue.get_queue_manager") as mock_get_manager:
            # Mock queue manager
            mock_manager = AsyncMock()
            mock_manager.enqueue_task.return_value = "test_task_123"
            mock_get_manager.return_value = mock_manager

            response = client.post(
                "/api/v1/queue/enqueue",
                json={
                    "task_type": "txt2img",
                    "parameters": {"prompt": "test prompt", "steps": 20},
                    "priority": "normal",
                    "user_id": "test_user",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["task_id"] == "test_task_123"

    def test_get_task_status_via_api(self, client):
        """Test task status retrieval via API"""
        with patch("app.api.v1.queue.get_queue_manager") as mock_get_manager:
            # Mock queue manager
            mock_manager = AsyncMock()
            mock_task = TaskInfo(
                task_id="test_task_123",
                task_type="txt2img",
                status=TaskStatus.RUNNING,
                priority=TaskPriority.NORMAL,
                progress_percent=50,
            )
            mock_manager.get_task_status.return_value = mock_task
            mock_get_manager.return_value = mock_manager

            response = client.get("/api/v1/queue/status/test_task_123")

            assert response.status_code == 200
            data = response.json()
            assert data["task_id"] == "test_task_123"
            assert data["status"] == "running"
            assert data["progress_percent"] == 50


# =====================================
# Benchmark Tests
# =====================================


@pytest.mark.benchmark
class TestBenchmarks:
    """Benchmark tests for performance measurement"""

    def test_task_creation_benchmark(self, benchmark):
        """Benchmark task creation performance"""

        def create_task():
            return TaskInfo(
                task_id="benchmark_task",
                task_type="txt2img",
                status=TaskStatus.PENDING,
                priority=TaskPriority.NORMAL,
                input_params={"prompt": "benchmark test", "steps": 25},
            )

        result = benchmark(create_task)
        assert result.task_id == "benchmark_task"

    @pytest.mark.asyncio
    async def test_redis_operations_benchmark(self, benchmark):
        """Benchmark Redis operations"""
        settings = get_testing_settings()

        async def redis_roundtrip():
            store = RedisTaskStore(settings.get_redis_url())
            await store.connect()

            task = TaskInfo(
                task_id="benchmark_redis",
                task_type="txt2img",
                status=TaskStatus.PENDING,
                priority=TaskPriority.NORMAL,
            )

            # Store and retrieve
            await store.store_task(task)
            retrieved = await store.get_task(task.task_id)

            await store.disconnect()
            return retrieved

        result = await benchmark(redis_roundtrip)
        assert result.task_id == "benchmark_redis"


# =====================================
# Test Configuration
# =====================================


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "benchmark: mark test as benchmark test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])
