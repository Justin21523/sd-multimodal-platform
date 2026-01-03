# tests/test_queue_manager_unit.py
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.core import queue_manager as qm
from app.core.queue_manager import (
    QueueManager,
    RedisTaskStore,
    TaskInfo,
    TaskPriority,
    TaskStatus,
)


class FakePipeline:
    def __init__(self, redis: "FakeRedis"):
        self.redis = redis
        self.ops: list[tuple] = []

    def hset(self, key: str, field: str, value: str):
        self.ops.append(("hset", key, field, value))
        return self

    def expire(self, key: str, ttl: int):
        self.ops.append(("expire", key, ttl))
        return self

    def zadd(self, key: str, mapping: dict[str, float]):
        self.ops.append(("zadd", key, mapping))
        return self

    def zrem(self, key: str, member: str):
        self.ops.append(("zrem", key, member))
        return self

    async def execute(self):
        for op in self.ops:
            if op[0] == "hset":
                _, key, field, value = op
                self.redis.hashes.setdefault(key, {})[field] = value
            elif op[0] == "expire":
                continue
            elif op[0] == "zadd":
                _, key, mapping = op
                zset = self.redis.sorted_sets.setdefault(key, {})
                zset.update(mapping)
            elif op[0] == "zrem":
                _, key, member = op
                self.redis.sorted_sets.get(key, {}).pop(member, None)
        self.ops.clear()
        return True


class FakeRedis:
    def __init__(self):
        self.hashes: dict[str, dict[str, str]] = {}
        self.sorted_sets: dict[str, dict[str, float]] = {}
        self.closed = False

    async def ping(self):
        return True

    async def close(self):
        self.closed = True

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self)

    async def hget(self, key: str, field: str):
        return self.hashes.get(key, {}).get(field)

    async def zscore(self, key: str, member: str):
        return self.sorted_sets.get(key, {}).get(member)

    async def zrevrange(self, key: str, start: int, end: int):
        members = self.sorted_sets.get(key, {})
        ordered = sorted(members.items(), key=lambda kv: kv[1], reverse=True)
        # Redis ZREVRANGE end is inclusive
        return [m for m, _ in ordered[start : end + 1]]


@pytest.mark.unit
class TestRedisTaskStoreWithMockRedis:
    async def test_connect_and_disconnect(self):
        store = RedisTaskStore("redis://example.invalid/0")
        fake = FakeRedis()

        with patch.object(qm, "REDIS_AVAILABLE", True), patch.object(
            qm, "redis_async", SimpleNamespace(from_url=lambda *args, **kwargs: fake)
        ):
            await store.connect()
            assert store.redis_client is fake
            await store.disconnect()
            assert fake.closed is True

    async def test_store_and_get_task_roundtrip(self):
        store = RedisTaskStore("redis://example.invalid/0")
        fake = FakeRedis()

        with patch.object(qm, "REDIS_AVAILABLE", True), patch.object(
            qm, "redis_async", SimpleNamespace(from_url=lambda *args, **kwargs: fake)
        ):
            await store.connect()

        created_at = datetime(2020, 1, 1, 0, 0, 0)
        task = TaskInfo(
            task_id="task_1",
            task_type="txt2img",
            status=TaskStatus.PENDING,
            priority=TaskPriority.NORMAL,
            user_id="user_1",
            created_at=created_at,
            input_params={"prompt": "test", "steps": 20},
        )

        ok = await store.store_task(task)
        assert ok is True

        loaded = await store.get_task("task_1")
        assert loaded is not None
        assert loaded.task_id == "task_1"
        assert loaded.status == TaskStatus.PENDING
        assert loaded.priority == TaskPriority.NORMAL
        assert loaded.created_at == created_at

    async def test_get_task_returns_none_when_missing(self):
        store = RedisTaskStore("redis://example.invalid/0")
        fake = FakeRedis()
        with patch.object(qm, "REDIS_AVAILABLE", True), patch.object(
            qm, "redis_async", SimpleNamespace(from_url=lambda *args, **kwargs: fake)
        ):
            await store.connect()

        assert await store.get_task("missing") is None

    async def test_update_task_status_sets_timestamps_and_blocks_terminal(self):
        store = RedisTaskStore("redis://example.invalid/0")
        fake = FakeRedis()

        with patch.object(qm, "REDIS_AVAILABLE", True), patch.object(
            qm, "redis_async", SimpleNamespace(from_url=lambda *args, **kwargs: fake)
        ):
            await store.connect()

        task = TaskInfo(
            task_id="task_2",
            task_type="txt2img",
            status=TaskStatus.PENDING,
            priority=TaskPriority.NORMAL,
            user_id="user_1",
            input_params={"prompt": "test"},
        )
        assert await store.store_task(task) is True

        # PENDING -> RUNNING should set started_at
        ok = await store.update_task_status("task_2", TaskStatus.RUNNING, progress_percent=10)
        assert ok is True
        running = await store.get_task("task_2")
        assert running is not None
        assert running.status == TaskStatus.RUNNING
        assert running.started_at is not None
        assert running.progress_percent == 10

        # RUNNING -> COMPLETED should set completed_at and processing_time
        ok = await store.update_task_status("task_2", TaskStatus.COMPLETED, result_data={"ok": True})
        assert ok is True
        completed = await store.get_task("task_2")
        assert completed is not None
        assert completed.status == TaskStatus.COMPLETED
        assert completed.completed_at is not None
        assert completed.processing_time is not None
        assert completed.result_data == {"ok": True}

        # Terminal tasks should not accept status changes
        ok = await store.update_task_status("task_2", TaskStatus.RUNNING)
        assert ok is False

    async def test_get_queue_tasks_orders_by_score(self):
        store = RedisTaskStore("redis://example.invalid/0")
        fake = FakeRedis()

        with patch.object(qm, "REDIS_AVAILABLE", True), patch.object(
            qm, "redis_async", SimpleNamespace(from_url=lambda *args, **kwargs: fake)
        ), patch("app.core.queue_manager.time.time", return_value=1000.0):
            await store.connect()

            low = TaskInfo(
                task_id="low_1",
                task_type="txt2img",
                status=TaskStatus.PENDING,
                priority=TaskPriority.LOW,
                user_id="user_1",
            )
            high = TaskInfo(
                task_id="high_1",
                task_type="txt2img",
                status=TaskStatus.PENDING,
                priority=TaskPriority.URGENT,
                user_id="user_1",
            )
            assert await store.store_task(low) is True
            assert await store.store_task(high) is True

            tasks = await store.get_queue_tasks(TaskStatus.PENDING, limit=10)
            task_ids = [t.task_id for t in tasks]
            assert task_ids[0] == "high_1"
            assert task_ids[1] == "low_1"

    async def test_get_user_tasks_aggregates_across_statuses(self):
        store = RedisTaskStore("redis://example.invalid/0")
        fake = FakeRedis()

        with patch.object(qm, "REDIS_AVAILABLE", True), patch.object(
            qm, "redis_async", SimpleNamespace(from_url=lambda *args, **kwargs: fake)
        ):
            await store.connect()

        now = datetime.now()
        older = TaskInfo(
            task_id="u1_old",
            task_type="txt2img",
            status=TaskStatus.PENDING,
            priority=TaskPriority.NORMAL,
            user_id="user_1",
            created_at=now - timedelta(days=1),
        )
        newer = TaskInfo(
            task_id="u1_new",
            task_type="txt2img",
            status=TaskStatus.RUNNING,
            priority=TaskPriority.NORMAL,
            user_id="user_1",
            created_at=now,
        )
        other_user = TaskInfo(
            task_id="u2_task",
            task_type="txt2img",
            status=TaskStatus.PENDING,
            priority=TaskPriority.NORMAL,
            user_id="user_2",
        )

        assert await store.store_task(older) is True
        assert await store.store_task(newer) is True
        assert await store.store_task(other_user) is True

        user_tasks = await store.get_user_tasks("user_1", limit=10)
        assert [t.task_id for t in user_tasks][:2] == ["u1_new", "u1_old"]


@pytest.mark.unit
class TestQueueManagerStatsAndGlobals:
    def test_estimate_task_duration_adjustments(self):
        manager = QueueManager()
        duration = manager._estimate_task_duration(
            "txt2img", {"steps": 40, "width": 2048, "height": 1024}
        )
        assert duration == pytest.approx(15.0 * 1.5 * 1.3)

    async def test_update_queue_stats(self):
        manager = QueueManager()
        manager.task_store = AsyncMock()

        now = datetime.now()
        completed_recent = TaskInfo(
            task_id="c1",
            task_type="txt2img",
            status=TaskStatus.COMPLETED,
            priority=TaskPriority.NORMAL,
            created_at=now - timedelta(minutes=10),
            started_at=now - timedelta(minutes=9),
            completed_at=now - timedelta(minutes=8),
            processing_time=60.0,
        )
        completed_old = TaskInfo(
            task_id="c2",
            task_type="txt2img",
            status=TaskStatus.COMPLETED,
            priority=TaskPriority.NORMAL,
            created_at=now - timedelta(hours=3),
            started_at=now - timedelta(hours=3) + timedelta(minutes=1),
            completed_at=now - timedelta(hours=2),
            processing_time=30.0,
        )

        async def get_queue_tasks(status: TaskStatus, limit: int = 100, offset: int = 0):
            if status == TaskStatus.PENDING:
                return [
                    TaskInfo(
                        task_id="p1",
                        task_type="txt2img",
                        status=TaskStatus.PENDING,
                        priority=TaskPriority.NORMAL,
                    )
                ]
            if status == TaskStatus.RUNNING:
                return []
            if status == TaskStatus.COMPLETED:
                return [completed_recent, completed_old]
            return []

        manager.task_store.get_queue_tasks.side_effect = get_queue_tasks  # type: ignore[attr-defined]

        manager.current_running_tasks = {"r1", "r2"}
        manager.max_concurrent_tasks = 5

        stats = await manager.get_queue_status()
        assert stats.pending_tasks == 1
        assert stats.completed_tasks == 2
        assert stats.total_tasks == 3
        assert stats.active_workers == 2
        assert stats.total_workers == 5
        assert stats.queue_throughput == pytest.approx(1 / 60)  # 1 task in last hour
        assert stats.average_wait_time > 0
        assert stats.average_processing_time > 0

    async def test_global_manager_singleton_and_shutdown(self, monkeypatch):
        monkeypatch.setattr(qm, "_queue_manager", None)

        with patch.object(qm.QueueManager, "initialize", new=AsyncMock()) as init_mock:
            manager1 = await qm.get_queue_manager()
            manager2 = await qm.get_queue_manager()
            assert manager1 is manager2
            init_mock.assert_awaited_once()

        with patch.object(qm.QueueManager, "shutdown", new=AsyncMock()) as shutdown_mock:
            await qm.shutdown_queue_manager()
            shutdown_mock.assert_awaited_once()
            assert qm._queue_manager is None
