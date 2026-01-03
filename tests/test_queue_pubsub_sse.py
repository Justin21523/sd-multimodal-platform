# tests/test_queue_pubsub_sse.py
from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from app.api.v1.queue import stream_user_tasks
from app.api.v1.queue import stream_task_status
from app.api.v1.queue import stream_queue_tasks
from app.core.queue_manager import RedisTaskStore, TaskInfo, TaskPriority, TaskStatus


@pytest.mark.unit
async def test_task_store_publishes_user_task_event_channel_and_payload():
    class FakeRedis:
        publish = AsyncMock(return_value=1)

    store = RedisTaskStore("redis://localhost:6379/0")
    store.redis_client = FakeRedis()

    task_info = TaskInfo(
        task_id="task_123",
        task_type="txt2img",
        status=TaskStatus.PENDING,
        priority=TaskPriority.NORMAL,
        user_id="user_abc",
    )

    await store._publish_user_task_event(task_info)

    channel, payload = store.redis_client.publish.await_args.args  # type: ignore[attr-defined]
    assert channel == "user:user_abc:events"
    assert json.loads(payload) == {"task_id": "task_123"}


@pytest.mark.unit
async def test_task_store_publishes_task_event_channel_and_payload():
    class FakeRedis:
        publish = AsyncMock(return_value=1)

    store = RedisTaskStore("redis://localhost:6379/0")
    store.redis_client = FakeRedis()

    task_info = TaskInfo(
        task_id="task_123",
        task_type="txt2img",
        status=TaskStatus.PENDING,
        priority=TaskPriority.NORMAL,
        user_id="user_abc",
    )

    await store._publish_task_event(task_info)

    channel, payload = store.redis_client.publish.await_args.args  # type: ignore[attr-defined]
    assert channel == "task:task_123:events"
    assert json.loads(payload) == {"task_id": "task_123"}


@pytest.mark.unit
async def test_task_store_publishes_queue_event_channel_and_payload():
    class FakeRedis:
        publish = AsyncMock(return_value=1)

    store = RedisTaskStore("redis://localhost:6379/0")
    store.redis_client = FakeRedis()

    task_info = TaskInfo(
        task_id="task_123",
        task_type="txt2img",
        status=TaskStatus.PENDING,
        priority=TaskPriority.NORMAL,
        user_id="user_abc",
    )

    await store._publish_queue_event(task_info)

    channel, payload = store.redis_client.publish.await_args.args  # type: ignore[attr-defined]
    assert channel == "queue:events"
    assert json.loads(payload) == {"task_id": "task_123"}


@pytest.mark.unit
async def test_sse_user_stream_consumes_pubsub_events_for_updates():
    class FakePubSub:
        def __init__(self):
            self.subscribed: list[str] = []
            self.unsubscribed: list[str] = []
            self.closed = False
            self._messages = [
                {"type": "message", "data": json.dumps({"task_id": "task_1"})}
            ]

        async def subscribe(self, channel: str) -> None:
            self.subscribed.append(channel)

        async def unsubscribe(self, channel: str) -> None:
            self.unsubscribed.append(channel)

        async def get_message(self, ignore_subscribe_messages: bool = True, timeout: float = 0.0):
            if self._messages:
                return self._messages.pop(0)
            return None

        async def close(self) -> None:
            self.closed = True

    class FakeRedis:
        def __init__(self, pubsub: FakePubSub):
            self._pubsub = pubsub

        def pubsub(self) -> FakePubSub:
            return self._pubsub

    class FakeTaskStore:
        def __init__(self, redis_client: FakeRedis):
            self.redis_client = redis_client
            self.user_tasks_prefix = "user:"
            self.user_events_suffix = ":events"

    class FakeQueueManager:
        def __init__(self, task_store: FakeTaskStore):
            self.task_store = task_store

            self._initial = TaskInfo(
                task_id="task_1",
                task_type="txt2img",
                status=TaskStatus.PENDING,
                priority=TaskPriority.NORMAL,
                user_id="user_abc",
            )
            self._updated = TaskInfo(
                task_id="task_1",
                task_type="txt2img",
                status=TaskStatus.RUNNING,
                priority=TaskPriority.NORMAL,
                user_id="user_abc",
                current_step="generating",
            )

        async def get_user_tasks(self, user_id: str, limit: int, offset: int = 0):
            assert user_id == "user_abc"
            return [self._initial]

        async def get_task_status(self, task_id: str):
            assert task_id == "task_1"
            return self._updated

    pubsub = FakePubSub()
    manager = FakeQueueManager(FakeTaskStore(FakeRedis(pubsub)))

    response = await stream_user_tasks(
        user_id="user_abc",
        limit=10,
        queue_manager=manager,  # type: ignore[arg-type]
    )

    iterator = response.body_iterator  # type: ignore[assignment]
    first = await anext(iterator)
    second = await anext(iterator)
    await iterator.aclose()

    assert pubsub.subscribed == ["user:user_abc:events"]

    assert isinstance(first, str) and first.startswith("data: ")
    payload_1 = json.loads(first[len("data: ") :].strip())
    assert payload_1["task_id"] == "task_1"
    assert payload_1["status"] == "pending"

    assert isinstance(second, str) and second.startswith("data: ")
    payload_2 = json.loads(second[len("data: ") :].strip())
    assert payload_2["task_id"] == "task_1"
    assert payload_2["status"] == "running"

    assert pubsub.closed is True


@pytest.mark.unit
async def test_sse_task_stream_consumes_pubsub_events_for_updates():
    class FakePubSub:
        def __init__(self):
            self.subscribed: list[str] = []
            self.unsubscribed: list[str] = []
            self.closed = False
            self._messages = [
                {"type": "message", "data": json.dumps({"task_id": "task_1"})}
            ]

        async def subscribe(self, channel: str) -> None:
            self.subscribed.append(channel)

        async def unsubscribe(self, channel: str) -> None:
            self.unsubscribed.append(channel)

        async def get_message(
            self, ignore_subscribe_messages: bool = True, timeout: float = 0.0
        ):
            if self._messages:
                return self._messages.pop(0)
            return None

        async def close(self) -> None:
            self.closed = True

    class FakeRedis:
        def __init__(self, pubsub: FakePubSub):
            self._pubsub = pubsub

        def pubsub(self) -> FakePubSub:
            return self._pubsub

    class FakeTaskStore:
        def __init__(self, redis_client: FakeRedis):
            self.redis_client = redis_client
            self.user_tasks_prefix = "user:"
            self.user_events_suffix = ":events"

    class FakeQueueManager:
        def __init__(self, task_store: FakeTaskStore):
            self.task_store = task_store
            self._calls = 0

            self._initial = TaskInfo(
                task_id="task_1",
                task_type="txt2img",
                status=TaskStatus.PENDING,
                priority=TaskPriority.NORMAL,
                user_id="user_abc",
            )
            self._updated = TaskInfo(
                task_id="task_1",
                task_type="txt2img",
                status=TaskStatus.RUNNING,
                priority=TaskPriority.NORMAL,
                user_id="user_abc",
                current_step="generating",
            )

        async def get_task_status(self, task_id: str):
            assert task_id == "task_1"
            self._calls += 1
            if self._calls == 1:
                return self._initial
            return self._updated

    pubsub = FakePubSub()
    manager = FakeQueueManager(FakeTaskStore(FakeRedis(pubsub)))

    response = await stream_task_status(
        task_id="task_1",
        queue_manager=manager,  # type: ignore[arg-type]
    )

    iterator = response.body_iterator  # type: ignore[assignment]
    first = await anext(iterator)
    second = await anext(iterator)
    await iterator.aclose()

    assert pubsub.subscribed == ["task:task_1:events"]

    assert isinstance(first, str) and first.startswith("data: ")
    payload_1 = json.loads(first[len("data: ") :].strip())
    assert payload_1["task_id"] == "task_1"
    assert payload_1["status"] == "pending"

    assert isinstance(second, str) and second.startswith("data: ")
    payload_2 = json.loads(second[len("data: ") :].strip())
    assert payload_2["task_id"] == "task_1"
    assert payload_2["status"] == "running"

    assert pubsub.closed is True


@pytest.mark.unit
async def test_sse_queue_stream_consumes_pubsub_events_for_updates():
    class FakePubSub:
        def __init__(self):
            self.subscribed: list[str] = []
            self.unsubscribed: list[str] = []
            self.closed = False
            self._messages = [
                {"type": "message", "data": json.dumps({"task_id": "task_1"})},
                {"type": "message", "data": json.dumps({"task_id": "task_1"})},
            ]

        async def subscribe(self, channel: str) -> None:
            self.subscribed.append(channel)

        async def unsubscribe(self, channel: str) -> None:
            self.unsubscribed.append(channel)

        async def get_message(
            self, ignore_subscribe_messages: bool = True, timeout: float = 0.0
        ):
            if self._messages:
                return self._messages.pop(0)
            return None

        async def close(self) -> None:
            self.closed = True

    class FakeRedis:
        def __init__(self, pubsub: FakePubSub):
            self._pubsub = pubsub

        def pubsub(self) -> FakePubSub:
            return self._pubsub

    class FakeTaskStore:
        def __init__(self, redis_client: FakeRedis):
            self.redis_client = redis_client
            self.queue_events_channel = "queue:events"

    class FakeQueueManager:
        def __init__(self, task_store: FakeTaskStore):
            self.task_store = task_store
            self._calls = 0

            self._initial = TaskInfo(
                task_id="task_1",
                task_type="txt2img",
                status=TaskStatus.PENDING,
                priority=TaskPriority.NORMAL,
                user_id="user_abc",
            )
            self._updated = TaskInfo(
                task_id="task_1",
                task_type="txt2img",
                status=TaskStatus.RUNNING,
                priority=TaskPriority.NORMAL,
                user_id="user_abc",
                current_step="generating",
            )

        async def get_task_status(self, task_id: str):
            assert task_id == "task_1"
            self._calls += 1
            if self._calls == 1:
                return self._initial
            return self._updated

    pubsub = FakePubSub()
    manager = FakeQueueManager(FakeTaskStore(FakeRedis(pubsub)))

    response = await stream_queue_tasks(queue_manager=manager)  # type: ignore[arg-type]

    iterator = response.body_iterator  # type: ignore[assignment]
    keep_alive = await anext(iterator)
    first = await anext(iterator)
    second = await anext(iterator)
    await iterator.aclose()

    assert keep_alive.startswith(":")
    assert pubsub.subscribed == ["queue:events"]

    assert isinstance(first, str) and first.startswith("data: ")
    payload_1 = json.loads(first[len("data: ") :].strip())
    assert payload_1["task_id"] == "task_1"
    assert payload_1["status"] == "pending"

    assert isinstance(second, str) and second.startswith("data: ")
    payload_2 = json.loads(second[len("data: ") :].strip())
    assert payload_2["task_id"] == "task_1"
    assert payload_2["status"] == "running"

    assert pubsub.closed is True
