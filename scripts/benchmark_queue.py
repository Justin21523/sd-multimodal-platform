#!/usr/bin/env python3
# scripts/benchmark_queue.py - Phase 6 Queue System Benchmark
import asyncio
import time
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.queue_manager import QueueManager, TaskPriority, TaskStatus
from app.config import get_testing_settings
import redis.asyncio as redis


class QueueBenchmark:
    """Comprehensive queue system benchmark"""

    def __init__(self):
        self.settings = get_testing_settings()
        self.redis_url = self.settings.get_redis_url()
        self.results = {}

    async def setup(self):
        """Setup benchmark environment"""
        print("🚀 Setting up benchmark environment...")

        # Clear Redis test DB
        redis_client = redis.from_url(self.redis_url)
        await redis_client.flushdb()
        await redis_client.close()

        # Initialize queue manager
        self.queue_manager = QueueManager()
        await self.queue_manager.initialize()

        print("✅ Setup completed")

    async def teardown(self):
        """Cleanup benchmark environment"""
        print("🧹 Cleaning up...")

        if hasattr(self, "queue_manager"):
            await self.queue_manager.shutdown()

        print("✅ Cleanup completed")

    async def benchmark_task_enqueue(self, num_tasks: int = 100) -> Dict[str, float]:
        """Benchmark task enqueuing performance"""
        print(f"📊 Benchmarking task enqueue ({num_tasks} tasks)...")

        start_time = time.time()
        task_ids = []

        # Enqueue tasks sequentially
        for i in range(num_tasks):
            task_id = await self.queue_manager.enqueue_task(
                task_type="txt2img",
                input_params={
                    "prompt": f"benchmark test {i}",
                    "steps": 20,
                    "width": 512,
                    "height": 512,
                },
                user_id=f"user_{i % 10}",  # 10 different users
                priority=TaskPriority.NORMAL,
            )
            if task_id:
                task_ids.append(task_id)

        end_time = time.time()
        duration = end_time - start_time

        results = {
            "total_tasks": num_tasks,
            "successful_tasks": len(task_ids),
            "failed_tasks": num_tasks - len(task_ids),
            "total_time": duration,
            "tasks_per_second": len(task_ids) / duration if duration > 0 else 0,
            "avg_time_per_task": duration / num_tasks if num_tasks > 0 else 0,
        }

        print(f"  ✅ Enqueued {len(task_ids)}/{num_tasks} tasks in {duration:.2f}s")
        print(f"  ⚡ {results['tasks_per_second']:.1f} tasks/second")

        return results

    async def benchmark_concurrent_enqueue(
        self, num_tasks: int = 100, concurrency: int = 10
    ) -> Dict[str, float]:
        """Benchmark concurrent task enqueuing"""
        print(
            f"📊 Benchmarking concurrent enqueue ({num_tasks} tasks, {concurrency} concurrent)..."
        )

        async def enqueue_batch(start_idx: int, batch_size: int) -> List[str]:
            """Enqueue a batch of tasks"""
            task_ids = []
            for i in range(start_idx, start_idx + batch_size):
                task_id = await self.queue_manager.enqueue_task(
                    task_type="txt2img",
                    input_params={"prompt": f"concurrent test {i}", "steps": 15},
                    user_id=f"user_{i % 20}",
                    priority=TaskPriority.NORMAL,
                )
                if task_id:
                    task_ids.append(task_id)
            return task_ids

        # Create batches
        batch_size = max(1, num_tasks // concurrency)
        batches = []

        for i in range(0, num_tasks, batch_size):
            actual_batch_size = min(batch_size, num_tasks - i)
            batches.append((i, actual_batch_size))

        start_time = time.time()

        # Run batches concurrently
        batch_tasks = [enqueue_batch(start, size) for start, size in batches]
        results_lists = await asyncio.gather(*batch_tasks, return_exceptions=True)

        end_time = time.time()
        duration = end_time - start_time

        # Count successful tasks
        successful_tasks = 0
        for result in results_lists:
            if isinstance(result, list):
                successful_tasks += len(result)

        results = {
            "total_tasks": num_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": num_tasks - successful_tasks,
            "concurrency": concurrency,
            "total_time": duration,
            "tasks_per_second": successful_tasks / duration if duration > 0 else 0,
            "avg_time_per_task": duration / num_tasks if num_tasks > 0 else 0,
        }

        print(f"  ✅ Enqueued {successful_tasks}/{num_tasks} tasks in {duration:.2f}s")
        print(f"  ⚡ {results['tasks_per_second']:.1f} tasks/second (concurrent)")

        return results

    async def benchmark_status_queries(
        self, num_queries: int = 1000
    ) -> Dict[str, float]:
        """Benchmark task status query performance"""
        print(f"📊 Benchmarking status queries ({num_queries} queries)...")

        # First enqueue some tasks
        task_ids = []
        for i in range(10):
            task_id = await self.queue_manager.enqueue_task(
                task_type="txt2img",
                input_params={"prompt": f"status test {i}"},
                user_id="benchmark_user",
            )
            if task_id:
                task_ids.append(task_id)

        if not task_ids:
            print("  ❌ No tasks available for status queries")
            return {}

        # Benchmark status queries
        query_times = []
        successful_queries = 0

        for i in range(num_queries):
            task_id = task_ids[i % len(task_ids)]  # Cycle through available tasks

            start = time.time()
            task_info = await self.queue_manager.get_task_status(task_id)
            end = time.time()

            query_times.append(end - start)
            if task_info:
                successful_queries += 1

        results = {
            "total_queries": num_queries,
            "successful_queries": successful_queries,
            "failed_queries": num_queries - successful_queries,
            "total_time": sum(query_times),
            "avg_query_time": statistics.mean(query_times),
            "median_query_time": statistics.median(query_times),
            "min_query_time": min(query_times),
            "max_query_time": max(query_times),
            "queries_per_second": (
                num_queries / sum(query_times) if sum(query_times) > 0 else 0
            ),
        }

        print(f"  ✅ {successful_queries}/{num_queries} queries successful")
        print(f"  ⚡ {results['queries_per_second']:.1f} queries/second")
        print(
            f"  📊 Avg: {results['avg_query_time']*1000:.1f}ms, "
            f"Median: {results['median_query_time']*1000:.1f}ms"
        )

        return results

    async def benchmark_rate_limiting(self) -> Dict[str, Any]:
        """Benchmark rate limiting functionality"""
        print("📊 Benchmarking rate limiting...")

        # Set low rate limit for testing
        original_limit = self.queue_manager.rate_limit_requests_per_hour
        self.queue_manager.rate_limit_requests_per_hour = 5

        try:
            user_id = "rate_limit_test_user"
            successful_requests = 0
            rate_limited_requests = 0

            # Make requests until rate limited
            for i in range(10):
                task_id = await self.queue_manager.enqueue_task(
                    task_type="txt2img",
                    input_params={"prompt": f"rate limit test {i}"},
                    user_id=user_id,
                )

                if task_id:
                    successful_requests += 1
                else:
                    rate_limited_requests += 1

            results = {
                "rate_limit": 5,
                "total_requests": 10,
                "successful_requests": successful_requests,
                "rate_limited_requests": rate_limited_requests,
                "rate_limiting_working": rate_limited_requests > 0,
            }

            print(
                f"  ✅ {successful_requests} successful, {rate_limited_requests} rate limited"
            )
            print(
                f"  🚦 Rate limiting {'working' if results['rate_limiting_working'] else 'not working'}"
            )

            return results

        finally:
            # Restore original rate limit
            self.queue_manager.rate_limit_requests_per_hour = original_limit

    async def benchmark_queue_statistics(self) -> Dict[str, Any]:
        """Benchmark queue statistics collection"""
        print("📊 Benchmarking queue statistics...")

        start_time = time.time()
        stats = await self.queue_manager.get_queue_status()
        end_time = time.time()

        stats_time = end_time - start_time

        results = {
            "stats_collection_time": stats_time,
            "total_tasks": stats.total_tasks,
            "pending_tasks": stats.pending_tasks,
            "running_tasks": stats.running_tasks,
            "completed_tasks": stats.completed_tasks,
            "failed_tasks": stats.failed_tasks,
            "cancelled_tasks": stats.cancelled_tasks,
        }

        print(f"  ✅ Statistics collected in {stats_time*1000:.1f}ms")
        print(f"  📊 Total tasks: {stats.total_tasks}")

        return results

    async def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns"""
        print("📊 Benchmarking memory usage...")

        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create and enqueue many tasks
        num_tasks = 1000
        task_ids = []

        memory_before_enqueue = process.memory_info().rss / 1024 / 1024

        for i in range(num_tasks):
            task_id = await self.queue_manager.enqueue_task(
                task_type="txt2img",
                input_params={
                    "prompt": f"memory test {i}" * 10,  # Large prompt
                    "negative_prompt": f"negative test {i}" * 5,
                    "steps": 25,
                    "width": 1024,
                    "height": 1024,
                },
                user_id=f"memory_user_{i % 100}",
            )
            if task_id:
                task_ids.append(task_id)

        memory_after_enqueue = process.memory_info().rss / 1024 / 1024

        # Query all tasks
        for task_id in task_ids:
            await self.queue_manager.get_task_status(task_id)

        memory_after_queries = process.memory_info().rss / 1024 / 1024

        # Cleanup (simulate task completion)
        # Note: In real implementation, you'd have proper cleanup

        final_memory = process.memory_info().rss / 1024 / 1024

        results = {
            "initial_memory_mb": initial_memory,
            "memory_before_enqueue_mb": memory_before_enqueue,
            "memory_after_enqueue_mb": memory_after_enqueue,
            "memory_after_queries_mb": memory_after_queries,
            "final_memory_mb": final_memory,
            "memory_growth_enqueue_mb": memory_after_enqueue - memory_before_enqueue,
            "memory_growth_queries_mb": memory_after_queries - memory_after_enqueue,
            "total_memory_growth_mb": final_memory - initial_memory,
            "tasks_created": len(task_ids),
            "memory_per_task_kb": (
                (memory_after_enqueue - memory_before_enqueue) * 1024 / len(task_ids)
                if task_ids
                else 0
            ),
        }

        print(f"  📊 Memory usage:")
        print(f"    Initial: {initial_memory:.1f}MB")
        print(
            f"    After enqueue: {memory_after_enqueue:.1f}MB (+{results['memory_growth_enqueue_mb']:.1f}MB)"
        )
        print(
            f"    After queries: {memory_after_queries:.1f}MB (+{results['memory_growth_queries_mb']:.1f}MB)"
        )
        print(f"    Final: {final_memory:.1f}MB")
        print(f"    Per task: {results['memory_per_task_kb']:.1f}KB")

        return results

    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite"""
        print("🎯 Starting comprehensive queue system benchmark...\n")

        await self.setup()

        try:
            benchmark_results = {}

            # Run individual benchmarks
            benchmark_results["enqueue_sequential"] = await self.benchmark_task_enqueue(
                100
            )
            print()

            benchmark_results["enqueue_concurrent"] = (
                await self.benchmark_concurrent_enqueue(100, 10)
            )
            print()

            benchmark_results["status_queries"] = await self.benchmark_status_queries(
                1000
            )
            print()

            benchmark_results["rate_limiting"] = await self.benchmark_rate_limiting()
            print()

            benchmark_results["queue_statistics"] = (
                await self.benchmark_queue_statistics()
            )
            print()

            benchmark_results["memory_usage"] = await self.benchmark_memory_usage()
            print()

            # Summary
            print("📋 Benchmark Summary:")
            print("=" * 50)

            if "enqueue_sequential" in benchmark_results:
                seq = benchmark_results["enqueue_sequential"]
                print(f"Sequential Enqueue: {seq['tasks_per_second']:.1f} tasks/sec")

            if "enqueue_concurrent" in benchmark_results:
                conc = benchmark_results["enqueue_concurrent"]
                print(f"Concurrent Enqueue: {conc['tasks_per_second']:.1f} tasks/sec")

            if "status_queries" in benchmark_results:
                queries = benchmark_results["status_queries"]
                print(
                    f"Status Queries: {queries['queries_per_second']:.1f} queries/sec"
                )

            if "memory_usage" in benchmark_results:
                memory = benchmark_results["memory_usage"]
                print(f"Memory per Task: {memory['memory_per_task_kb']:.1f}KB")

            print("=" * 50)

            return {
                "timestamp": datetime.now().isoformat(),
                "settings": {
                    "redis_url": self.redis_url,
                    "max_concurrent_tasks": self.settings.MAX_CONCURRENT_TASKS,
                    "rate_limit_per_hour": self.settings.RATE_LIMIT_PER_HOUR,
                },
                "results": benchmark_results,
            }

        finally:
            await self.teardown()

    def save_results(self, results: Dict[str, Any], filename: str = None):  # type: ignore
        """Save benchmark results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"

        output_path = Path("benchmark_results") / filename
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"💾 Results saved to {output_path}")


async def main():
    """Main benchmark execution"""
    benchmark = QueueBenchmark()

    try:
        results = await benchmark.run_full_benchmark()
        benchmark.save_results(results)

        # Exit with success
        return 0

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
