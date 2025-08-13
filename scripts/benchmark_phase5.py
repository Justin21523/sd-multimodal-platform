# scripts/benchmark_phase5.py
"""
Phase 5 benchmark testing for queue system and post-processing
"""

import asyncio
import time
import requests
import json
from typing import List, Dict, Any
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import settings


class Phase5Benchmark:
    """Benchmark Phase 5 queue and post-processing performance"""

    def __init__(self):
        self.base_url = f"http://localhost:{settings.PORT}"
        self.api_prefix = settings.API_PREFIX
        self.results = {}

    def submit_test_task(self, task_type: str = "txt2img") -> str:
        """Submit a test task for benchmarking"""

        if task_type == "txt2img":
            task_data = {
                "prompt": "a realistic photo of a cat",
                "width": 512,
                "height": 512,
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
            }
            endpoint = f"{self.base_url}{self.api_prefix}/queue/submit/txt2img"
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        response = requests.post(endpoint, json=task_data)

        if response.status_code == 200:
            return response.json()["data"]["task_id"]
        else:
            raise Exception(f"Task submission failed: {response.status_code}")

    def wait_for_task_completion(
        self, task_id: str, timeout: int = 300
    ) -> Dict[str, Any]:
        """Wait for task completion and measure performance"""

        start_time = time.time()
        submission_time = start_time

        while time.time() - start_time < timeout:
            response = requests.get(
                f"{self.base_url}{self.api_prefix}/queue/status/{task_id}"
            )

            if response.status_code == 200:
                task_info = response.json()["data"]
                status = task_info["status"]

                if status == "success":
                    completion_time = time.time()
                    return {
                        "success": True,
                        "total_time": completion_time - submission_time,
                        "task_info": task_info,
                    }
                elif status == "failure":
                    return {
                        "success": False,
                        "error": task_info.get("error_message", "Unknown error"),
                        "total_time": time.time() - submission_time,
                    }
                elif status in ["pending", "started", "processing"]:
                    time.sleep(2)
                    continue

            time.sleep(2)

        return {"success": False, "error": "Timeout", "total_time": timeout}

    def benchmark_single_task(self, task_type: str = "txt2img") -> Dict[str, Any]:
        """Benchmark single task execution"""

        print(f"🔥 Benchmarking single {task_type} task...")

        try:
            # Submit task
            task_id = self.submit_test_task(task_type)
            print(f"   📋 Task submitted: {task_id}")

            # Wait for completion
            result = self.wait_for_task_completion(task_id)

            if result["success"]:
                print(f"   ✅ Completed in {result['total_time']:.2f}s")
                return {
                    "success": True,
                    "task_id": task_id,
                    "execution_time": result["total_time"],
                    "task_info": result["task_info"],
                }
            else:
                print(f"   ❌ Failed: {result['error']}")
                return {"success": False, "error": result["error"]}

        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
            return {"success": False, "error": str(e)}

    def benchmark_concurrent_tasks(self, num_tasks: int = 3) -> Dict[str, Any]:
        """Benchmark concurrent task execution"""

        print(f"🔥 Benchmarking {num_tasks} concurrent txt2img tasks...")

        try:
            # Submit multiple tasks quickly
            task_ids = []
            submission_start = time.time()

            for i in range(num_tasks):
                task_id = self.submit_test_task("txt2img")
                task_ids.append(task_id)
                print(f"   📋 Task {i+1} submitted: {task_id}")

            submission_time = time.time() - submission_start
            print(f"   📊 All tasks submitted in {submission_time:.2f}s")

            # Wait for all tasks to complete
            completed_tasks = []
            start_time = time.time()

            while len(completed_tasks) < num_tasks and (time.time() - start_time) < 600:
                for task_id in task_ids:
                    if task_id in [t["task_id"] for t in completed_tasks]:
                        continue

                    response = requests.get(
                        f"{self.base_url}{self.api_prefix}/queue/status/{task_id}"
                    )

                    if response.status_code == 200:
                        task_info = response.json()["data"]
                        status = task_info["status"]

                        if status in ["success", "failure"]:
                            completed_tasks.append(
                                {
                                    "task_id": task_id,
                                    "status": status,
                                    "completion_time": time.time(),
                                    "task_info": task_info,
                                }
                            )
                            print(f"   ✅ Task {task_id} completed ({status})")

                time.sleep(3)

            total_time = time.time() - start_time
            successful_tasks = [t for t in completed_tasks if t["status"] == "success"]

            print(f"   📊 {len(successful_tasks)}/{num_tasks} tasks successful")
            print(f"   📊 Total time: {total_time:.2f}s")

            if successful_tasks:
                avg_time = sum(
                    t["completion_time"] - start_time for t in successful_tasks
                ) / len(successful_tasks)
                print(f"   📊 Average task time: {avg_time:.2f}s")

            return {
                "success": len(successful_tasks) > 0,
                "total_tasks": num_tasks,
                "successful_tasks": len(successful_tasks),
                "total_time": total_time,
                "submission_time": submission_time,
                "results": completed_tasks,
            }

        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
            return {"success": False, "error": str(e)}

    def benchmark_queue_performance(self) -> Dict[str, Any]:
        """Benchmark queue system performance"""

        print("🔥 Benchmarking queue system performance...")

        try:
            # Test queue stats endpoint
            start_time = time.time()
            response = requests.get(f"{self.base_url}{self.api_prefix}/queue/stats")
            stats_time = time.time() - start_time

            if response.status_code == 200:
                stats = response.json()["data"]
                print(f"   ✅ Queue stats retrieved in {stats_time*1000:.1f}ms")
                print(f"   📊 Active workers: {stats.get('active_workers', 0)}")
                print(f"   📊 Redis connected: {stats.get('redis_connected', False)}")

                return {
                    "success": True,
                    "stats_response_time": stats_time,
                    "queue_stats": stats,
                }
            else:
                print(f"   ❌ Queue stats failed: {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}

        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
            return {"success": False, "error": str(e)}

    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete Phase 5 benchmark suite"""

        print("🚀 Starting Phase 5 Performance Benchmark")
        print("=" * 60)

        # Individual benchmarks
        benchmarks = [
            ("Queue Performance", self.benchmark_queue_performance),
            ("Single Task", lambda: self.benchmark_single_task("txt2img")),
            ("Concurrent Tasks", lambda: self.benchmark_concurrent_tasks(3)),
        ]

        results = {}

        for benchmark_name, benchmark_func in benchmarks:
            print(f"\n{'='*40}")
            print(f"Running: {benchmark_name}")
            print("=" * 40)

            result = benchmark_func()
            results[benchmark_name.lower().replace(" ", "_")] = result

            if result.get("success"):
                print(f"✅ {benchmark_name} completed successfully")
            else:
                print(
                    f"❌ {benchmark_name} failed: {result.get('error', 'Unknown error')}"
                )

        # Summary
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        successful_benchmarks = sum(1 for r in results.values() if r.get("success"))
        total_benchmarks = len(results)

        print(f"Successful benchmarks: {successful_benchmarks}/{total_benchmarks}")

        # Detailed performance metrics
        if results.get("single_task", {}).get("success"):
            single_time = results["single_task"]["execution_time"]
            print(f"Single task execution: {single_time:.2f}s")

        if results.get("concurrent_tasks", {}).get("success"):
            concurrent_result = results["concurrent_tasks"]
            efficiency = (
                concurrent_result["successful_tasks"]
                / concurrent_result["total_tasks"]
                * 100
            )
            print(f"Concurrent tasks efficiency: {efficiency:.1f}%")

        if results.get("queue_performance", {}).get("success"):
            queue_time = results["queue_performance"]["stats_response_time"]
            print(f"Queue stats response: {queue_time*1000:.1f}ms")

        # Performance rating
        overall_score = successful_benchmarks / total_benchmarks * 100
        if overall_score >= 90:
            rating = "🌟 Excellent"
        elif overall_score >= 70:
            rating = "✅ Good"
        elif overall_score >= 50:
            rating = "⚠️ Fair"
        else:
            rating = "❌ Poor"

        print(f"Overall performance: {rating} ({overall_score:.1f}%)")

        return {
            "overall_success": successful_benchmarks == total_benchmarks,
            "success_rate": overall_score,
            "performance_rating": rating,
            "detailed_results": results,
        }


def main():
    """Main benchmark function"""
    import argparse

    parser = argparse.ArgumentParser(description="Phase 5 performance benchmark")
    parser.add_argument(
        "--single", action="store_true", help="Run single task benchmark only"
    )
    parser.add_argument(
        "--concurrent", action="store_true", help="Run concurrent benchmark only"
    )
    parser.add_argument("--queue", action="store_true", help="Run queue benchmark only")
    parser.add_argument(
        "--tasks", type=int, default=3, help="Number of concurrent tasks"
    )

    args = parser.parse_args()

    benchmark = Phase5Benchmark()

    if args.single:
        result = benchmark.benchmark_single_task()
    elif args.concurrent:
        result = benchmark.benchmark_concurrent_tasks(args.tasks)
    elif args.queue:
        result = benchmark.benchmark_queue_performance()
    else:
        result = benchmark.run_full_benchmark()

    # Save results to file
    results_file = Path("benchmark_results_phase5.json")
    with open(results_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n📄 Results saved to: {results_file}")

    return result.get("overall_success", result.get("success", False))


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
