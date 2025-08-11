#!/usr/bin/env python3
# scripts/run_tests.py
"""
Test runner script for SD Multi-Modal Platform
Phase 2: Backend Framework & Basic API Services
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import fastapi
import uvicorn
import pydantic

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_command(cmd, description=""):
    """Run command and handle errors."""
    print(f"🔄 {description}")
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ {description} - SUCCESS")
        if result.stdout:
            print(result.stdout)
        return True
    else:
        print(f"❌ {description} - FAILED")
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.stdout:
            print("STDOUT:", result.stdout)
        return False


def install_test_dependencies():
    """Install testing dependencies."""
    dependencies = [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.0.0",
        "pytest-timeout>=2.1.0",
        "httpx>=0.24.0",
        "coverage>=7.0.0",
    ]

    cmd = [sys.executable, "-m", "pip", "install"] + dependencies
    return run_command(cmd, "Installing test dependencies")


def run_linting():
    """Run code linting checks."""
    print("\n🔍 Running code quality checks...")

    # Check if black is installed
    try:
        subprocess.run(
            [sys.executable, "-m", "black", "--version"],
            capture_output=True,
            check=True,
        )

        # Run black check
        cmd = [
            sys.executable,
            "-m",
            "black",
            "--check",
            "--diff",
            "app/",
            "tests/",
            "utils/",
        ]
        if not run_command(cmd, "Code formatting check (black)"):
            print("💡 Run 'python -m black app/ tests/ utils/' to fix formatting")
            return False
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Black not installed, skipping formatting check")

    return True


def run_unit_tests(verbose=True, coverage=True):
    """Run unit tests."""
    print("\n🧪 Running unit tests...")

    cmd = [sys.executable, "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(
            [
                "--cov=app",
                "--cov=utils",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-fail-under=70",
            ]
        )

    # Add test markers
    cmd.extend(["-m", "not slow", "tests/"])  # Skip slow tests by default

    return run_command(cmd, "Unit tests")


def run_integration_tests():
    """Run integration tests."""
    print("\n🔗 Running integration tests...")

    cmd = [sys.executable, "-m", "pytest", "-v", "-m", "integration", "tests/"]

    return run_command(cmd, "Integration tests")


def run_performance_tests():
    """Run performance benchmark tests."""
    print("\n⚡ Running performance tests...")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-v",
        "-s",  # Don't capture output for benchmarks
        "-m",
        "performance",
        "tests/",
    ]

    return run_command(cmd, "Performance tests")


def run_specific_test(test_path):
    """Run a specific test file or test function."""
    print(f"\n🎯 Running specific test: {test_path}")

    cmd = [sys.executable, "-m", "pytest", "-v", "-s", test_path]

    return run_command(cmd, f"Specific test: {test_path}")


def generate_test_report():
    """Generate comprehensive test report."""
    print("\n📊 Generating test report...")

    # Run all tests with detailed output
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--tb=short",
        "--cov=app",
        "--cov=utils",
        "--cov-report=html:htmlcov",
        "--cov-report=xml:coverage.xml",
        "--junit-xml=test-results.xml",
        "tests/",
    ]

    success = run_command(cmd, "Comprehensive test report")

    if success:
        print("\n📋 Test report generated:")
        print("  - HTML coverage: htmlcov/index.html")
        print("  - XML coverage: coverage.xml")
        print("  - JUnit XML: test-results.xml")

    return success


def check_environment():
    """Check if test environment is properly configured."""
    print("🔍 Checking test environment...")

    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print(
            f"❌ Python {python_version.major}.{python_version.minor} is too old. Need 3.8+"
        )
        return False

    # Check if we're in project root
    if not Path("app").exists() or not Path("tests").exists():
        print("❌ Please run from project root directory")
        return False

    # Check required modules
    try:
        print("✅ Core dependencies available")
    except ImportError as e:
        print(f"❌ Missing core dependency: {e}")
        return False

    return True


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="SD Multi-Modal Platform Test Runner")
    parser.add_argument(
        "--install-deps", action="store_true", help="Install test dependencies"
    )
    parser.add_argument("--lint", action="store_true", help="Run linting checks only")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests only"
    )
    parser.add_argument(
        "--performance", action="store_true", help="Run performance tests only"
    )
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument(
        "--report", action="store_true", help="Generate comprehensive test report"
    )
    parser.add_argument("--test", type=str, help="Run specific test (file::function)")
    parser.add_argument(
        "--no-coverage", action="store_true", help="Disable coverage reporting"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    print("🧪 SD Multi-Modal Platform - Test Runner")
    print("=" * 50)

    # Check environment first
    if not check_environment():
        return 1

    # Install dependencies if requested
    if args.install_deps:
        if not install_test_dependencies():
            return 1
        print("✅ Test dependencies installed successfully")
        return 0

    # Determine what to run
    run_lint = args.lint or args.all
    run_unit = (
        args.unit
        or args.all
        or (
            not any(
                [args.lint, args.integration, args.performance, args.test, args.report]
            )
        )
    )
    run_integration = args.integration or args.all
    run_performance = args.performance or args.all

    success = True

    # Run linting
    if run_lint:
        success &= run_linting()

    # Run specific test
    if args.test:
        success &= run_specific_test(args.test)
        return 0 if success else 1

    # Generate report
    if args.report:
        success &= generate_test_report()
        return 0 if success else 1

    # Run unit tests
    if run_unit:
        success &= run_unit_tests(verbose=not args.quiet, coverage=not args.no_coverage)

    # Run integration tests
    if run_integration:
        success &= run_integration_tests()

    # Run performance tests
    if run_performance:
        success &= run_performance_tests()

    # Summary
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed successfully!")
        print("\nNext steps:")
        print("  1. Check coverage report: htmlcov/index.html")
        print("  2. Commit your changes if all tests pass")
        print("  3. Proceed to Phase 3 development")
        return 0
    else:
        print("❌ Some tests failed. Please fix issues before proceeding.")
        return 1


if __name__ == "__main__":
    main()
