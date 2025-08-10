# scripts/test_backend.py
"""
Manual Backend Testing Script

For testing the backend API manually during development.
"""

import requests
import json
import time
from pathlib import Path


class BackendTester:
    """Helper class for manual backend testing"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def test_health(self):
        """Test health endpoint"""
        print("🔍 Testing health endpoint...")

        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()

            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            print(f"   Model loaded: {data['model_loaded']}")
            print(f"   Device: {data['device']}")

            return True

        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False

    def test_model_info(self):
        """Test model information"""
        print("\n🔍 Testing model info...")

        try:
            response = self.session.get(f"{self.base_url}/api/v1/models/info")
            response.raise_for_status()

            data = response.json()
            print(f"✅ Model info retrieved")
            print(f"   Current model: {data['model_info']}")
            print(f"   Available models: {list(data['available_models'].keys())}")

            return True

        except Exception as e:
            print(f"❌ Model info failed: {e}")
            return False

    def test_text_to_image(self, prompt: str = "A beautiful sunset over mountains"):
        """Test text-to-image generation"""
        print(f"\n🔍 Testing text-to-image: '{prompt}'...")

        request_data = {
            "prompt": prompt,
            "negative_prompt": "blurry, low quality",
            "width": 512,
            "height": 512,
            "num_inference_steps": 10,  # Faster for testing
            "guidance_scale": 7.5,
            "batch_size": 1,
        }

        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/api/v1/txt2img/generate",
                json=request_data,
                timeout=300,  # 5 minute timeout
            )
            response.raise_for_status()

            data = response.json()
            generation_time = time.time() - start_time

            print(f"✅ Image generation completed in {generation_time:.2f}s")
            print(f"   Generated images: {len(data['images'])}")
            print(f"   Server generation time: {data['generation_time']:.2f}s")

            # Print image URLs
            for i, img_data in enumerate(data["images"]):
                print(f"   Image {i+1}: {img_data['url']}")

            return True

        except requests.exceptions.Timeout:
            print("❌ Request timeout (model might be downloading)")
            return False
        except Exception as e:
            print(f"❌ Image generation failed: {e}")
            return False

    def test_parameters(self):
        """Test parameter endpoint"""
        print("\n🔍 Testing parameters endpoint...")

        try:
            response = self.session.get(f"{self.base_url}/api/v1/txt2img/parameters")
            response.raise_for_status()

            data = response.json()
            print(f"✅ Parameters retrieved")
            print(f"   Available schedulers: {data['schedulers']}")
            print(
                f"   Width range: {data['parameters']['width']['min']}-{data['parameters']['width']['max']}"
            )
            print(
                f"   Height range: {data['parameters']['height']['min']}-{data['parameters']['height']['max']}"
            )

            return True

        except Exception as e:
            print(f"❌ Parameters test failed: {e}")
            return False

    def run_all_tests(self):
        """Run all tests in sequence"""
        print("🚀 Starting backend API tests...\n")

        tests = [
            self.test_health,
            self.test_model_info,
            self.test_parameters,
            self.test_text_to_image,
        ]

        passed = 0
        total = len(tests)

        for test in tests:
            if test():
                passed += 1

        print(f"\n📊 Test Results: {passed}/{total} tests passed")

        if passed == total:
            print("🎉 All tests passed!")
        else:
            print("⚠️ Some tests failed. Check the logs above.")

        return passed == total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test SD Backend API")
    parser.add_argument("--url", default="http://localhost:8000", help="Backend URL")
    parser.add_argument(
        "--prompt", default="A beautiful sunset over mountains", help="Test prompt"
    )

    args = parser.parse_args()

    tester = BackendTester(args.url)

    # Run individual test or all tests
    if args.prompt != "A beautiful sunset over mountains":
        tester.test_text_to_image(args.prompt)
    else:
        tester.run_all_tests()
