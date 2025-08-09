# scripts/download_models.py - Architecture Analysis


class ModelDownloader:
    """
    Model Management System with Following Design Principles:

    1. **Declarative Configuration**: Models defined as metadata dictionaries
    2. **Idempotent Operations**: Safe to run multiple times
    3. **Resumable Downloads**: Handle interrupted downloads gracefully
    4. **Path Abstraction**: Centralized path management
    5. **Error Handling**: Comprehensive exception management
    """

    def __init__(self, base_path: str = "models"):
        # Design Pattern: Dependency Injection
        # Allows different base paths for dev/test/prod environments
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

        # Design Pattern: Configuration as Data
        # Separates model metadata from download logic
        self.models = {
            "sd-1.5": {
                "repo_id": "runwayml/stable-diffusion-v1-5",
                "path": "stable-diffusion/sd-1.5",
                "description": "Stable Diffusion v1.5 baseline model",
            },
            # ... more models
        }

    def download_model(self, model_name: str, force: bool = False):
        """
        Core Download Logic with Key Design Decisions:

        1. **Validation First**: Check model exists in registry
        2. **Conflict Resolution**: Handle existing files gracefully
        3. **Atomic Operations**: Create directories before download
        4. **Resume Support**: Use huggingface_hub's resume capability
        5. **Explicit Error Handling**: Clear feedback on failures
        """

        # Input Validation Pattern
        if model_name not in self.models:
            logger.error(f"Unknown model: {model_name}")
            logger.info(f"Available models: {list(self.models.keys())}")
            return False

        model_config = self.models[model_name]
        download_path = self.base_path / model_config["path"]

        # Idempotent Check Pattern
        if download_path.exists() and not force:
            logger.info(f"Model already exists: {download_path}")
            return True

        try:
            # Atomic Directory Creation
            download_path.mkdir(parents=True, exist_ok=True)

            # HuggingFace Hub Integration
            # Key Parameters:
            # - local_dir_use_symlinks=False: Actual files vs symlinks
            # - resume_download=True: Handle partial downloads
            snapshot_download(
                repo_id=model_config["repo_id"],
                local_dir=download_path,
                local_dir_use_symlinks=False,
                resume_download=True,
            )

            return True

        except Exception as e:
            # Comprehensive Error Handling
            logger.error(f"Download failed: {e}")
            return False

    def check_models(self):
        """
        Model Verification System:

        1. **Status Reporting**: Visual feedback on model states
        2. **Storage Analytics**: Calculate actual disk usage
        3. **Path Validation**: Verify expected directory structure
        """

        for name, config in self.models.items():
            model_path = self.base_path / config["path"]
            if model_path.exists():
                # Storage Analysis: Recursive file size calculation
                size = sum(
                    f.stat().st_size for f in model_path.rglob("*") if f.is_file()
                )
                size_gb = size / (1024**3)
                print(f"✅ {name:20} | {size_gb:.2f} GB | {model_path}")
            else:
                print(f"❌ {name:20} | Not found | {model_path}")


# Command Line Interface Design
def main():
    """
    CLI Design Principles:

    1. **Single Responsibility**: Each flag handles one operation
    2. **Safe Defaults**: Prevent accidental bulk downloads
    3. **Informational Commands**: List and check before download
    4. **Override Options**: Force and path customization
    """

    parser = argparse.ArgumentParser(description="SD Model Download Utility")

    # Mutually Exclusive Operations
    parser.add_argument("--model", help="Download specific model")
    parser.add_argument("--all", action="store_true", help="Download essential models")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--check", action="store_true", help="Check model status")

    # Modifier Options
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--base-path", default="models", help="Custom model directory")


# Key Benefits of This Architecture:
#
# 1. **Maintainability**: Easy to add new models via configuration
# 2. **Reliability**: Handles network failures and partial downloads
# 3. **Observability**: Clear logging and status reporting
# 4. **Flexibility**: Supports different environments and use cases
# 5. **User Experience**: Informative CLI with safety checks
