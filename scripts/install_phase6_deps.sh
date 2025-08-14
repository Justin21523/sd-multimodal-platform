#!/bin/bash
# install_phase6_deps.sh - Install Phase 6 dependencies

echo "🚀 Installing Phase 6 dependencies..."

# Core queue dependencies
echo "📦 Installing core queue dependencies..."
pip install redis celery flower

# Image processing dependencies
echo "🖼️ Installing image processing dependencies..."
pip install opencv-python opencv-contrib-python
pip install aiofiles aiohttp

# Real-ESRGAN dependencies
echo "🔍 Installing Real-ESRGAN dependencies..."
pip install realesrgan
pip install basicsr

# GFPGAN dependencies
echo "👤 Installing GFPGAN dependencies..."
pip install gfpgan

# CodeFormer dependencies (optional)
echo "🎭 Installing CodeFormer dependencies (optional)..."
pip install facexlib

# Additional utilities
echo "🛠️ Installing additional utilities..."
pip install psutil  # For system monitoring
pip install prometheus-client  # For metrics

# Create weights directory
echo "📁 Creating weights directories..."
mkdir -p weights
mkdir -p models

# Download sample models (optional)
echo "⬇️ Downloading sample models..."

# Create download script for models
cat > download_models.py << 'EOF'
#!/usr/bin/env python3
import os
import urllib.request
from pathlib import Path

def download_file(url, filepath):
    """Download file with progress"""
    print(f"Downloading {filepath.name}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"✅ Downloaded {filepath.name}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filepath.name}: {e}")
        return False

# Model URLs
models = {
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
}

# Create weights directory
weights_dir = Path("weights")
weights_dir.mkdir(exist_ok=True)

# Download models
for filename, url in models.items():
    filepath = weights_dir / filename
    if not filepath.exists():
        download_file(url, filepath)
    else:
        print(f"⏭️ {filename} already exists")

print("✅ Model download completed!")
EOF

python download_models.py

# Create requirements.txt for Phase 6
echo "📋 Creating requirements.txt..."
cat > requirements_phase6.txt << 'EOF'
# Phase 6 - Queue System & Rate Limiting Dependencies

# Core FastAPI and async
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
pydantic-settings>=2.1.0

# Queue system
redis>=5.0.0
celery>=5.3.0
flower>=2.0.0

# AI/ML frameworks
torch>=2.1.0
torchvision>=0.16.0
diffusers>=0.24.0
transformers>=4.35.0
accelerate>=0.24.0
xformers>=0.0.22

# Image processing
Pillow>=10.0.0
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0
numpy>=1.24.0

# Post-processing models
realesrgan>=0.3.0
basicsr>=1.4.2
gfpgan>=1.3.8
facexlib>=0.3.0

# Utilities
aiofiles>=23.2.0
aiohttp>=3.9.0
python-multipart>=0.0.6

# Monitoring and metrics
psutil>=5.9.0
prometheus-client>=0.19.0

# Development and testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-mock>=3.12.0
pytest-benchmark>=4.0.0

# Logging and configuration
python-dotenv>=1.0.0
PyYAML>=6.0.1
EOF

# Install from requirements
echo "📦 Installing from requirements.txt..."
pip install -r requirements_phase6.txt

# Verify installations
echo "🔍 Verifying installations..."

python -c "
import sys
modules = [
    'redis', 'celery', 'cv2', 'PIL', 'torch',
    'diffusers', 'transformers', 'accelerate'
]

missing = []
for module in modules:
    try:
        __import__(module)
        print(f'✅ {module}')
    except ImportError:
        print(f'❌ {module}')
        missing.append(module)

if missing:
    print(f'⚠️ Missing modules: {missing}')
    sys.exit(1)
else:
    print('🎉 All core dependencies installed successfully!')
"

# Optional: Install advanced dependencies
echo "🚀 Installing advanced dependencies (optional)..."
echo "These may require compilation and take longer..."

# xFormers for memory optimization
pip install xformers --index-url https://download.pytorch.org/whl/cu121 || echo "⚠️ xFormers installation failed (optional)"

# Advanced face restoration models
pip install codeformer || echo "⚠️ CodeFormer installation failed (optional)"

echo "✅ Phase 6 dependency installation completed!"
echo ""
echo "📋 Next steps:"
echo "1. Start Redis: docker run -d --name redis -p 6379:6379 redis:7-alpine"
echo "2. Test installation: python -c 'from app.core.queue_manager import QueueManager; print(\"Queue manager import OK\")'"
echo "3. Run application: python -m uvicorn app.main:app --reload"
echo "4. Start workers: celery -A app.workers.celery_worker worker --loglevel=info"
echo "5. Monitor with Flower: celery -A app.workers.celery_worker flower"