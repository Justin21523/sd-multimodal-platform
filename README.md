# SD Multi-Modal Platform

## ✅ 快速開始（目前推薦流程）

- 啟用環境：`conda activate ai_env`
- 依照 `~/Desktop/data_model_structure.md`：
  - 模型：`/mnt/c/ai_models`
  - 快取：`/mnt/c/ai_cache`（建議設定 `HF_HOME`/`TRANSFORMERS_CACHE`/`TORCH_HOME`/`XDG_CACHE_HOME`）
  - 產出：`/mnt/data/training/runs/sd-multimodal-platform/outputs`
- 設定環境：`cp .env.example .env`（不要提交 secrets）
- 啟動後端：`uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
- （可選）啟用非同步佇列（Redis + Celery worker）：
  - Redis（擇一）：`redis-server` 或 `docker run -p 6379:6379 redis:7-alpine`
  - Worker：`celery -A app.workers.celery_worker worker --loglevel=info --queues=generation,postprocess`
- 啟動前端（React）：`cd frontend/react && npm install && VITE_API_BASE_URL=http://localhost:8000 npm run dev`
- API 文件：`http://localhost:8000/api/v1/docs`（健康檢查：`http://localhost:8000/health`）

> 備註：下方的「資料夾架構」區塊包含早期原型（`backend/`、`frontend/web/` 等），目前主要後端以 `app/` 為準、主要前端以 `frontend/react/` 為準。

## 🐳 Docker Compose（容器化啟動）

> 重要：容器仍遵守 `~/Desktop/data_model_structure.md`。預設會 bind-mount：
> - 模型：`${AI_MODELS_ROOT:-/mnt/c/ai_models}`
> - 快取：`${AI_CACHE_ROOT:-/mnt/c/ai_cache}`
> - 產出：`${AI_OUTPUT_ROOT:-/mnt/data/training/runs/sd-multimodal-platform}`

1) （建議）先準備 `.env`（用於 compose 變數插值；不會提交）：
```bash
cp .env.example .env
```

2) 啟動後端 + Redis + Celery workers（CPU/auto 模式）：
```bash
docker compose up --build
```

3) 啟動前端（Vite dev server，預設 profile）：
```bash
docker compose --profile frontend up --build
```

4) GPU（需要 NVIDIA Container Toolkit / Docker Desktop GPU 支援）：
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

常用連結：
- API：`http://localhost:8000/`（Docs：`http://localhost:8000/api/v1/docs`）
- Frontend（profile=frontend）：`http://localhost:5173/`

# 📁 專案資料夾架構

```
sd-multimodal-platform/
├── .gitignore
├── .env.example
├── README.md
├── requirements.txt
├── docker-compose.yml
├── environment.yml           # Conda 環境配置
│
├── backend/                  # 後端 API 服務
│   ├── __init__.py
│   ├── main.py              # FastAPI 主程式
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py      # 環境變數配置
│   │   └── model_config.py  # 模型路徑配置
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── endpoints/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── txt2img.py
│   │   │   │   ├── img2img.py
│   │   │   │   ├── controlnet.py
│   │   │   │   └── models.py
│   │   │   └── router.py
│   │   └── dependencies.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── sd_pipeline.py   # SD 推理核心
│   │   ├── model_loader.py  # 模型載入管理
│   │   └── image_utils.py   # 圖片處理工具
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── requests.py      # API 請求模型
│   │   └── responses.py     # API 回應模型
│   └── tests/
│       ├── __init__.py
│       └── test_api.py
│
├── frontend/                 # 前端界面
│   ├── web/                 # Web 界面 (Vue.js/React)
│   │   ├── package.json
│   │   ├── src/
│   │   │   ├── components/
│   │   │   │   ├── ChatInterface.vue
│   │   │   │   ├── ImageGenerator.vue
│   │   │   │   └── ParameterPanel.vue
│   │   │   ├── views/
│   │   │   ├── router/
│   │   │   ├── store/
│   │   │   └── main.js
│   │   └── dist/
│   │
│   ├── gradio_app/          # Gradio WebUI
│   │   ├── __init__.py
│   │   ├── app.py           # Gradio 主程式
│   │   ├── components/
│   │   │   ├── __init__.py
│   │   │   ├── txt2img_tab.py
│   │   │   ├── img2img_tab.py
│   │   │   └── controlnet_tab.py
│   │   └── utils.py
│   │
│   └── desktop/             # 桌面應用 (PyQt6)
│       ├── __init__.py
│       ├── main.py          # 桌面應用主程式
│       ├── ui/
│       │   ├── __init__.py
│       │   ├── main_window.py
│       │   ├── chat_widget.py
│       │   └── generator_widget.py
│       └── resources/
│           ├── icons/
│           └── styles/
│
├── models/                   # 模型檔案目錄
│   ├── stable-diffusion/
│   │   ├── sd-1.5/
│   │   ├── sdxl/
│   │   └── custom/
│   ├── controlnet/
│   ├── lora/
│   └── vae/
│
├── data/                    # 資料目錄
│   ├── images/
│   │   ├── input/
│   │   ├── output/
│   │   └── temp/
│   ├── prompts/
│   └── logs/
│
├── scripts/                 # 工具腳本
│   ├── download_models.py   # 模型下載腳本
│   ├── setup.py            # 環境設置
│   └── benchmark.py        # 性能測試
│
├── docs/                    # 文件
│   ├── api.md
│   ├── setup.md
│   └── user_guide.md
│
└── deployment/              # 部署配置
    ├── docker/
    │   ├── Dockerfile.backend
    │   ├── Dockerfile.frontend
    │   └── nginx.conf
    └── k8s/
        ├── backend-deployment.yaml
        └── frontend-deployment.yaml
```

## 🐍 Conda 環境配置

### 主環境：`sd-platform`
```yaml
name: sd-platform
channels:
  - nvidia
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  # PyTorch (穩定＋廣泛相容)
  - pytorch=2.3.*
  - torchvision
  - torchaudio
  - pytorch-cuda=12.1 # 由 nvidia channel 提供；取代 cudatoolkit
  - pip
  - pip:
      # Core ML
      - diffusers>=0.29.0
      - transformers>=4.43.0
      - accelerate>=0.31.0
      - xformers==0.0.27.post2 # 與 torch 2.3 + cu121 相容
      - controlnet-aux>=0.6.0
      - safetensors>=0.4.3
      - einops>=0.7.0
      - huggingface_hub>=0.24.0

      # API & Web
      - fastapi>=0.111.0
      - uvicorn[standard]>=0.30.0
      - gradio>=4.36.0
      - pydantic>=2.8.0
      - python-multipart
      - aiofiles

      # Desktop UI
      - PyQt6>=6.6.0
      - PyQt6-tools

      # Utils
      - pillow>=10.3.0
      - opencv-python-headless>=4.10.0.84
      - numpy>=1.26.4
      - scipy>=1.13.0

      # Test
      - pytest>=8.2.0

```

## 🔧 Git 工作流程與分支策略

### 分支命名規範
- `main` - 穩定發布版本
- `develop` - 開發主分支
- `feature/backend-api` - 後端API功能
- `feature/gradio-ui` - Gradio界面
- `feature/desktop-app` - 桌面應用
- `feature/chat-interface` - 聊天界面
- `fix/model-loading` - Bug修復
- `docs/api-documentation` - 文件更新
