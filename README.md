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
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch=2.1.0
  - torchvision
  - torchaudio
  - cudatoolkit=11.8
  - pip
  - pip:
    # Core ML
    - diffusers>=0.24.0
    - transformers>=4.35.0
    - accelerate>=0.24.0
    - xformers>=0.0.22
    - controlnet-aux>=0.4.0

    # API & Web
    - fastapi>=0.104.0
    - uvicorn[standard]>=0.24.0
    - gradio>=4.8.0
    - streamlit>=1.28.0

    # Desktop UI
    - PyQt6>=6.6.0
    - PyQt6-tools

    # Utils
    - pillow>=10.0.0
    - opencv-python>=4.8.0
    - numpy>=1.24.0
    - pydantic>=2.4.0
    - python-multipart
    - aiofiles
    - pytest
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