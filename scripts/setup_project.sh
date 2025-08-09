#!/bin/bash
# scripts/setup_project.sh

echo "🚀 建立 SD Multimodal Platform 專案結構..."

# 建立主要目錄結構
mkdir -p backend/{config,api/v1/endpoints,core,schemas,tests}
mkdir -p frontend/{web/src/{components,views,router,store},gradio_app/components,desktop/{ui,resources/{icons,styles}}}
mkdir -p models/{stable-diffusion/{sd-1.5,sdxl,custom},controlnet,lora,vae}
mkdir -p data/{images/{input,output,temp},prompts,logs}
mkdir -p scripts
mkdir -p docs
mkdir -p deployment/{docker,k8s}

# 建立 __init__.py 檔案
find backend frontend -name "*.py" -o -name "*/" | grep -E "(backend|frontend)" | sed 's/$/\/__init__.py/' | xargs -I {} touch {}

# 建立核心檔案
touch backend/__init__.py
touch backend/main.py
touch backend/config/__init__.py
touch backend/config/settings.py
touch backend/config/model_config.py
touch backend/api/__init__.py
touch backend/api/v1/__init__.py
touch backend/api/v1/router.py
touch backend/api/v1/endpoints/{__init__.py,txt2img.py,img2img.py,controlnet.py,models.py}
touch backend/api/dependencies.py
touch backend/core/{__init__.py,sd_pipeline.py,model_loader.py,image_utils.py}
touch backend/schemas/{__init__.py,requests.py,responses.py}
touch backend/tests/{__init__.py,test_api.py}

touch frontend/gradio_app/{__init__.py,app.py,utils.py}
touch frontend/gradio_app/components/{__init__.py,txt2img_tab.py,img2img_tab.py,controlnet_tab.py}
touch frontend/desktop/{__init__.py,main.py}
touch frontend/desktop/ui/{__init__.py,main_window.py,chat_widget.py,generator_widget.py}

# 建立配置檔案
touch .gitignore
touch .env.example
touch README.md
touch requirements.txt
touch environment.yml
touch docker-compose.yml

# 腳本檔案
touch scripts/{download_models.py,benchmark.py}

# 文件檔案
touch docs/{api.md,setup.md,user_guide.md}

# 部署檔案
touch deployment/docker/{Dockerfile.backend,Dockerfile.frontend,nginx.conf}
touch deployment/k8s/{backend-deployment.yaml,frontend-deployment.yaml}

echo "✅ 專案結構建立完成！"
echo "📁 總共建立了 $(find . -type f | wc -l) 個檔案"