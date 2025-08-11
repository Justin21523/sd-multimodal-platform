# SD Multi-Modal Platform - Testing Guide
## Phase 2: Backend Framework & Basic API Services

### 🚀 Quick Start Testing

#### 1. 環境準備
```bash
# 確保在專案根目錄
cd sd-multimodal-platform

# 安裝測試依賴
pip install -r requirements-test.txt

# 或使用測試腳本自動安裝
python scripts/run_tests.py --install-deps
```

#### 2. 快速驗證（30秒內）
```bash
# 運行基本測試套件
python scripts/run_tests.py --unit

# 或直接使用 pytest
pytest tests/test_phase2_api.py::TestHealthEndpoints -v
```

#### 3. 完整測試（5分鐘內）
```bash
# 運行所有測試並生成報告
python scripts/run_tests.py --all --report
```

---

## 📋 測試類別說明

### Unit Tests (單元測試)
- **目標**: 測試個別函數和類別的功能
- **執行時間**: < 30 秒
- **範圍**: API端點、配置管理、工具函數

```bash
# 運行單元測試
pytest -m "unit" -v

# 或使用腳本
python scripts/run_tests.py --unit
```

### Integration Tests (整合測試)
- **目標**: 測試模組間的互動
- **執行時間**: < 2 分鐘
- **範圍**: 中間件、錯誤處理、端到端API流程

```bash
# 運行整合測試
pytest -m "integration" -v

# 或使用腳本
python scripts/run_tests.py --integration
```

### Performance Tests (效能測試)
- **目標**: 驗證回應時間和併發處理能力
- **執行時間**: < 1 分鐘
- **範圍**: API回應時間、併發請求、記憶體使用

```bash
# 運行效能測試
pytest -m "performance" -v -s

# 或使用腳本
python scripts/run_tests.py --performance
```

---

## 🎯 具體測試命令

### 1. 健康檢查測試
```bash
# 測試所有健康檢查端點
pytest tests/test_phase2_api.py::TestHealthEndpoints -v

# 測試特定健康檢查功能
pytest tests/test_phase2_api.py::TestHealthEndpoints::test_health_check_basic -v
```

### 2. 中間件測試
```bash
# 測試請求追蹤中間件
pytest tests/test_phase2_api.py::TestMiddleware -v

# 測試特定中間件功能
pytest tests/test_phase2_api.py::TestMiddleware::test_request_id_middleware -v
```

### 3. 錯誤處理測試
```bash
# 測試錯誤處理機制
pytest tests/test_phase2_api.py::TestErrorHandling -v
```

### 4. 效能基準測試
```bash
# 運行效能基準測試
pytest tests/test_phase2_api.py::TestBenchmarks -v -s
```

---

## 📊 測試覆蓋率要求

### 目標覆蓋率
- **最低要求**: 70%
- **建議目標**: 85%
- **優秀標準**: 90%+

### 查看覆蓋率報告
```bash
# 生成 HTML 覆蓋率報告
pytest --cov=app --cov=utils --cov-report=html:htmlcov tests/

# 在瀏覽器中查看
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### 覆蓋率檢查重點
- **app/main.py**: 中間件和例外處理
- **app/config.py**: 配置驗證邏輯
- **app/api/v1/health.py**: 健康檢查邏輯
- **utils/logging_utils.py**: 日誌格式化

---

## 🐛 除錯與故障排除

### 常見問題

#### 1. 測試無法找到模組
```bash
# 確保 PYTHONPATH 正確設置
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 或使用 pytest 從專案根目錄執行
python -m pytest tests/
```

#### 2. CUDA 相關測試失敗
```bash
# 使用 CPU 模式運行測試
DEVICE=cpu pytest tests/

# 或跳過 CUDA 測試
pytest -k "not cuda" tests/
```

#### 3. 權限錯誤
```bash
# 確保測試目錄有寫入權限
chmod -R 755 tests/
mkdir -p logs outputs models assets
```

#### 4. 依賴版本衝突
```bash
# 重新安裝測試依賴
pip install --force-reinstall -r requirements-test.txt
```

### 除錯技巧

#### 1. 詳細輸出模式
```bash
# 顯示詳細測試輸出
pytest -v -s tests/

# 顯示測試覆蓋率詳情
pytest --cov-report=term-missing tests/
```

#### 2. 只運行失敗的測試
```bash
# 重新運行上次失敗的測試
pytest --lf tests/

# 在第一個失敗時停止
pytest -x tests/
```

#### 3. 測試特定功能
```bash
# 使用關鍵字過濾
pytest -k "health" tests/

# 使用標記過濾
pytest -m "api and not slow" tests/
```

---

## 📈 效能基準

### 預期效能指標

#### API 回應時間
- **健康檢查**: < 50ms (平均)
- **詳細健康檢查**: < 100ms (平均)
- **API 文檔**: < 200ms (平均)

#### 併發處理
- **10 個併發請求**: 全部成功
- **請求 ID 唯一性**: 100%
- **記憶體洩漏**: 無

#### 系統資源
- **記憶體使用**: < 100MB (基本 API)
- **CPU 使用**: < 10% (待機狀態)

### 基準測試命令
```bash
# 運行完整效能測試
python scripts/run_tests.py --performance

# 或直接使用 pytest
pytest tests/test_phase2_api.py::TestBenchmarks::test_health_check_benchmark -v -s
```

---

## 🔧 測試配置

### pytest.ini 配置說明
```ini
[tool:pytest]
# 測試目錄和檔案模式
testpaths = tests
python_files = test_*.py

# 輸出選項
addopts = -v --tb=short --cov=app --cov-fail-under=70

# 測試標記
markers =
    unit: 單元測試
    integration: 整合測試
    api: API 測試
    performance: 效能測試
    slow: 慢速測試

# 環境變數
env =
    TESTING = true
    LOG_LEVEL = WARNING
    DEVICE = cpu
```

### 環境變數配置
```bash
# 測試專用環境變數
export TESTING=true
export LOG_LEVEL=WARNING
export DEVICE=cpu
export MAX_WORKERS=1
```

---

## ✅ 驗收標準

### Phase 2 測試必須全部通過

#### 1. 基本功能測試
- [ ] 健康檢查端點回應正常
- [ ] API 文檔可正常存取
- [ ] 錯誤處理回傳標準格式
- [ ] CORS 配置正確

#### 2. 中間件測試
- [ ] 請求 ID 正確生成
- [ ] 處理時間正確計算
- [ ] 日誌格式正確

#### 3. 效能測試
- [ ] 健康檢查平均回應時間 < 100ms
- [ ] 併發請求處理正常
- [ ] 無記憶體洩漏

#### 4. 覆蓋率要求
- [ ] 整體覆蓋率 ≥ 70%
- [ ] 關鍵模組覆蓋率 ≥ 80%

### 驗收命令
```bash
# 執行完整驗收測試
python scripts/run_tests.py --all

# 檢查覆蓋率
python scripts/run_tests.py --report
```

---

## 🎯 下一步測試規劃

### Phase 3 測試準備
- **模型載入測試**: 測試 Stable Diffusion 模型載入
- **圖像生成測試**: 測試 txt2img API 功能
- **記憶體管理測試**: 測試 VRAM 使用和清理
- **型別安全測試**: 測試 pipeline 結果處理

### 測試基礎設施擴展
- **Mock 模型**: 建立假的 Stable Diffusion 模型用於測試
- **圖像比較**: 添加圖像相似度比較工具
- **效能監控**: 集成 Prometheus 指標收集
- **E2E 測試**: 建立端到端測試流程

---

## 📞 尋求協助

如果測試遇到問題：

1. **檢查環境**: 確保 Python 版本和依賴正確
2. **查看日誌**: 檢查 `logs/` 目錄中的錯誤日誌
3. **運行診斷**: 使用 `python scripts/start_phase2.py --test-only`
4. **查看覆蓋率**: 檢查未測試的程式碼區塊
5. **參考文檔**: 查看 API 文檔了解預期行為

記住：**測試是確保程式碼品質的關鍵，不要跳過測試步驟！** 🎯