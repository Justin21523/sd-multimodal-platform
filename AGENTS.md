# Repository Guidelines

## Project Structure & Module Organization
- `app/`: primary FastAPI service (`app/main.py`), middleware, schemas, and workers.
- `services/`: model registry, generation pipelines, queue system, and post-processing.
- `utils/`: shared utilities (logging, image/file helpers).
- `tests/`: pytest suite and fixtures (`tests/conftest.py`).
- `scripts/`: local tooling (env setup, smoke checks, model install/download).
- `frontend/react/`: React + TypeScript (Vite) frontend (preferred UI for this repo).
- `frontend/gradio_app/`, `frontend/desktop/`, `frontend/web/`, `backend/`: legacy prototypes (keep read-only unless migrating).

**Storage layout (must follow `~/Desktop/data_model_structure.md`)**
- Models: `/mnt/c/ai_models`
- Caches: `/mnt/c/ai_cache` (set `HF_HOME`, `TRANSFORMERS_CACHE`, `TORCH_HOME`, `XDG_CACHE_HOME`)
- Outputs/assets/logs: `/mnt/data/training/runs/sd-multimodal-platform/{outputs,assets,logs}`

## Build, Test, and Development Commands
```bash
conda activate ai_env
python scripts/setup_env.py
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
redis-server  # optional (for async queue)
celery -A app.workers.celery_worker worker --loglevel=info --queues=generation,postprocess
pytest
cd frontend/react && npm install && npm run dev
```
- Frontend env: `VITE_API_BASE_URL=http://localhost:8000` (default).
- Async queue requires Redis + Celery workers; without them the API still boots and queue endpoints return 503.

## Coding Style & Naming Conventions
- Python: 4-space indentation, type hints at API/service boundaries; format with `black .`.
- React/TS: components `PascalCase`, hooks `useX`, variables `camelCase`; keep UI in `frontend/react/src/`.
- Prefer URLs over absolute filesystem paths in API responses (mounted via `/outputs` and `/assets`).

## Testing Guidelines
- Framework: `pytest` with markers (`unit`, `integration`, `api`, `performance`, `slow`).
- Conventions: files `tests/test_*.py`, classes `Test*`, functions `test_*`.
- Coverage: enforced via `pytest.ini` (`--cov=app`, fail-under 80%).
- Example: `pytest -m unit -v` or `pytest tests/test_phase2_api.py::TestHealthEndpoints -v`

## Commit & Pull Request Guidelines
- Use Conventional Commits seen in history: `feat(queue): ...`, `fix(api): ...`, `refactor(utils): ...`, `chore(config): ...`, `test(...): ...`.
- Target `develop` for feature work; branch patterns: `feature/<area>-<desc>`, `fix/<desc>`.
- PRs: describe the behavior change, include test command/output, and attach screenshots/logs for API/UI changes; link issues when available.

## Security & Configuration Tips
- Copy `.env.example` → `.env`; never commit secrets, tokens, datasets, or model weights.
- Keep large files out of Git; store models/caches/outputs under `/mnt/c` + `/mnt/data` per the storage spec.
