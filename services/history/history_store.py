"""
Persistent History v1 storage.

Stores one JSON record per run under:
  <AI_OUTPUT_ROOT>/logs/history

Notes:
- Designed to be lightweight and file-based (no DB required).
- Avoids storing large base64 blobs; inputs are redacted when present.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable

from app.config import settings
from utils.file_utils import ensure_directory


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_dt(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _normalize_dt(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_history_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-")
    if not cleaned:
        raise ValueError("Invalid history_id")
    return cleaned


def _maybe_redact_base64(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if len(value) < 256:
        return value
    if value.startswith("data:image"):
        return "<base64_redacted>"
    # Heuristic: long-ish base64-ish strings without whitespace.
    if re.fullmatch(r"[A-Za-z0-9+/=]+", value):
        return "<base64_redacted>"
    return value


def _redact_input_params(params: Dict[str, Any]) -> Dict[str, Any]:
    redacted: Dict[str, Any] = {}
    for k, v in params.items():
        if k in {"init_image", "mask_image", "image"}:
            redacted[k] = _maybe_redact_base64(v)
            continue
        if k == "controlnet" and isinstance(v, dict):
            cn = dict(v)
            if "image" in cn:
                cn["image"] = _maybe_redact_base64(cn.get("image"))
            redacted[k] = cn
            continue
        redacted[k] = v
    return redacted


def _extract_source_assets(task_type: str, params: Dict[str, Any]) -> Dict[str, Optional[str]]:
    init_asset_id = params.get("init_asset_id")
    mask_asset_id = params.get("mask_asset_id")
    image_asset_id = params.get("image_asset_id") or params.get("asset_id")
    control_asset_id: Optional[str] = None
    cn = params.get("controlnet")
    if isinstance(cn, dict):
        control_asset_id = cn.get("asset_id") or cn.get("image_asset_id")  # type: ignore[assignment]
    if not control_asset_id:
        control_asset_id = params.get("control_asset_id")

    return {
        "image_asset_id": str(image_asset_id) if isinstance(image_asset_id, str) and image_asset_id else None,
        "init_asset_id": str(init_asset_id) if isinstance(init_asset_id, str) and init_asset_id else None,
        "mask_asset_id": str(mask_asset_id) if isinstance(mask_asset_id, str) and mask_asset_id else None,
        "control_asset_id": str(control_asset_id) if isinstance(control_asset_id, str) and control_asset_id else None,
    }


def _extract_output_images(result_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    images: List[Dict[str, Any]] = []
    result = result_data.get("result")
    if isinstance(result, dict) and isinstance(result.get("images"), list):
        for it in result["images"]:
            if isinstance(it, dict):
                images.append(it)
    return images


def _build_index_entry(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    history_id = record.get("history_id")
    if not isinstance(history_id, str) or not history_id:
        return None

    task_id = record.get("task_id") if isinstance(record.get("task_id"), str) else history_id
    task_type = record.get("task_type") if isinstance(record.get("task_type"), str) else ""
    run_mode = record.get("run_mode") if isinstance(record.get("run_mode"), str) else ""
    user_id = record.get("user_id") if isinstance(record.get("user_id"), str) else None
    created_at = record.get("created_at") if isinstance(record.get("created_at"), str) else None

    input_params = record.get("input_params") if isinstance(record.get("input_params"), dict) else {}
    prompt = input_params.get("prompt") if isinstance(input_params.get("prompt"), str) else ""
    negative_prompt = (
        input_params.get("negative_prompt") if isinstance(input_params.get("negative_prompt"), str) else ""
    )
    model_id = input_params.get("model_id") if isinstance(input_params.get("model_id"), str) else ""

    output_urls: List[str] = []
    output_images = record.get("output_images")
    if isinstance(output_images, list):
        for it in output_images:
            if not isinstance(it, dict):
                continue
            url = it.get("image_url") or it.get("url")
            if isinstance(url, str) and url:
                output_urls.append(url)

    return {
        "version": "history.index.v1",
        "history_id": history_id,
        "task_id": task_id,
        "task_type": task_type,
        "run_mode": run_mode,
        "user_id": user_id,
        "created_at": created_at,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "model_id": model_id,
        "output_image_urls": output_urls[:8],
    }


def _iter_jsonl_reverse(path: Path, *, chunk_size: int = 8192) -> Iterable[Dict[str, Any]]:
    """
    Iterate JSONL entries newest-first without loading the full file into memory.
    """
    if not path.exists():
        return []

    def _gen() -> Iterable[Dict[str, Any]]:
        with open(path, "rb") as f:
            f.seek(0, 2)
            pos = f.tell()
            buffer = b""

            while pos > 0:
                read_size = min(chunk_size, pos)
                pos -= read_size
                f.seek(pos)
                data = f.read(read_size)
                buffer = data + buffer

                parts = buffer.split(b"\n")
                buffer = parts[0]
                for raw in reversed(parts[1:]):
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line.decode("utf-8"))
                    except Exception:
                        continue
                    if isinstance(item, dict):
                        yield item

            tail = buffer.strip()
            if tail:
                try:
                    item = json.loads(tail.decode("utf-8"))
                except Exception:
                    item = None
                if isinstance(item, dict):
                    yield item

    return _gen()


def _matches_q_index(entry: Dict[str, Any], q: str) -> bool:
    q_norm = q.strip().lower()
    if not q_norm:
        return True

    candidates: List[str] = []
    for key in ("history_id", "task_id", "task_type", "run_mode", "user_id", "prompt", "negative_prompt", "model_id"):
        val = entry.get(key)
        if isinstance(val, str) and val:
            candidates.append(val)

    urls = entry.get("output_image_urls")
    if isinstance(urls, list):
        for u in urls:
            if isinstance(u, str) and u:
                candidates.append(u)

    return any(q_norm in s.lower() for s in candidates)


class HistoryStore:
    def __init__(self, history_dir: Optional[Path] = None):
        output_root = Path(str(settings.OUTPUT_PATH)).expanduser().resolve().parent
        self.history_dir = history_dir or (output_root / "logs" / "history")
        self.index_path = self.history_dir / "index.jsonl"
        ensure_directory(self.history_dir)

    def _record_path(self, history_id: str) -> Path:
        safe_id = _safe_history_id(history_id)
        return self.history_dir / f"{safe_id}.json"

    def _iter_files(self) -> Iterable[Path]:
        return sorted(
            self.history_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    def _matches_q(self, record: Dict[str, Any], q: str) -> bool:
        q_norm = q.strip().lower()
        if not q_norm:
            return True

        def _add(val: Any, bucket: List[str]) -> None:
            if isinstance(val, str) and val:
                bucket.append(val)

        candidates: List[str] = []
        for key in ("history_id", "task_id", "task_type", "run_mode", "user_id"):
            _add(record.get(key), candidates)

        params = record.get("input_params")
        if isinstance(params, dict):
            for key in ("prompt", "negative_prompt", "model_id"):
                _add(params.get(key), candidates)

        for img in record.get("output_images") or []:
            if isinstance(img, dict):
                _add(img.get("image_url") or img.get("url"), candidates)

        return any(q_norm in s.lower() for s in candidates)

    def list_records(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        q: Optional[str] = None,
        task_type: Optional[str] = None,
        user_id: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        task_type_norm = task_type.strip().lower() if isinstance(task_type, str) and task_type.strip() else None
        user_id_norm = user_id.strip() if isinstance(user_id, str) and user_id.strip() else None
        q_norm = q.strip() if isinstance(q, str) and q.strip() else None
        since_dt = _normalize_dt(since)
        until_dt = _normalize_dt(until)

        records: List[Dict[str, Any]] = []
        seen = 0

        if not self.index_path.exists():
            try:
                self.rebuild_index()
            except Exception:
                pass

        if self.index_path.exists():
            seen_ids: set[str] = set()
            for entry in _iter_jsonl_reverse(self.index_path):
                history_id = entry.get("history_id")
                if not isinstance(history_id, str) or not history_id:
                    continue
                if history_id in seen_ids:
                    continue
                seen_ids.add(history_id)

                if task_type_norm:
                    et = entry.get("task_type")
                    if not (isinstance(et, str) and et.strip().lower() == task_type_norm):
                        continue

                if user_id_norm is not None:
                    eu = entry.get("user_id")
                    if not (isinstance(eu, str) and eu == user_id_norm):
                        continue

                created = _parse_dt(entry.get("created_at"))
                if created is not None:
                    if since_dt is not None and created < since_dt:
                        continue
                    if until_dt is not None and created > until_dt:
                        continue

                if q_norm and not _matches_q_index(entry, q_norm):
                    continue

                if seen < offset:
                    seen += 1
                    continue
                if len(records) >= limit:
                    break

                path = self._record_path(history_id)
                if not path.exists():
                    continue
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        record = json.load(f)
                except Exception:
                    continue
                if not isinstance(record, dict):
                    continue

                records.append(record)
                seen += 1

            return records

        # Fallback: scan files (slower, but robust).
        files = sorted(
            self.history_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for path in files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    record = json.load(f)
            except Exception:
                continue

            if not isinstance(record, dict):
                continue

            if task_type_norm:
                rt = record.get("task_type")
                if not (isinstance(rt, str) and rt.strip().lower() == task_type_norm):
                    continue

            if user_id_norm is not None:
                ru = record.get("user_id")
                if not (isinstance(ru, str) and ru == user_id_norm):
                    continue

            created = _parse_dt(record.get("created_at"))
            if created is not None:
                if since_dt is not None and created < since_dt:
                    continue
                if until_dt is not None and created > until_dt:
                    continue

            if q_norm and not self._matches_q(record, q_norm):
                continue

            if seen < offset:
                seen += 1
                continue
            if len(records) >= limit:
                break

            records.append(record)
            seen += 1
        return records

    def cleanup_records(self, *, older_than_days: int) -> Dict[str, Any]:
        if older_than_days <= 0:
            raise ValueError("older_than_days must be > 0")

        cutoff = datetime.now(timezone.utc) - timedelta(days=int(older_than_days))
        deleted = 0
        deleted_ids: List[str] = []

        for path in self.history_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    record = json.load(f)
            except Exception:
                record = None

            created = _parse_dt(record.get("created_at")) if isinstance(record, dict) else None
            if created is None:
                try:
                    created = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                except Exception:
                    created = None

            if created is None or created > cutoff:
                continue

            try:
                history_id = record.get("history_id") if isinstance(record, dict) else None
                path.unlink(missing_ok=True)
                deleted += 1
                if isinstance(history_id, str) and history_id:
                    deleted_ids.append(history_id)
            except Exception:
                continue

        index_info: Optional[Dict[str, Any]] = None
        try:
            index_info = self.rebuild_index()
        except Exception:
            index_info = None

        return {
            "deleted": deleted,
            "deleted_ids": deleted_ids,
            "cutoff": cutoff.isoformat(),
            "index": index_info,
        }

    def get_record(self, history_id: str) -> Optional[Dict[str, Any]]:
        path = self._record_path(history_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def write_record(self, record: Dict[str, Any]) -> Path:
        history_id = record.get("history_id")
        if not isinstance(history_id, str) or not history_id:
            raise ValueError("history_id is required")
        path = self._record_path(history_id)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        tmp.replace(path)

        try:
            entry = _build_index_entry(record)
            if entry is not None:
                with open(self.index_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            # Index is best-effort: list_records can rebuild if needed.
            pass

        return path

    def rebuild_index(self) -> Dict[str, Any]:
        """
        Rebuild history index.jsonl from existing record files.

        Useful after bulk deletions or when upgrading from older history formats.
        """
        tmp = self.index_path.with_suffix(".jsonl.tmp")
        files = sorted(self.history_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        entries: List[Dict[str, Any]] = []
        for path in files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    record = json.load(f)
            except Exception:
                continue
            if not isinstance(record, dict):
                continue
            entry = _build_index_entry(record)
            if entry is not None:
                entries.append(entry)

        with open(tmp, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        tmp.replace(self.index_path)
        return {"entries": len(entries), "index_path": str(self.index_path)}

    def record_completion(
        self,
        *,
        history_id: str,
        task_type: str,
        run_mode: str,
        user_id: Optional[str],
        input_params: Dict[str, Any],
        result_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        safe_id = _safe_history_id(history_id)
        redacted_params = _redact_input_params(input_params)
        record = {
            "version": "history.v1",
            "history_id": safe_id,
            "task_id": history_id,
            "task_type": task_type,
            "run_mode": run_mode,
            "user_id": user_id,
            "created_at": _utc_now_iso(),
            "input_params": redacted_params,
            "source_assets": _extract_source_assets(task_type, redacted_params),
            "output_images": _extract_output_images(result_data),
            "result_data": result_data,
        }
        self.write_record(record)
        return record


def get_history_store() -> HistoryStore:
    return HistoryStore()
