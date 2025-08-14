# utils/file_utils.py
"""
File system utilities for SD Multi-Modal Platform.
Handles file operations, directory management, and safe filename generation.
"""

import os
import re
import time
import shutil
import aiofiles
import uuid
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Union
from PIL import Image
from app.config import settings
from utils.logging_utils import setup_logging, get_request_logger


logger = logging.getLogger(__name__)


class FileManager:
    """檔案管理工具類"""

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path("./outputs")
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def save_image(
        self,
        image: Image.Image,
        filename: str,
        subfolder: str = "",
        user_id: Optional[str] = None,
    ) -> Path:
        """異步保存圖像"""
        # 構建保存路徑
        save_dir = self.base_path / subfolder
        if user_id:
            save_dir = save_dir / user_id

        save_dir.mkdir(parents=True, exist_ok=True)

        # 處理檔名衝突
        save_path = save_dir / filename
        counter = 1
        while save_path.exists():
            name_parts = filename.rsplit(".", 1)
            if len(name_parts) == 2:
                new_filename = f"{name_parts[0]}_{counter}.{name_parts[1]}"
            else:
                new_filename = f"{filename}_{counter}"
            save_path = save_dir / new_filename
            counter += 1

        # 保存圖像
        image.save(save_path)
        return save_path

    async def save_text(
        self,
        content: str,
        filename: str,
        subfolder: str = "",
        user_id: Optional[str] = None,
    ) -> Path:
        """異步保存文字檔案"""
        save_dir = self.base_path / subfolder
        if user_id:
            save_dir = save_dir / user_id

        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename

        async with aiofiles.open(save_path, "w", encoding="utf-8") as f:
            await f.write(content)

        return save_path

    async def read_text(self, file_path: Path) -> str:
        """異步讀取文字檔案"""
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            return await f.read()

    async def delete_file(self, file_path: Path) -> bool:
        """刪除檔案"""
        try:
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception:
            return False

    async def cleanup_old_files(self, days: int = 7, subfolder: str = "") -> int:
        """清理舊檔案"""
        cleanup_dir = self.base_path / subfolder if subfolder else self.base_path
        if not cleanup_dir.exists():
            return 0

        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_count = 0

        for file_path in cleanup_dir.rglob("*"):
            if file_path.is_file():
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime < cutoff_date:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception:
                        pass

        return deleted_count

    def get_file_size(self, file_path: Path) -> int:
        """獲取檔案大小（位元組）"""
        return file_path.stat().st_size if file_path.exists() else 0

    def list_files(self, subfolder: str = "", pattern: str = "*") -> List[Path]:
        """列出檔案"""
        search_dir = self.base_path / subfolder if subfolder else self.base_path
        if search_dir.exists():
            return list(search_dir.glob(pattern))
        return []


def ensure_directory(path: Path) -> Path:
    """
    Create directory if it doesn't exist.

    Args:
        path: Directory path to create

    Returns:
        Path: The created/existing directory path
    """
    try:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise


def safe_filename(filename: str, max_length: int = 50) -> str:
    """
    Generate a safe filename by removing/replacing special characters.

    Args:
        filename: Original filename
        max_length: Maximum length of resulting filename

    Returns:
        str: Safe filename
    """
    # Remove or replace unsafe characters
    safe_name = re.sub(r"[^\w\s-]", "", filename)
    safe_name = re.sub(r"[-\s]+", "-", safe_name)  # Replace spaces with underscores
    safe_name = re.sub(r"_{2,}", "_", safe_name)  # Replace multiple underscores

    # Truncate if too long, preserving extension
    if len(safe_name) > max_length:
        name_part, ext_part = os.path.splitext(safe_name)
        available_length = max_length - len(ext_part)
        safe_name = name_part[:available_length] + ext_part

    # Ensure it's not empty
    if not safe_name:
        safe_name = "unnamed"

    return safe_name


def get_file_size(file_path: Union[str, Path]) -> int:
    """Get file size in bytes"""
    try:
        return Path(file_path).stat().st_size
    except (OSError, FileNotFoundError):
        return 0


def get_directory_size(directory: Path) -> int:
    """
    Calculate total directory size in bytes.

    Args:
        directory: Directory to measure

    Returns:
        int: Total size in bytes
    """
    total_size = 0
    directory = Path(directory)
    try:
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
    except Exception as e:
        logger.error(f"Error calculating directory size for {directory}: {e}")

    return total_size


def cleanup_old_files(directory: Path, days: int) -> int:
    """
    Remove files older than specified days.

    Args:
        directory: Directory to clean
        days: Files older than this many days will be removed

    Returns:
        int: Number of files removed
    """
    if days <= 0:
        return 0

    directory = Path(directory)
    if not directory.exists():
        return 0

    cutoff_time = time.time() - (days * 24 * 60 * 60)
    removed_count = 0

    try:
        for file_path in Path(directory).rglob("*"):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(f"🗑️ Deleted old file: {file_path}")
                except OSError as e:
                    logger.warning(f"Failed to delete {file_path}: {e}")
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old files from {directory}")

    except (OSError, PermissionError) as e:
        logger.error(f"Failed to cleanup directory {directory}: {e}")

    return deleted_count


def copy_file_safe(src: Path, dst: Path) -> bool:
    """
    Safely copy a file with error handling.

    Args:
        src: Source file path
        dst: Destination file path

    Returns:
        bool: True if copy successful
    """
    try:
        src = Path(src)
        dst = Path(dst)

        # Ensure source file exists
        ensure_directory(dst.parent)

        # Copy the file
        shutil.copy2(src, dst)
        logger.debug(f"Copied {src} to {dst}")
        return True

    except Exception as exc:
        logger.error(f"Failed to copy {src} to {dst}: {exc}")
        return False


def move_file_safe(src: Union[str, Path], dst: Union[str, Path]) -> bool:
    """Safely move file with error handling"""
    try:
        src_path = Path(src)
        dst_path = Path(dst)

        # Ensure destination directory exists
        ensure_directory(dst_path.parent)

        # Move file
        shutil.move(str(src_path), str(dst_path))
        return True
    except Exception as e:
        logger.error(f"Failed to move {src} to {dst}: {e}")
        return False


def generate_unique_filename(
    base_name: str,
    extension: str,
    directory: Union[str, Path],
    include_timestamp: bool = True,
) -> str:
    """Generate unique filename in directory"""
    directory = Path(directory)

    # Prepare base filename
    safe_base = safe_filename(base_name)
    if include_timestamp:
        timestamp = int(time.time() * 1000)  # milliseconds
        filename = f"{timestamp}_{safe_base}.{extension.lstrip('.')}"
    else:
        filename = f"{safe_base}.{extension.lstrip('.')}"

    # Ensure uniqueness
    counter = 1
    original_filename = filename
    while (directory / filename).exists():
        name_part, ext_part = os.path.splitext(original_filename)
        filename = f"{name_part}_{counter:03d}{ext_part}"
        counter += 1

    return filename


async def save_generation_output(
    image: Image.Image,
    task_id: str,
    subfolder: str = "txt2img",
    filename: Optional[str] = None,
    format: str = "PNG",
    quality: int = 95,
) -> Path:
    """
    Save generated image to output directory with proper naming and metadata
    """
    try:
        # Prepare output directory
        output_dir = Path(settings.OUTPUT_PATH) / subfolder
        ensure_directory(output_dir)

        # Generate filename if not provided
        if filename is None:
            timestamp = int(time.time() * 1000)
            filename = f"{task_id}_{timestamp}.{format.lower()}"
        else:
            filename = safe_filename(filename)

        # Ensure unique filename
        file_path = output_dir / filename
        counter = 1
        original_path = file_path
        while file_path.exists():
            stem = original_path.stem
            suffix = original_path.suffix
            file_path = output_dir / f"{stem}_{counter:03d}{suffix}"
            counter += 1

        # Save image
        save_kwargs = {"format": format}
        if format.upper() == "JPEG":
            save_kwargs["quality"] = quality  # type: ignore[arg-type]
            save_kwargs["optimize"] = True  # type: ignore[arg-type]
            # Convert RGBA to RGB for JPEG
            if image.mode == "RGBA":
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(
                    image, mask=image.split()[-1] if image.mode == "RGBA" else None
                )
                image = background

        image.save(file_path, **save_kwargs)

        logger.info(f"💾 Saved generation output: {file_path}")
        return file_path

    except Exception as e:
        logger.error(f"❌ Failed to save generation output: {str(e)}")
        raise


def get_output_url(file_path: Union[str, Path], base_url: str = "") -> str:
    """Convert file path to URL for API responses"""
    file_path = Path(file_path)

    # Get relative path from output directory
    try:
        relative_path = file_path.relative_to(Path(settings.OUTPUT_PATH))
        url = f"{base_url}/outputs/{relative_path}".replace("\\", "/")
        return url
    except ValueError:
        # File is not in output directory, return absolute path
        return str(file_path)


def list_generation_outputs(
    subfolder: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    file_extension: Optional[str] = None,
) -> List[dict]:
    """List generated output files with metadata"""
    try:
        output_dir = Path(settings.OUTPUT_PATH)
        if subfolder:
            output_dir = output_dir / subfolder

        if not output_dir.exists():
            return []

        # Collect files
        files = []
        pattern = f"*.{file_extension}" if file_extension else "*"

        for file_path in output_dir.rglob(pattern):
            if file_path.is_file():
                stat = file_path.stat()
                files.append(
                    {
                        "filename": file_path.name,
                        "path": str(file_path),
                        "url": get_output_url(file_path),
                        "size": stat.st_size,
                        "created_at": stat.st_mtime,
                        "subfolder": str(
                            file_path.parent.relative_to(Path(settings.OUTPUT_PATH))
                        ),
                    }
                )

        # Sort by creation time (newest first)
        files.sort(key=lambda x: x["created_at"], reverse=True)

        # Apply pagination
        return files[offset : offset + limit]

    except Exception as e:
        logger.error(f"❌ Failed to list generation outputs: {str(e)}")
        return []


def cleanup_generation_outputs(days: int = 7, dry_run: bool = False) -> dict:
    """Clean up old generation outputs"""
    try:
        output_dir = Path(settings.OUTPUT_PATH)
        if not output_dir.exists():
            return {"deleted_files": 0, "freed_space": 0}

        cutoff_time = time.time() - (days * 24 * 60 * 60)
        deleted_files = 0
        freed_space = 0

        for file_path in output_dir.rglob("*"):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                file_size = file_path.stat().st_size

                if not dry_run:
                    try:
                        file_path.unlink()
                        logger.info(f"🗑️ Deleted old output: {file_path}")
                    except OSError as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")
                        continue

                deleted_files += 1
                freed_space += file_size

        return {
            "deleted_files": deleted_files,
            "freed_space": freed_space,
            "freed_space_mb": round(freed_space / 1024 / 1024, 2),
        }

    except Exception as e:
        logger.error(f"❌ Failed to cleanup outputs: {str(e)}")
        return {"deleted_files": 0, "freed_space": 0}


def archive_generation_outputs(
    archive_name: Optional[str] = None,
    subfolder: Optional[str] = None,
    days_old: int = 30,
) -> Optional[Path]:
    """Archive old generation outputs to ZIP file"""
    try:
        import zipfile

        output_dir = Path(settings.OUTPUT_PATH)
        if subfolder:
            output_dir = output_dir / subfolder

        if not output_dir.exists():
            return None

        # Prepare archive name
        if archive_name is None:
            timestamp = int(time.time())
            archive_name = f"archive_{timestamp}.zip"

        archive_path = Path(settings.OUTPUT_PATH) / "archives" / archive_name
        ensure_directory(archive_path.parent)

        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        archived_files = 0

        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in output_dir.rglob("*"):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    # Add to archive with relative path
                    arcname = file_path.relative_to(output_dir)
                    zipf.write(file_path, arcname)
                    archived_files += 1

        if archived_files > 0:
            logger.info(f"📦 Archived {archived_files} files to {archive_path}")
            return archive_path
        else:
            # Remove empty archive
            archive_path.unlink()
            return None

    except Exception as e:
        logger.error(f"❌ Failed to archive outputs: {str(e)}")
        return None


def validate_file_type(
    file_path: Union[str, Path], allowed_extensions: List[str]
) -> bool:
    """Validate file type based on extension"""
    file_path = Path(file_path)
    extension = file_path.suffix.lower().lstrip(".")
    return extension in [ext.lower().lstrip(".") for ext in allowed_extensions]


def get_file_hash(file_path: Union[str, Path]) -> str:
    """Calculate SHA256 hash of file"""
    import hashlib

    try:
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"❌ Failed to calculate hash for {file_path}: {str(e)}")
        return ""


def create_directory_structure(base_path: Union[str, Path], structure: dict) -> bool:
    """Create directory structure from nested dict"""
    try:
        base_path = Path(base_path)

        def create_nested(current_path: Path, struct: dict):
            for name, content in struct.items():
                new_path = current_path / name
                if isinstance(content, dict):
                    ensure_directory(new_path)
                    create_nested(new_path, content)
                else:
                    ensure_directory(new_path)

        create_nested(base_path, structure)
        return True

    except Exception as e:
        logger.error(f"❌ Failed to create directory structure: {str(e)}")
        return False


def find_files_by_pattern(directory: Path, pattern: str) -> List[Path]:
    """
    Find files matching a pattern.

    Args:
        directory: Directory to search
        pattern: Glob pattern to match

    Returns:
        List[Path]: List of matching file paths
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    try:
        return list(directory.glob(pattern))
    except Exception as e:
        logger.error(f"Error searching for pattern {pattern} in {directory}: {e}")
        return []


def get_file_info(file_path: Path) -> Optional[dict]:
    """
    Get detailed file information.

    Args:
        file_path: Path to file

    Returns:
        dict: File information or None if error
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return None

        stat = file_path.stat()

        return {
            "name": file_path.name,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified_time": stat.st_mtime,
            "created_time": stat.st_ctime,
            "is_file": file_path.is_file(),
            "is_directory": file_path.is_dir(),
            "extension": file_path.suffix.lower(),
        }

    except Exception as e:
        logger.error(f"Error getting file info for {file_path}: {e}")
        return None
