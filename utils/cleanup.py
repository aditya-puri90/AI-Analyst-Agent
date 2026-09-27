"""
File and Session Cleanup Utilities for AI Data Analyst Agent (Phase 15).
Provides automated and on-demand pruning of old uploads, processed datasets,
temporary analysis artifacts, and conversation session memory.
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from config.settings import Config
from utils.security import safe_path_resolve, validate_safe_identifier

logger = logging.getLogger(__name__)


def cleanup_old_files(max_age_hours: Optional[int] = None) -> Dict[str, Any]:
    """
    Scan storage directories (uploads, processed, reports, temp) and remove files
    exceeding the retention threshold.
    
    Args:
        max_age_hours: Maximum allowable file age in hours. Defaults to Config.FILE_RETENTION_HOURS.
        
    Returns:
        Dict[str, Any]: Summary of cleaned and retained files.
    """
    if max_age_hours is None:
        max_age_hours = getattr(Config, "FILE_RETENTION_HOURS", 24)

    cutoff_timestamp = time.time() - (max_age_hours * 3600)
    cleaned_count = {
        "uploads": 0,
        "processed": 0,
        "reports": 0,
        "temp": 0,
        "total_freed_bytes": 0,
        "errors": [],
    }

    folders_to_scan = [
        ("uploads", Config.UPLOAD_FOLDER),
        ("processed", Config.PROCESSED_FOLDER),
        ("reports", getattr(Config, "REPORTS_FOLDER", Config.BASE_DIR / "data/reports")),
        ("temp", getattr(Config, "TEMP_FOLDER", Config.BASE_DIR / "data/temp")),
    ]

    # Special preserved files that should never be purged
    protected_filenames = {".gitkeep", "_metadata.json", ".gitignore"}

    for category, folder in folders_to_scan:
        if not folder.exists():
            continue

        for item in folder.iterdir():
            if item.is_file() and item.name not in protected_filenames:
                try:
                    stat = item.stat()
                    if stat.st_mtime < cutoff_timestamp:
                        file_size = stat.st_size
                        # Attempt to unlink
                        item.unlink()
                        cleaned_count[category] += 1
                        cleaned_count["total_freed_bytes"] += file_size
                        logger.info("Cleaned expired %s file: %s (%d bytes)", category, item.name, file_size)
                except (PermissionError, OSError) as e:
                    logger.warning("Could not delete file %s (in use or permission error): %s", item, e)
                    cleaned_count["errors"].append(f"{item.name}: {str(e)}")

    # Clean metadata registry of removed files
    try:
        from utils.file_handler import _read_registry, _write_registry
        registry = _read_registry()
        updated_registry = {}
        for ds_id, meta in registry.items():
            filename = meta.get("filename", "")
            is_proc = meta.get("is_processed", False)
            base_dir = Config.PROCESSED_FOLDER if is_proc else Config.UPLOAD_FOLDER
            if (base_dir / filename).exists():
                updated_registry[ds_id] = meta
        if len(updated_registry) != len(registry):
            _write_registry(updated_registry)
    except Exception as e:
        logger.warning("Failed to prune metadata registry after file cleanup: %s", e)

    return cleaned_count


def cleanup_dataset(dataset_id: str) -> Dict[str, Any]:
    """
    Explicitly remove all raw files, processed transformations, and cached data
    associated with a specific dataset ID.
    
    Args:
        dataset_id: Identifier of the dataset to purge.
        
    Returns:
        Dict[str, Any]: Cleanup results.
    """
    if not validate_safe_identifier(dataset_id):
        return {"success": False, "error": f"Invalid dataset identifier: '{dataset_id}'."}

    from utils.file_handler import _read_registry, _write_registry
    registry = _read_registry()
    removed_files = []
    freed_bytes = 0

    # 1. Clean from UPLOAD_FOLDER
    for file_path in Config.UPLOAD_FOLDER.glob(f"*{dataset_id}*"):
        if file_path.is_file() and file_path.name != "_metadata.json":
            try:
                freed_bytes += file_path.stat().st_size
                file_path.unlink()
                removed_files.append(str(file_path.name))
            except Exception as e:
                logger.warning("Failed to delete upload file %s: %s", file_path, e)

    # 2. Clean from PROCESSED_FOLDER
    for file_path in Config.PROCESSED_FOLDER.glob(f"*{dataset_id}*"):
        if file_path.is_file() and file_path.name != "_metadata.json":
            try:
                freed_bytes += file_path.stat().st_size
                file_path.unlink()
                removed_files.append(str(file_path.name))
            except Exception as e:
                logger.warning("Failed to delete processed file %s: %s", file_path, e)

    # 3. Clean from REPORTS_FOLDER
    reports_dir = getattr(Config, "REPORTS_FOLDER", Config.BASE_DIR / "data/reports")
    if reports_dir.exists():
        for file_path in reports_dir.glob(f"*{dataset_id}*"):
            if file_path.is_file():
                try:
                    freed_bytes += file_path.stat().st_size
                    file_path.unlink()
                    removed_files.append(str(file_path.name))
                except Exception as e:
                    logger.warning("Failed to delete report file %s: %s", file_path, e)

    # 4. Clean Registry
    updated_registry = {k: v for k, v in registry.items() if dataset_id not in k and v.get("parent_dataset_id") != dataset_id}
    _write_registry(updated_registry)

    # 5. Clean in-memory session memory
    try:
        from agent.question_router import session_manager
        session_manager.clear_history(dataset_id)
    except Exception:
        pass

    logger.info("Purged dataset %s (%d files removed, %d bytes freed)", dataset_id, len(removed_files), freed_bytes)
    return {
        "success": True,
        "dataset_id": dataset_id,
        "removed_files": removed_files,
        "freed_bytes": freed_bytes,
    }


def cleanup_temp_files() -> int:
    """
    Remove all temporary files from data/temp/.
    
    Returns:
        int: Number of files removed.
    """
    temp_dir = getattr(Config, "TEMP_FOLDER", Config.BASE_DIR / "data/temp")
    if not temp_dir.exists():
        return 0

    removed = 0
    for item in temp_dir.iterdir():
        if item.is_file() and item.name != ".gitkeep":
            try:
                item.unlink()
                removed += 1
            except Exception as e:
                logger.warning("Error deleting temp file %s: %s", item, e)

    return removed
