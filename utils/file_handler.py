"""
File management utilities for safe storage, loading, and directory operations.
Maintains strict separation between raw uploads (immutable) and processed datasets.
Provides dataset summary extraction for ingested files.
"""

import os
import uuid
import json
import math
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

import pandas as pd
import numpy as np
from werkzeug.datastructures import FileStorage

from config.settings import Config
from utils.validators import (
    sanitize_filename,
    validate_file_upload,
    detect_encoding_and_delimiter,
    validate_csv_structure,
)
from analysis.profiler import infer_column_type

logger = logging.getLogger(__name__)

# Metadata registry file path
METADATA_FILE = Config.UPLOAD_FOLDER / "_metadata.json"


def _format_bytes(num_bytes: int) -> str:
    """Format bytes into human-readable string (B, KB, MB, GB)."""
    if num_bytes <= 0:
        return "0 Bytes"
    units = ["Bytes", "KB", "MB", "GB", "TB"]
    i = min(int(math.floor(math.log(num_bytes, 1024))), len(units) - 1)
    val = round(num_bytes / (1024 ** i), 2)
    return f"{val} {units[i]}"


def _safe_json_value(val: Any) -> Any:
    """Safely convert numpy/pandas scalars into JSON-serializable primitives."""
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        if math.isnan(val) or math.isinf(val):
            return None
        return round(float(val), 4)
    if isinstance(val, (pd.Timestamp, np.datetime64)):
        return str(val)
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    return str(val)


def _read_registry() -> Dict[str, Any]:
    """Read the dataset registry json."""
    if not METADATA_FILE.exists():
        return {}
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to read metadata registry: %s", e)
        return {}


def _write_registry(registry: Dict[str, Any]) -> None:
    """Write to the dataset registry json."""
    try:
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
    except Exception as e:
        logger.error("Failed to write metadata registry: %s", e)


def get_file_path(file_identifier: str, is_processed: bool = False) -> Optional[Path]:
    """
    Safely resolve a file path within the allowed directories, preventing path traversal.
    
    Args:
        file_identifier: File ID or filename.
        is_processed: Whether to search in processed or uploads folder.
        
    Returns:
        Optional[Path]: Resolved Path if exists and valid, None otherwise.
    """
    base_folder = Config.PROCESSED_FOLDER if is_processed else Config.UPLOAD_FOLDER
    sanitized = sanitize_filename(file_identifier)
    target_path = (base_folder / sanitized).resolve()

    # Ensure resolved path is strictly within the allowed directory
    try:
        target_path.relative_to(base_folder.resolve())
    except ValueError:
        logger.warning("Path traversal attempt detected: %s", file_identifier)
        return None

    return target_path if target_path.exists() else None


def save_uploaded_file(file_storage: FileStorage) -> Tuple[Optional[str], Optional[Path], Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate, sanitize, and save an uploaded CSV file to data/uploads/.
    Guarantees unique session identifiers and raw file immutability (never overwrites existing files).
    
    Args:
        file_storage: Incoming Werkzeug FileStorage object.
        
    Returns:
        Tuple[Optional[str], Optional[Path], Optional[str], Optional[Dict[str, Any]]]:
        (dataset_id, saved_path, error_message, structural_metadata)
    """
    # 1. Pre-save validation (file presence, extension, 0-byte check, size limit)
    is_valid, error = validate_file_upload(file_storage)
    if not is_valid:
        return None, None, error, None

    original_filename = file_storage.filename
    clean_name = sanitize_filename(original_filename)

    # 2. Generate unique dataset ID and safe unique on-disk filename (never collides)
    dataset_id = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    disk_filename = f"{dataset_id}_{clean_name}"
    target_path = Config.UPLOAD_FOLDER / disk_filename

    try:
        # 3. Save raw file to disk
        file_storage.save(str(target_path))

        # 4. Post-save structural validation
        is_struct_valid, struct_error, metadata = validate_csv_structure(target_path)
        if not is_struct_valid:
            if target_path.exists():
                target_path.unlink()  # Remove corrupted or empty upload
            return None, None, struct_error, None

        # 5. Record metadata in registry
        file_size_bytes = target_path.stat().st_size
        registry = _read_registry()
        rows_est = metadata.get("row_count", 0)
        cols_cnt = metadata.get("column_count", 0)
        registry[dataset_id] = {
            "id": dataset_id,
            "dataset_id": dataset_id,
            "filename": disk_filename,
            "original_name": original_filename,
            "original_filename": original_filename,
            "uploaded_at": datetime.utcnow().isoformat(),
            "size_bytes": file_size_bytes,
            "size_formatted": _format_bytes(file_size_bytes),
            "encoding": metadata.get("encoding", "utf-8"),
            "delimiter": metadata.get("delimiter", ","),
            "columns_count": cols_cnt,
            "columns": cols_cnt,
            "total_columns": cols_cnt,
            "rows_estimate": rows_est,
            "rows": rows_est,
            "total_rows": rows_est,
        }
        _write_registry(registry)

        logger.info("Successfully saved dataset %s (%s, %s bytes)", dataset_id, original_filename, file_size_bytes)
        return dataset_id, target_path, None, metadata

    except Exception as e:
        logger.error("Failed to save uploaded file: %s", e, exc_info=True)
        if target_path.exists():
            target_path.unlink()
        return None, None, f"Failed to save file: {str(e)}", None


def load_dataset(file_identifier: str, is_processed: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Safely load a dataset into a Pandas DataFrame using detected encoding and delimiter.
    
    Args:
        file_identifier: Dataset ID or filename.
        is_processed: True to load from data/processed/, False for data/uploads/.
        
    Returns:
        Tuple[Optional[pd.DataFrame], Optional[str]]: (DataFrame, error_message)
    """
    registry = _read_registry()
    filename = None
    encoding = None
    delimiter = None

    if file_identifier in registry:
        meta = registry[file_identifier]
        filename = meta["filename"]
        encoding = meta.get("encoding")
        delimiter = meta.get("delimiter")
    else:
        filename = file_identifier

    target_path = get_file_path(filename, is_processed=is_processed)
    if not target_path or not target_path.exists():
        base_folder = Config.PROCESSED_FOLDER if is_processed else Config.UPLOAD_FOLDER
        matches = list(base_folder.glob(f"{file_identifier}*"))
        if matches:
            target_path = matches[0]
        else:
            return None, f"Dataset '{file_identifier}' was not found."

    try:
        if not encoding or not delimiter:
            encoding, delimiter = detect_encoding_and_delimiter(target_path)

        df = pd.read_csv(
            target_path,
            delimiter=delimiter,
            encoding=encoding,
            low_memory=False,
            on_bad_lines="skip",
        )
        return df, None

    except Exception as e:
        logger.error("Error loading dataset %s into Pandas: %s", target_path, e, exc_info=True)
        return None, f"Failed to parse CSV with Pandas: {str(e)}"


def get_dataset_summary(file_identifier: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Extract comprehensive ingestion summary of the uploaded dataset:
    - File name & unique ID
    - Number of rows & columns
    - Memory usage (formatted in KB/MB and raw bytes)
    - Column names, Pandas data types, and inferred semantic types
    - Preview of the first 10 rows.
    
    Args:
        file_identifier: Dataset ID or filename.
        
    Returns:
        Tuple[Optional[Dict[str, Any]], Optional[str]]: (Summary dictionary, error message)
    """
    df, error = load_dataset(file_identifier)
    if error:
        return None, error

    registry = _read_registry()
    meta = registry.get(file_identifier, {})
    original_name = meta.get("original_name", file_identifier)

    total_rows, total_cols = df.shape
    memory_bytes = int(df.memory_usage(deep=True).sum())
    memory_formatted = _format_bytes(memory_bytes)

    # Column-level metadata
    null_counts = df.isnull().sum()
    columns_info = []

    for col in df.columns:
        series = df[col]
        pandas_dtype = str(series.dtype)
        inferred = infer_column_type(series)
        null_count = int(null_counts[col])
        non_null_count = int(total_rows - null_count)
        null_pct = round((null_count / total_rows * 100), 2) if total_rows > 0 else 0.0

        columns_info.append({
            "name": str(col),
            "pandas_dtype": pandas_dtype,
            "inferred_type": inferred,
            "non_null_count": non_null_count,
            "null_count": null_count,
            "null_percentage": null_pct,
        })

    # Preview of first 10 rows
    head_df = df.head(10).replace({np.nan: None})
    preview_records = []
    for idx, row in head_df.iterrows():
        row_dict = {"#": int(idx) + 1}
        for col in df.columns:
            row_dict[str(col)] = _safe_json_value(row[col])
        preview_records.append(row_dict)

    summary = {
        "dataset_id": file_identifier,
        "file_name": original_name,
        "total_rows": total_rows,
        "total_columns": total_cols,
        "memory_usage_bytes": memory_bytes,
        "memory_usage_formatted": memory_formatted,
        "encoding": meta.get("encoding", "utf-8"),
        "delimiter": meta.get("delimiter", ","),
        "columns": columns_info,
        "column_names": [str(c) for c in df.columns],
        "preview_first_10_rows": preview_records,
    }

    return summary, None


def list_uploaded_datasets() -> List[Dict[str, Any]]:
    """
    List all uploaded datasets in the system sorted by upload timestamp descending.
    
    Returns:
        List[Dict[str, Any]]: List of normalized metadata objects.
    """
    registry = _read_registry()
    datasets = []

    for dataset_id, raw_meta in sorted(
        registry.items(),
        key=lambda item: item[1].get("uploaded_at", ""),
        reverse=True,
    ):
        if not raw_meta.get("is_processed", False):
            filename = raw_meta.get("filename", "")
            file_path = Config.UPLOAD_FOLDER / filename
            if file_path.exists():
                ds_id = raw_meta.get("id") or raw_meta.get("dataset_id") or dataset_id
                orig_name = raw_meta.get("original_name") or raw_meta.get("original_filename") or filename
                rows = raw_meta.get("rows_estimate") or raw_meta.get("total_rows") or raw_meta.get("rows") or 0
                cols = raw_meta.get("columns_count") or raw_meta.get("total_columns") or raw_meta.get("columns") or 0
                
                meta = dict(raw_meta)
                meta.update({
                    "id": ds_id,
                    "dataset_id": ds_id,
                    "filename": filename,
                    "original_name": orig_name,
                    "original_filename": orig_name,
                    "rows_estimate": rows,
                    "rows": rows,
                    "total_rows": rows,
                    "columns_count": cols,
                    "columns": cols,
                    "total_columns": cols,
                })
                datasets.append(meta)

    return datasets


def get_dataset_metadata(file_identifier: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve normalized metadata dict for a dataset from registry.
    
    Args:
        file_identifier: Dataset ID or filename.
        
    Returns:
        Optional[Dict[str, Any]]: Metadata object or None.
    """
    registry = _read_registry()
    meta = registry.get(file_identifier)
    if not meta:
        for k, v in registry.items():
            if v.get("id") == file_identifier or v.get("filename") == file_identifier:
                meta = v
                break
    if not meta:
        return None

    ds_id = meta.get("id") or meta.get("dataset_id") or file_identifier
    orig_name = meta.get("original_name") or meta.get("original_filename") or meta.get("filename", ds_id)
    rows = meta.get("rows_estimate") or meta.get("total_rows") or meta.get("rows") or 0
    cols = meta.get("columns_count") or meta.get("total_columns") or meta.get("columns") or 0

    normalized = dict(meta)
    normalized.update({
        "id": ds_id,
        "dataset_id": ds_id,
        "original_name": orig_name,
        "original_filename": orig_name,
        "rows_estimate": rows,
        "rows": rows,
        "total_rows": rows,
        "columns_count": cols,
        "columns": cols,
        "total_columns": cols,
    })
    return normalized


def save_processed_dataset(
    df: pd.DataFrame,
    original_dataset_id: str,
    cleaning_summary: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[Path], Optional[str]]:
    """
    Save a cleaned/transformed DataFrame into data/processed/.
    Guarantees raw dataset immutability in data/uploads/ and records processed lineage.
    
    Args:
        df: The cleaned pandas DataFrame.
        original_dataset_id: Unique ID of the parent raw dataset.
        cleaning_summary: Summary metrics of operations applied.
        
    Returns:
        Tuple[Optional[str], Optional[Path], Optional[str]]:
        (processed_dataset_id, saved_path, error_message)
    """
    registry = _read_registry()
    parent_meta = registry.get(original_dataset_id, {})
    parent_name = parent_meta.get("original_name", "dataset.csv")
    clean_base = sanitize_filename(parent_name).replace(".csv", "")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    processed_id = f"proc_{timestamp}_{uuid.uuid4().hex[:6]}"
    disk_filename = f"cleaned_{original_dataset_id}_{clean_base}.csv"
    target_path = Config.PROCESSED_FOLDER / disk_filename

    try:
        # Save cleaned dataset non-destructively to data/processed/
        df.to_csv(target_path, index=False, encoding="utf-8")
        file_size = target_path.stat().st_size

        processed_meta = {
            "id": processed_id,
            "filename": disk_filename,
            "original_name": f"cleaned_{parent_name}",
            "parent_dataset_id": original_dataset_id,
            "is_processed": True,
            "processed_at": datetime.utcnow().isoformat(),
            "size_bytes": file_size,
            "size_formatted": _format_bytes(file_size),
            "encoding": "utf-8",
            "delimiter": ",",
            "columns_count": len(df.columns),
            "rows_count": len(df),
            "cleaning_summary": cleaning_summary or {},
        }

        registry[processed_id] = processed_meta
        # Link latest processed ID to parent dataset record
        if original_dataset_id in registry:
            registry[original_dataset_id]["latest_processed_id"] = processed_id
            registry[original_dataset_id]["latest_cleaned_filename"] = disk_filename

        _write_registry(registry)
        logger.info("Saved processed dataset %s for parent %s (%d rows)", processed_id, original_dataset_id, len(df))
        return processed_id, target_path, None

    except Exception as e:
        logger.error("Failed to save processed dataset for %s: %s", original_dataset_id, e, exc_info=True)
        return None, None, f"Failed to save processed dataset: {str(e)}"


def get_latest_processed_file(dataset_id: str) -> Optional[Path]:
    """
    Retrieve the Path to the latest processed CSV file for a given dataset ID.
    
    Args:
        dataset_id: Raw dataset ID or processed ID.
        
    Returns:
        Optional[Path]: Path to processed CSV if found, None otherwise.
    """
    registry = _read_registry()
    if dataset_id in registry:
        meta = registry[dataset_id]
        if meta.get("is_processed", False):
            return get_file_path(meta["filename"], is_processed=True)
        # Check if parent has a latest processed record
        latest_proc_id = meta.get("latest_processed_id")
        if latest_proc_id and latest_proc_id in registry:
            return get_file_path(registry[latest_proc_id]["filename"], is_processed=True)
        latest_filename = meta.get("latest_cleaned_filename")
        if latest_filename:
            return get_file_path(latest_filename, is_processed=True)

    # Fallback search by pattern in processed folder
    matches = list(Config.PROCESSED_FOLDER.glob(f"*{dataset_id}*.csv"))
    if matches:
        return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    return None

