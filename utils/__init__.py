"""
Utilities package for AI Data Analyst Agent (Phase 15).
Provides file validation, safe storage, path resolution, security filters, and cleanup routines.
"""

from .validators import (
    validate_file_upload,
    allowed_file,
    sanitize_filename,
    detect_encoding_and_delimiter,
    validate_csv_structure,
)
from .file_handler import (
    save_uploaded_file,
    load_dataset,
    get_dataset_summary,
    list_uploaded_datasets,
    get_file_path,
    save_processed_dataset,
    get_latest_processed_file,
    get_dataset_metadata,
)
from .security import (
    safe_path_resolve,
    validate_safe_identifier,
    mask_sensitive_data,
    SensitiveDataFilter,
    sanitize_input_string,
    is_binary_payload,
)
from .cleanup import (
    cleanup_old_files,
    cleanup_dataset,
    cleanup_temp_files,
)

__all__ = [
    "validate_file_upload",
    "allowed_file",
    "sanitize_filename",
    "detect_encoding_and_delimiter",
    "validate_csv_structure",
    "save_uploaded_file",
    "load_dataset",
    "get_dataset_summary",
    "list_uploaded_datasets",
    "get_file_path",
    "save_processed_dataset",
    "get_latest_processed_file",
    "get_dataset_metadata",
    "safe_path_resolve",
    "validate_safe_identifier",
    "mask_sensitive_data",
    "SensitiveDataFilter",
    "sanitize_input_string",
    "is_binary_payload",
    "cleanup_old_files",
    "cleanup_dataset",
    "cleanup_temp_files",
]
