"""
Validation utilities for file uploads and CSV structural integrity.
Enforces file security, size caps, encoding resilience, delimiter sniffing, and data validation.
"""

import csv
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from werkzeug.utils import secure_filename
from config.settings import Config

logger = logging.getLogger(__name__)


def allowed_file(filename: str) -> bool:
    """
    Check if the uploaded file has an allowed extension (.csv).
    
    Args:
        filename: Name of the uploaded file.
        
    Returns:
        bool: True if allowed, False otherwise.
    """
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower().strip()
    return ext in Config.ALLOWED_EXTENSIONS


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal or unsafe characters.
    
    Args:
        filename: Raw filename string.
        
    Returns:
        str: Safe, sanitized filename.
    """
    if not filename:
        return "dataset.csv"
    cleaned = secure_filename(filename)
    if not cleaned or cleaned.startswith("."):
        cleaned = "dataset.csv"
    if not cleaned.lower().endswith(".csv"):
        cleaned += ".csv"
    return cleaned


def detect_encoding_and_delimiter(file_path: Path) -> Tuple[str, str]:
    """
    Resiliently detect the text encoding and CSV delimiter of a file.
    Tests candidate encodings: utf-8, utf-8-sig, latin-1, cp1252, iso-8859-1.
    
    Args:
        file_path: Path to the target CSV file.
        
    Returns:
        Tuple[str, str]: (detected_encoding, detected_delimiter)
    """
    detected_encoding = "utf-8"
    sample_bytes = b""

    # Try candidate encodings in priority order
    for encoding in Config.FALLBACK_ENCODINGS:
        try:
            with open(file_path, "rb") as f:
                sample_bytes = f.read(65536)  # 64KB sample
            sample_bytes.decode(encoding)
            detected_encoding = encoding
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    # Delimiter detection
    detected_delimiter = ","
    try:
        sample_text = sample_bytes.decode(detected_encoding)
        if sample_text.strip():
            # Use Python's csv.Sniffer
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample_text[:8192], delimiters=[",", ";", "\t", "|"])
            detected_delimiter = dialect.delimiter
    except Exception as e:
        logger.debug("CSV Sniffer delimiter detection fallback: %s", e)
        # Fallback: Count delimiters in the first non-empty lines
        try:
            lines = [line for line in sample_text.splitlines() if line.strip()]
            if lines:
                candidate_counts = {d: lines[0].count(d) for d in [",", ";", "\t", "|"]}
                best_delimiter = max(candidate_counts, key=candidate_counts.get)
                if candidate_counts[best_delimiter] > 0:
                    detected_delimiter = best_delimiter
        except Exception:
            detected_delimiter = ","

    return detected_encoding, detected_delimiter


def validate_file_upload(file_storage) -> Tuple[bool, Optional[str]]:
    """
    Validate an incoming Flask FileStorage object before saving to disk.
    
    Args:
        file_storage: Werkzeug FileStorage instance.
        
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if file_storage is None or not getattr(file_storage, "filename", None):
        return False, "No file was selected for upload. Please choose a CSV file."

    if not allowed_file(file_storage.filename):
        allowed_str = ", ".join([f".{ext}" for ext in Config.ALLOWED_EXTENSIONS])
        return False, f"Invalid file format '{file_storage.filename}'. Only CSV files ({allowed_str}) are supported."

    # Check for empty content (0 bytes)
    file_storage.seek(0, 2)  # Seek to end of file
    file_size = file_storage.tell()
    file_storage.seek(0)  # Reset stream position to beginning

    if file_size == 0:
        return False, "The uploaded file is empty (0 bytes). Please upload a valid CSV containing data."

    if file_size > Config.MAX_CONTENT_LENGTH:
        max_mb = Config.MAX_CONTENT_LENGTH / (1024 * 1024)
        actual_mb = file_size / (1024 * 1024)
        return False, f"File size ({actual_mb:.1f} MB) exceeds maximum allowed limit of {max_mb:.0f} MB."

    return True, None


def validate_csv_structure(file_path: Path) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Verify that the CSV file can be properly decoded and contains valid tabular data.
    Checks for:
    - Empty or whitespace-only files
    - Missing or all-blank header rows
    - Absence of data rows (header-only CSV)
    - Structural corruption (mismatched quotes, unparseable lines)
    
    Args:
        file_path: Path to the saved CSV file.
        
    Returns:
        Tuple[bool, Optional[str], Optional[Dict[str, Any]]]: (is_valid, error_message, metadata)
    """
    if not file_path.exists():
        return False, f"File does not exist at path: {file_path}", None

    file_size = file_path.stat().st_size
    if file_size == 0:
        return False, "The uploaded CSV file is empty (0 bytes).", None

    try:
        encoding, delimiter = detect_encoding_and_delimiter(file_path)

        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            reader = csv.reader(f, delimiter=delimiter)
            
            # Read first row as header
            try:
                header = next(reader)
            except StopIteration:
                return False, "CSV file is completely empty.", None
            except csv.Error as e:
                return False, f"CSV parsing error in header row: {str(e)}", None

            if not header or all(not str(col).strip() for col in header):
                return False, "CSV header row is missing or contains only empty column names.", None

            # Clean header column names
            clean_headers = [str(col).strip() for col in header]

            # Count rows and check for corruption
            row_count = 0
            inconsistent_rows = 0
            expected_cols = len(clean_headers)

            for line_idx, row in enumerate(reader, start=2):
                row_count += 1
                if len(row) != expected_cols:
                    inconsistent_rows += 1
                # Sample up to first 500 rows for deep integrity validation
                if line_idx > 500:
                    # File is large and valid up to 500 rows
                    for _ in reader:
                        row_count += 1
                    break

            if row_count == 0:
                return False, "The CSV file contains a header row but has no data rows. At least 1 row of data is required.", None

        metadata = {
            "encoding": encoding,
            "delimiter": delimiter,
            "column_count": expected_cols,
            "row_count": row_count,
            "approx_row_count": row_count,
            "columns": clean_headers,
            "inconsistent_rows_count": inconsistent_rows,
        }

        return True, None, metadata

    except UnicodeError as ue:
        logger.error("Encoding error validating CSV: %s", ue)
        return False, f"Encoding error: Unable to decode file. Supported encodings are UTF-8, Latin-1, CP1252.", None
    except csv.Error as ce:
        logger.error("CSV structural syntax error: %s", ce)
        return False, f"CSV formatting error: {str(ce)}. Please check for unescaped quotes or corrupted lines.", None
    except Exception as e:
        logger.error("CSV structural validation failed: %s", e, exc_info=True)
        return False, f"Failed to parse CSV: {str(e)}", None
