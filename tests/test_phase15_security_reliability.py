"""
Comprehensive Test Suite for Phase 15: Security, Reliability, and Production Hardening.
Tests:
- Secure file upload validation (extensions, double extensions, sizes, binary payloads, empty files)
- Path traversal protection (directory escapes, null bytes, Windows ADS)
- Arbitrary code execution protection (ToolRegistry whitelist, no eval/exec)
- API key masking & leak prevention (providers route, logs, error responses)
- Session isolation & memory sliding window
- Centralized error handlers & HTTP security headers
- Safe file cleanup & dataset deletion
- Production configuration validation & WSGI entrypoint
"""

import io
import os
import logging
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from config.settings import Config
from app import create_app
from utils.security import (
    safe_path_resolve,
    validate_safe_identifier,
    mask_sensitive_data,
    SensitiveDataFilter,
    sanitize_input_string,
    is_binary_payload,
)
from utils.validators import (
    allowed_file,
    sanitize_filename,
    validate_file_upload,
    validate_csv_structure,
)
from utils.file_handler import (
    save_uploaded_file,
    load_dataset,
    get_file_path,
    save_processed_dataset,
    get_latest_processed_file,
)
from utils.cleanup import (
    cleanup_old_files,
    cleanup_dataset,
    cleanup_temp_files,
)
from analysis.tool_registry import registry, ToolExecutionError
from agent.question_router import session_manager, ConversationSessionManager, QuestionRouter


@pytest.fixture
def app():
    """Create Flask application fixture configured for testing."""
    test_app = create_app()
    test_app.config["TESTING"] = True
    test_app.config["DEBUG"] = False
    return test_app


@pytest.fixture
def client(app):
    """Create Flask test client."""
    return app.test_client()


@pytest.fixture
def sample_csv_bytes():
    """Generate valid sample CSV file bytes."""
    csv_content = "ID,Name,Value,Category\n1,Alpha,100.5,Tech\n2,Beta,200.0,Finance\n3,Gamma,300.2,Health\n"
    return io.BytesIO(csv_content.encode("utf-8"))


# ==============================================================================
# 1. FILE UPLOAD SECURITY TESTS
# ==============================================================================

class TestFileUploadSecurity:
    """Validate file upload restrictions, allowed extensions, and malicious payload blocking."""

    def test_allowed_file_valid_csv(self):
        """Verify normal CSV filenames are allowed."""
        assert allowed_file("data.csv") is True
        assert allowed_file("monthly_sales_2024.CSV") is True
        assert allowed_file("customer-records.csv") is True

    def test_allowed_file_rejects_non_csv(self):
        """Verify non-CSV extensions are rejected."""
        assert allowed_file("data.txt") is False
        assert allowed_file("executable.exe") is False
        assert allowed_file("script.py") is False
        assert allowed_file("image.png") is False
        assert allowed_file("document.pdf") is False
        assert allowed_file("archive.zip") is False
        assert allowed_file("no_extension") is False

    def test_allowed_file_rejects_double_extensions(self):
        """Verify malicious multi-part / double extensions are rejected."""
        assert allowed_file("exploit.php.csv") is False
        assert allowed_file("malicious.exe.csv") is False
        assert allowed_file("script.sh.csv") is False
        assert allowed_file("backdoor.phtml.csv") is False
        assert allowed_file("payload.js.csv") is False
        assert allowed_file("attack.bat.csv") is False

    def test_allowed_file_rejects_null_bytes_and_hidden_files(self):
        """Verify null bytes and dotfiles are rejected."""
        assert allowed_file("data.csv\x00.exe") is False
        assert allowed_file("data.csv%00.txt") is False
        assert allowed_file(".htaccess") is False
        assert allowed_file(".env") is False
        assert allowed_file(".bashrc") is False

    def test_sanitize_filename(self):
        """Verify filename sanitization removes paths, null bytes, and invalid chars."""
        assert sanitize_filename("normal_file.csv") == "normal_file.csv"
        assert sanitize_filename("../../etc/passwd.csv") == "etc_passwd.csv"
        assert sanitize_filename("..\\..\\windows\\system32\\calc.csv") == "windows_system32_calc.csv"
        assert sanitize_filename("test\x00payload.csv") == "testpayload.csv"
        assert sanitize_filename("") == "dataset.csv"
        assert sanitize_filename("invalid_chars!@#$%^&*().csv").endswith(".csv")

    def test_binary_payload_detection(self):
        """Verify binary executable headers and zip signatures are identified."""
        assert is_binary_payload(b"MZ\x90\x00\x03\x00\x00\x00") is True  # Windows PE
        assert is_binary_payload(b"\x7fELF\x02\x01\x01\x00") is True  # Linux ELF
        assert is_binary_payload(b"PK\x03\x04\x14\x00\x00\x00") is True  # Zip Archive
        assert is_binary_payload(b"%PDF-1.5\x0d\x0a") is True  # PDF
        assert is_binary_payload(b"ID,Name,Age,Score\n1,Alice,25,95\n") is False  # CSV Text

    def test_validate_file_upload_rejects_binary_file(self):
        """Verify upload validation rejects binary executables disguised as CSV."""
        bin_storage = MagicMock()
        bin_storage.filename = "fake_dataset.csv"
        bin_storage.read.return_value = b"MZ\x90\x00\x03\x00\x00\x00"
        bin_storage.tell.return_value = 1024
        
        is_valid, err = validate_file_upload(bin_storage)
        assert is_valid is False
        assert "Binary or executable" in err

    def test_validate_file_upload_rejects_empty_file(self):
        """Verify 0-byte uploads are rejected."""
        empty_storage = MagicMock()
        empty_storage.filename = "empty.csv"
        empty_storage.tell.return_value = 0
        
        is_valid, err = validate_file_upload(empty_storage)
        assert is_valid is False
        assert "empty" in err.lower()

    def test_api_upload_endpoint_valid(self, client, sample_csv_bytes):
        """Verify successful upload via REST API."""
        data = {"file": (sample_csv_bytes, "valid_sample.csv")}
        response = client.post("/api/upload", data=data, content_type="multipart/form-data")
        assert response.status_code == 201
        res_json = response.get_json()
        assert res_json["success"] is True
        assert "dataset_id" in res_json
        assert "summary" in res_json
        assert res_json["summary"]["total_rows"] == 3


# ==============================================================================
# 2. PATH TRAVERSAL PROTECTION TESTS
# ==============================================================================

class TestPathTraversalProtection:
    """Verify strict directory containment and path traversal rejection."""

    def test_safe_path_resolve_valid_file(self, tmp_path):
        """Verify safe path resolution for legitimate files within directory."""
        test_file = tmp_path / "valid.csv"
        test_file.write_text("a,b\n1,2")
        resolved = safe_path_resolve(tmp_path, "valid.csv")
        assert resolved is not None
        assert resolved.exists()
        assert resolved == test_file.resolve()

    def test_safe_path_resolve_rejects_parent_traversal(self, tmp_path):
        """Verify ../ escapes are strictly blocked."""
        assert safe_path_resolve(tmp_path, "../outside.csv") is None
        assert safe_path_resolve(tmp_path, "../../etc/passwd") is None
        assert safe_path_resolve(tmp_path, "..\\..\\windows\\win.ini") is None
        assert safe_path_resolve(tmp_path, "sub/../../outside.csv") is None

    def test_safe_path_resolve_rejects_null_bytes(self, tmp_path):
        """Verify null byte injection in paths is rejected."""
        assert safe_path_resolve(tmp_path, "valid.csv\x00.exe") is None
        assert safe_path_resolve(tmp_path, "valid.csv%00.txt") is None

    def test_validate_safe_identifier(self):
        """Verify safe identifier validation."""
        assert validate_safe_identifier("dataset_20240101_abcdef") is True
        assert validate_safe_identifier("proc_12345_xyz") is True
        assert validate_safe_identifier("../../etc/passwd") is False
        assert validate_safe_identifier("dataset;rm -rf /") is False
        assert validate_safe_identifier("dataset|calc.exe") is False
        assert validate_safe_identifier("data\x00set") is False

    def test_api_summary_rejects_path_traversal(self, client):
        """Verify /api/summary/<dataset_id> rejects path traversal."""
        response = client.get("/api/summary/..%2f..%2fetc%2fpasswd")
        assert response.status_code in (400, 404)

    def test_api_cleaning_download_rejects_path_traversal(self, client):
        """Verify /api/cleaning/download/<dataset_id> rejects path traversal."""
        response = client.get("/api/cleaning/download/..%2f..%2fetc%2fpasswd")
        assert response.status_code in (400, 403, 404)

    def test_api_report_download_rejects_path_traversal(self, client):
        """Verify /api/report/download/<dataset_id> rejects path traversal."""
        response = client.get("/api/report/download/..%2f..%2fetc%2fpasswd")
        assert response.status_code in (400, 403, 404)


# ==============================================================================
# 3. ARBITRARY CODE EXECUTION PROTECTION TESTS
# ==============================================================================

class TestArbitraryCodeExecutionProtection:
    """Verify strictly controlled ToolRegistry and elimination of arbitrary code execution."""

    def test_tool_registry_rejects_unregistered_tools(self):
        """Verify unregistered tool names raise ToolExecutionError."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        with pytest.raises(ToolExecutionError) as exc_info:
            registry.execute("execute_custom_python_script", df, code="import os; os.system('echo hacked')")
        assert "Arbitrary code execution is strictly prohibited" in str(exc_info.value)

    def test_tool_registry_rejects_eval_and_exec(self):
        """Verify eval and exec builtins cannot be invoked via registry."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        with pytest.raises(ToolExecutionError):
            registry.execute("eval", df, expression="1+1")
        with pytest.raises(ToolExecutionError):
            registry.execute("exec", df, code="print(1)")
        with pytest.raises(ToolExecutionError):
            registry.execute("__import__", df, name="os")

    def test_api_chat_tools_lists_authorized_tools_only(self, client):
        """Verify /api/chat/tools declares strictly controlled execution policy."""
        response = client.get("/api/chat/tools")
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "Strictly Controlled" in data["execution_policy"]
        assert len(data["tools"]) >= 8


# ==============================================================================
# 4. API KEY PROTECTION & SENSITIVE DATA REDACTION TESTS
# ==============================================================================

class TestAPIKeyProtectionAndRedaction:
    """Verify secrets are masked in logs, error messages, and API responses."""

    def test_mask_sensitive_data_gemini_key(self):
        """Verify Gemini API keys are redacted."""
        raw = "Error calling https://generativelanguage.googleapis.com?key=AIzaSyD-fakeKey1234567890abcdefghijkl"
        masked = mask_sensitive_data(raw)
        assert "AIzaSyD" not in masked
        assert "[REDACTED" in masked

    def test_mask_sensitive_data_openai_key(self):
        """Verify OpenAI API keys are redacted."""
        raw = "Failed auth: sk-abcdefghijklmnopqrstuvwxyz1234567890"
        masked = mask_sensitive_data(raw)
        assert "sk-abcdef" not in masked
        assert "[REDACTED_OPENAI_KEY]" in masked

    def test_mask_sensitive_data_anthropic_key(self):
        """Verify Anthropic API keys are redacted."""
        raw = "Failed auth: ant-abcdefghijklmnopqrstuvwxyz1234567890"
        masked = mask_sensitive_data(raw)
        assert "ant-abcdef" not in masked
        assert "[REDACTED_ANTHROPIC_KEY]" in masked

    def test_mask_sensitive_data_bearer_tokens(self):
        """Verify Bearer tokens are redacted."""
        raw = "Authorization header: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        masked = mask_sensitive_data(raw)
        assert "Bearer [REDACTED_TOKEN]" in masked

    def test_sensitive_data_logging_filter(self):
        """Verify SensitiveDataFilter redacts log records."""
        filter_instance = SensitiveDataFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="API key AIzaSyD1234567890abcdefghijklmnopqrstuvwxyz leaked",
            args=(),
            exc_info=None,
        )
        filter_instance.filter(record)
        assert "AIzaSyD" not in record.msg
        assert "[REDACTED" in record.msg

    def test_api_insights_providers_never_leaks_keys(self, client):
        """Verify /api/insights/providers returns boolean status, never secret keys."""
        response = client.get("/api/insights/providers")
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        for provider_key, provider_info in data["providers"].items():
            assert "configured" in provider_info
            assert isinstance(provider_info["configured"], bool)
            assert "api_key" not in provider_info
            assert "key" not in provider_info
            assert "secret" not in provider_info


# ==============================================================================
# 5. SESSION ISOLATION & MEMORY MANAGEMENT TESTS
# ==============================================================================

class TestSessionIsolation:
    """Verify multi-tenant dataset session isolation and memory bounded growth."""

    def test_session_manager_isolation_between_datasets(self):
        """Verify different dataset sessions do not cross-contaminate."""
        mgr = ConversationSessionManager()
        mgr.clear_history("ds_alpha")
        mgr.clear_history("ds_beta")

        mgr.add_turn("ds_alpha", "What is total sales?", "Sales is 500.", "aggregation_analysis", {}, {})
        mgr.add_turn("ds_beta", "What is employee count?", "Count is 42.", "dataset_summary", {}, {})

        hist_alpha = mgr.get_history("ds_alpha")
        hist_beta = mgr.get_history("ds_beta")

        assert len(hist_alpha) == 1
        assert len(hist_beta) == 1
        assert hist_alpha[0]["user_query"] == "What is total sales?"
        assert hist_beta[0]["user_query"] == "What is employee count?"

    def test_session_manager_sliding_window_cap(self):
        """Verify conversation session enforces maximum turns cap."""
        mgr = ConversationSessionManager()
        ds_id = "ds_sliding_test"
        mgr.clear_history(ds_id)

        for i in range(120):
            mgr.add_turn(ds_id, f"Query {i}", f"Answer {i}", "dataset_summary", {}, {})

        history = mgr.get_history(ds_id)
        assert len(history) <= mgr.MAX_SESSION_TURNS
        assert history[-1]["user_query"] == "Query 119"

    def test_api_chat_clear_session(self, client, sample_csv_bytes):
        """Verify /api/chat/clear/<dataset_id> clears only the targeted dataset history."""
        # 1. Upload dataset
        data = {"file": (sample_csv_bytes, "session_test.csv")}
        res = client.post("/api/upload", data=data, content_type="multipart/form-data")
        ds_id = res.get_json()["dataset_id"]

        # 2. Ask question
        client.post("/api/chat/ask", json={"dataset_id": ds_id, "query": "What is the average Value?"})
        hist_res = client.get(f"/api/chat/history/{ds_id}")
        assert len(hist_res.get_json()["history"]) >= 1

        # 3. Clear session
        clear_res = client.post(f"/api/chat/clear/{ds_id}")
        assert clear_res.status_code == 200
        assert clear_res.get_json()["cleared"] is True

        # 4. Verify cleared
        hist_after = client.get(f"/api/chat/history/{ds_id}")
        assert len(hist_after.get_json()["history"]) == 0


# ==============================================================================
# 6. ERROR HANDLING & SECURITY HEADERS TESTS
# ==============================================================================

class TestErrorHandlingAndSecurityHeaders:
    """Verify standardized JSON error responses and HTTP security headers."""

    def test_404_json_error_for_api_routes(self, client):
        """Verify non-existent API routes return structured JSON with 404 code."""
        response = client.get("/api/nonexistent_route")
        assert response.status_code == 404
        data = response.get_json()
        assert data["success"] is False
        assert "Requested resource was not found" in data["error"]

    def test_405_method_not_allowed(self, client):
        """Verify wrong HTTP method returns 405 JSON response."""
        response = client.get("/api/upload")  # Upload requires POST
        assert response.status_code == 405
        data = response.get_json()
        assert data["success"] is False
        assert data["code"] == 405

    def test_security_headers_present(self, client):
        """Verify security headers are attached to responses."""
        response = client.get("/api/health")
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "SAMEORIGIN"
        assert response.headers.get("X-XSS-Protection") == "1; mode=block"
        assert response.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"


# ==============================================================================
# 7. FILE & SESSION CLEANUP TESTS
# ==============================================================================

class TestFileAndSessionCleanup:
    """Verify cleanup of old uploads, processed datasets, and temporary files."""

    def test_cleanup_temp_files(self):
        """Verify temporary files in data/temp/ are cleaned."""
        temp_dir = Config.TEMP_FOLDER
        temp_dir.mkdir(parents=True, exist_ok=True)
        test_temp_file = temp_dir / "temp_scratch_12345.tmp"
        test_temp_file.write_text("temporary data")
        
        removed = cleanup_temp_files()
        assert removed >= 1
        assert not test_temp_file.exists()

    def test_cleanup_dataset(self, sample_csv_bytes, client):
        """Verify cleanup_dataset deletes raw and processed files."""
        # 1. Upload dataset
        data = {"file": (sample_csv_bytes, "cleanup_test.csv")}
        res = client.post("/api/upload", data=data, content_type="multipart/form-data")
        ds_id = res.get_json()["dataset_id"]

        # 2. Cleanup dataset
        cleanup_res = cleanup_dataset(ds_id)
        assert cleanup_res["success"] is True
        assert ds_id == cleanup_res["dataset_id"]

        # 3. Verify dataset not found after deletion
        load_res, err = load_dataset(ds_id)
        assert load_res is None

    def test_api_delete_dataset_endpoint(self, client, sample_csv_bytes):
        """Verify DELETE /api/dataset/<dataset_id> endpoint."""
        data = {"file": (sample_csv_bytes, "delete_endpoint_test.csv")}
        res = client.post("/api/upload", data=data, content_type="multipart/form-data")
        ds_id = res.get_json()["dataset_id"]

        del_res = client.delete(f"/api/dataset/{ds_id}")
        assert del_res.status_code == 200
        assert del_res.get_json()["success"] is True

    def test_api_admin_cleanup_endpoint(self, client):
        """Verify POST /api/admin/cleanup endpoint."""
        res = client.post("/api/admin/cleanup", json={"max_age_hours": 48})
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert "cleanup_summary" in data


# ==============================================================================
# 8. PRODUCTION CONFIGURATION TESTS
# ==============================================================================

class TestProductionConfiguration:
    """Verify production safeguards and environment configuration."""

    def test_production_config_warnings(self):
        """Verify production warnings when DEBUG is True or default SECRET_KEY is used."""
        with patch.object(Config, "FLASK_ENV", "production"):
            with patch.object(Config, "DEBUG", True):
                with patch.object(Config, "SECRET_KEY", "dev-secret-key-change-in-production"):
                    warnings = Config.validate_production_config()
                    assert len(warnings) >= 2
                    assert any("DEBUG" in w for w in warnings)
                    assert any("SECRET_KEY" in w for w in warnings)

    def test_wsgi_module_loads_successfully(self):
        """Verify wsgi.py entrypoint imports and initializes Flask app."""
        import wsgi
        assert wsgi.app is not None
        assert wsgi.app.name == "app"

    def test_config_security_summary(self):
        """Verify Config.get_security_summary() reports expected keys."""
        summary = Config.get_security_summary()
        assert "environment" in summary
        assert "max_upload_size_mb" in summary
        assert "max_rows_limit" in summary
        assert "allowed_extensions" in summary

    def test_security_audit_api_endpoint(self, client):
        """Verify GET /api/security/audit returns security controls and warnings."""
        res = client.get("/api/security/audit")
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert "security" in data
        assert "production_warnings" in data
