import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app import app
from db_util import init_db, clean_db


class TestExtractEndpoint(unittest.TestCase):
    """Integration tests for POST /extract"""

    def setUp(self):
        init_db()
        self.client = TestClient(app)

    def tearDown(self):
        clean_db()

    # ---------- 1. INVALID FILE TYPE ----------
    def test_extract_non_pdf_returns_400(self):
        response = self.client.post(
            "/extract",
            files={"file": ("test.txt", b"not a pdf", "text/plain")}
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid document", response.json()["detail"])

    # ---------- 2. OCI SERVICE FAILURE ----------
    @patch("app.doc_client")
    def test_extract_oci_failure_returns_503(self, mock_doc_client):
        mock_doc_client.analyze_document.side_effect = Exception("OCI down")

        response = self.client.post(
            "/extract",
            files={"file": ("invoice.pdf", b"%PDF-1.4 fake", "application/pdf")}
        )

        self.assertEqual(response.status_code, 503)
        self.assertIn("service is currently unavailable", response.json()["detail"])

    # ---------- 3. SUCCESS PATH (MOCKED OCI RESPONSE) ----------
    @patch("app.doc_client")
    def test_extract_success(self, mock_doc_client):

        # ---- Mock OCI response structure ----
        mock_field_value = type("obj", (), {"text": "Amazon"})()
        mock_field_label = type("obj", (), {"name": "VendorName", "confidence": 0.95})()

        mock_field = type(
            "obj",
            (),
            {"field_label": mock_field_label, "field_value": mock_field_value}
        )()

        mock_page = type("obj", (), {"document_fields": [mock_field]})()
        mock_doc_type = type("obj", (), {"confidence": 0.95})()

        mock_response = MagicMock()
        mock_response.data.pages = [mock_page]
        mock_response.data.detected_document_types = [mock_doc_type]

        mock_doc_client.analyze_document.return_value = mock_response

        # ---- Call API ----
        response = self.client.post(
            "/extract",
            files={"file": ("invoice.pdf", b"%PDF-1.4 fake content", "application/pdf")}
        )

        # ---- Assertions ----
        self.assertEqual(response.status_code, 200)

        result = response.json()
        self.assertIn("data", result)
        self.assertIn("VendorName", result["data"])
        self.assertEqual(result["data"]["VendorName"], "Amazon")
