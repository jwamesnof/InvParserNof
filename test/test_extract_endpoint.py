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

    @patch("oci.ai_document.AIServiceDocumentClient")
    @patch("oci.config.from_file", return_value={})
    def test_extract_non_pdf_file_returns_400(self, mock_config, mock_client):
        response = self.client.post(
            "/extract",
            files={"file": ("test.txt", b"not a pdf", "text/plain")}
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Only PDF files are supported", response.json()["detail"])