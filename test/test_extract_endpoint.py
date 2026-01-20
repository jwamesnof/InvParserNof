import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import base64
import asyncio

# AsyncMock is available in Python 3.8+; on 3.7 fallback to MagicMock
if sys.version_info >= (3, 8):
    from unittest.mock import AsyncMock
else:
    from unittest.mock import MagicMock as AsyncMock

# Ensure SQLite backend for tests (set BEFORE importing app)
os.environ["DB_BACKEND"] = "sqlite"

from fastapi.testclient import TestClient
from app.main import app
from app.db.database import Base, engine, init_db


class TestExtractEndpoint(unittest.TestCase):

    def setUp(self):
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        # If init_db() does anything beyond create_all, keep this:
        init_db()
        self.client = TestClient(app)

    def tearDown(self):
        Base.metadata.drop_all(bind=engine)

    def test_extract_valid_pdf_success(self):
        fake_result = {
            "confidence": 1,
            "data": {"InvoiceId": "36259"},
            "dataConfidence": {}
        }

        with patch("app.controllers.extract_controller.analyze_document", return_value=fake_result):
            response = self.client.post(
                "/extract",
                files={"file": ("invoice.pdf", b"%PDF-1.4 fake", "application/pdf")},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("confidence", body)
        self.assertIn("data", body)
        self.assertIn("dataConfidence", body)

    def test_extract_low_confidence_rejected(self):
        fake_result = {
            "confidence": 0.7,
            "data": {"InvoiceId": "36259"},
            "dataConfidence": {}
        }
        with patch("app.controllers.extract_controller.analyze_document", return_value=fake_result):
            response = self.client.post(
                "/extract",
                files={"file": ("invoice.pdf", b"%PDF-1.4 fake", "application/pdf")},
            )
        self.assertEqual(response.status_code, 400)

    def test_extract_non_pdf_rejected(self):
        response = self.client.post(
            "/extract",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("PDF", response.json()["detail"])

    def test_extract_oci_failure_returns_503(self):
        with patch("app.controllers.extract_controller.analyze_document", side_effect=Exception("OCI down")):
            response = self.client.post(
                "/extract",
                files={"file": ("invoice.pdf", b"%PDF-1.4 fake", "application/pdf")},
            )

        self.assertEqual(response.status_code, 503)


class TestOCIServiceIntegration(unittest.TestCase):
    """Test OCI service analyze_document function"""

    @patch('app.services.oci_service.oci.ai_document.AIServiceDocumentClient')
    @patch('app.services.oci_service.oci.config.from_file')
    def test_analyze_document_with_multiple_fields(self, mock_config, mock_client_class):
        """Test analyze_document extracts multiple fields correctly"""
        # Setup mock config
        mock_config.return_value = {"tenancy": "ocid1.tenancy.test"}
        
        # Setup mock client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Create mock fields
        mock_field1 = MagicMock()
        mock_field1.field_name = "InvoiceId"
        mock_field1.field_value = "INV-2024-001"
        mock_field1.confidence = 0.99
        
        mock_field2 = MagicMock()
        mock_field2.field_name = "VendorName"
        mock_field2.field_value = "ACME Corp"
        mock_field2.confidence = 0.95
        
        mock_field3 = MagicMock()
        mock_field3.field_name = "InvoiceTotal"
        mock_field3.field_value = "1500.50"
        mock_field3.confidence = 0.97
        
        # Setup mock response
        mock_response = MagicMock()
        mock_response.data.document.fields = [mock_field1, mock_field2, mock_field3]
        mock_client.analyze_document.return_value = mock_response
        
        # Execute
        from app.services.oci_service import analyze_document
        
        mock_file = MagicMock()
        test_content = b"PDF_TEST_CONTENT"
        
        # Create a proper async mock for file.read()
        async def mock_read():
            return test_content
        
        mock_file.read = mock_read
        
        result = asyncio.run(analyze_document(mock_file))
        
        # Verify structure
        self.assertIn("data", result)
        self.assertIn("dataConfidence", result)
        self.assertIn("confidence", result)
        
        # Verify extracted data
        self.assertEqual(result["data"]["InvoiceId"], "INV-2024-001")
        self.assertEqual(result["data"]["VendorName"], "ACME Corp")
        self.assertEqual(result["data"]["InvoiceTotal"], "1500.50")
        
        # Verify confidences
        self.assertEqual(result["dataConfidence"]["InvoiceId"], 0.99)
        self.assertEqual(result["dataConfidence"]["VendorName"], 0.95)
        self.assertEqual(result["dataConfidence"]["InvoiceTotal"], 0.97)
        
        # Verify average confidence
        expected_confidence = (0.99 + 0.95 + 0.97) / 3
        self.assertAlmostEqual(result["confidence"], expected_confidence, places=4)
        
        # Verify base64 encoding was used
        mock_client.analyze_document.assert_called_once()
        call_args = mock_client.analyze_document.call_args[0][0]
        expected_encoded = base64.b64encode(test_content).decode("utf-8")
        self.assertEqual(call_args.document.data, expected_encoded)

    @patch('app.services.oci_service.oci.ai_document.AIServiceDocumentClient')
    @patch('app.services.oci_service.oci.config.from_file')
    def test_analyze_document_with_empty_fields(self, mock_config, mock_client_class):
        """Test analyze_document handles empty field list"""
        # Setup mock config
        mock_config.return_value = {"tenancy": "ocid1.tenancy.test"}
        
        # Setup mock client with no fields
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.data.document.fields = []
        mock_client.analyze_document.return_value = mock_response
        
        # Execute
        from app.services.oci_service import analyze_document
        
        mock_file = MagicMock()
        
        async def mock_read():
            return b"TEST"
        
        mock_file.read = mock_read
        
        result = asyncio.run(analyze_document(mock_file))
        
        # Verify empty result with 0 confidence
        self.assertEqual(result["data"], {})
        self.assertEqual(result["dataConfidence"], {})
        self.assertEqual(result["confidence"], 0)

    @patch('app.services.oci_service.oci.ai_document.AIServiceDocumentClient')
    @patch('app.services.oci_service.oci.config.from_file')
    def test_analyze_document_uses_correct_features(self, mock_config, mock_client_class):
        """Test analyze_document requests both KEY_VALUE_EXTRACTION and TABLE_EXTRACTION"""
        # Setup mocks
        mock_config.return_value = {"tenancy": "ocid1.tenancy.test"}
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.data.document.fields = []
        mock_client.analyze_document.return_value = mock_response
        
        # Execute
        from app.services.oci_service import analyze_document
        
        mock_file = MagicMock()
        
        async def mock_read():
            return b"PDF"
        
        mock_file.read = mock_read
        
        asyncio.run(analyze_document(mock_file))
        
        # Verify analyze_document was called
        mock_client.analyze_document.assert_called_once()
        
        # Get the request details
        call_args = mock_client.analyze_document.call_args[0][0]
        
        # Verify compartment_id uses tenancy
        self.assertEqual(call_args.compartment_id, "ocid1.tenancy.test")
        
        # Verify features list has 2 items
        self.assertEqual(len(call_args.features), 2)
        
        # Verify feature types
        feature_types = [f.feature_type for f in call_args.features]
        self.assertIn("KEY_VALUE_EXTRACTION", feature_types)
        self.assertIn("TABLE_EXTRACTION", feature_types)

    @patch('app.services.oci_service.oci.ai_document.AIServiceDocumentClient')
    @patch('app.services.oci_service.oci.config.from_file')
    def test_analyze_document_confidence_calculation(self, mock_config, mock_client_class):
        """Test confidence calculation with various confidence values"""
        # Setup mocks
        mock_config.return_value = {"tenancy": "ocid1.tenancy.test"}
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Create fields with specific confidence values
        fields = []
        test_confidences = [0.95, 0.85, 0.90, 0.88, 0.92]
        
        for i, conf in enumerate(test_confidences):
            mock_field = MagicMock()
            mock_field.field_name = f"Field{i}"
            mock_field.field_value = f"Value{i}"
            mock_field.confidence = conf
            fields.append(mock_field)
        
        mock_response = MagicMock()
        mock_response.data.document.fields = fields
        mock_client.analyze_document.return_value = mock_response
        
        # Execute
        from app.services.oci_service import analyze_document
        
        mock_file = MagicMock()
        
        async def mock_read():
            return b"TEST"
        
        mock_file.read = mock_read
        
        result = asyncio.run(analyze_document(mock_file))
        
        # Verify confidence is average of all field confidences
        expected_avg = sum(test_confidences) / len(test_confidences)
        self.assertAlmostEqual(result["confidence"], expected_avg, places=4)
        
        # Verify all individual confidences
        for i, conf in enumerate(test_confidences):
            self.assertEqual(result["dataConfidence"][f"Field{i}"], conf)

    @patch('app.services.oci_service.oci.ai_document.AIServiceDocumentClient')
    @patch('app.services.oci_service.oci.config.from_file')
    def test_analyze_document_base64_encoding(self, mock_config, mock_client_class):
        """Test that file content is properly base64 encoded"""
        # Setup mocks
        mock_config.return_value = {"tenancy": "ocid1.tenancy.test"}
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.data.document.fields = []
        mock_client.analyze_document.return_value = mock_response
        
        # Execute with specific content
        from app.services.oci_service import analyze_document
        
        test_content = b"Special PDF Content: \x00\x01\x02\xff"
        expected_encoded = base64.b64encode(test_content).decode("utf-8")
        
        mock_file = MagicMock()
        
        async def mock_read():
            return test_content
        
        mock_file.read = mock_read
        
        asyncio.run(analyze_document(mock_file))
        
        # Verify the InlineDocumentContent was created with correctly encoded data
        call_args = mock_client.analyze_document.call_args[0][0]
        self.assertEqual(call_args.document.data, expected_encoded)


if __name__ == "__main__":
    unittest.main()






