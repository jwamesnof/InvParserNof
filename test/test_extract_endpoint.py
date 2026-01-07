import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app import app
from db_util import init_db, clean_db
import io
from app import format_date, amount_format, get_value



class TestExtractEndpoint(unittest.TestCase):

    def setUp(self):
        init_db()
        self.client = TestClient(app)

    def tearDown(self):
        clean_db()

    # ---------- HELPERS ----------
    def _mock_oci_response(self, confidence=0.95, invoice_date="Mar 06 2012", amount="$1,234.50"):
        field_date = MagicMock()
        field_date.field_label.name = "InvoiceDate"
        field_date.field_label.confidence = 0.9
        field_date.field_value.text = invoice_date

        field_amount = MagicMock()
        field_amount.field_label.name = "InvoiceTotal"
        field_amount.field_label.confidence = 0.8
        field_amount.field_value.text = amount

        page = MagicMock()
        page.document_fields = [field_date, field_amount]

        response = MagicMock()
        response.data.pages = [page]

        doc_type = MagicMock()
        doc_type.confidence = confidence
        response.data.detected_document_types = [doc_type]

        return response

    # ---------- TESTS ----------

    def test_extract_valid_pdf_success(self):
        with patch("app.doc_client.analyze_document") as mock_oci:
            mock_oci.return_value = self._mock_oci_response()

            response = self.client.post(
                "/extract",
                files={"file": ("invoice.pdf", b"%PDF-1.4", "application/pdf")}
            )

            self.assertEqual(response.status_code, 200)

    def test_extract_non_pdf_rejected(self):
        response = self.client.post(
            "/extract",
            files={"file": ("test.txt", b"hello", "text/plain")}
        )

        self.assertEqual(response.status_code, 400)

    def test_extract_missing_file(self):
        response = self.client.post("/extract")
        self.assertEqual(response.status_code, 422)

    def test_extract_low_confidence_document(self):
        with patch("app.doc_client.analyze_document") as mock_oci:
            mock_oci.return_value = self._mock_oci_response(confidence=0.5)

            response = self.client.post(
                "/extract",
                files={"file": ("invoice.pdf", b"%PDF", "application/pdf")}
            )

            self.assertEqual(response.status_code, 400)

    def test_extract_oci_failure_returns_503(self):
        with patch("app.doc_client.analyze_document", side_effect=Exception("OCI down")):
            response = self.client.post(
                "/extract",
                files={"file": ("invoice.pdf", b"%PDF", "application/pdf")}
            )

            self.assertEqual(response.status_code, 503)

    def test_invoice_date_is_formatted(self):
        with patch("app.doc_client.analyze_document") as mock_oci:
            mock_oci.return_value = self._mock_oci_response()

            response = self.client.post(
                "/extract",
                files={"file": ("invoice.pdf", b"%PDF", "application/pdf")}
            )

            date = response.json()["data"]["InvoiceDate"]
            self.assertIn("2012-03-06", date)

    def test_format_date_empty(self):
        self.assertEqual(format_date(""), "")
        self.assertEqual(format_date(None), "")


    def test_format_date_invalid_string(self):
        invalid_date = "2024/01/01"
        self.assertEqual(format_date(invalid_date), invalid_date)


    def test_amount_is_float(self):
        with patch("app.doc_client.analyze_document") as mock_oci:
            mock_oci.return_value = self._mock_oci_response()

            response = self.client.post(
                "/extract",
                files={"file": ("invoice.pdf", b"%PDF", "application/pdf")}
            )

            amount = response.json()["data"]["InvoiceTotal"]
            self.assertIsInstance(amount, float)

    def test_amount_format_empty(self):
        self.assertEqual(amount_format(""), "")
        self.assertEqual(amount_format(None), "")

    
    def test_amount_format_invalid_value(self):
        invalid_amount = "abc123"
        self.assertEqual(amount_format(invalid_amount), invalid_amount)


    def test_confidence_field_exists(self):
        with patch("app.doc_client.analyze_document") as mock_oci:
            mock_oci.return_value = self._mock_oci_response()

            response = self.client.post(
                "/extract",
                files={"file": ("invoice.pdf", b"%PDF", "application/pdf")}
            )

            self.assertIn("confidence", response.json())

    def test_prediction_time_exists(self):
        with patch("app.doc_client.analyze_document") as mock_oci:
            mock_oci.return_value = self._mock_oci_response()

            response = self.client.post(
                "/extract",
                files={"file": ("invoice.pdf", b"%PDF", "application/pdf")}
            )

            self.assertIn("predictionTime", response.json())

    def test_data_confidence_exists(self):
        with patch("app.doc_client.analyze_document") as mock_oci:
            mock_oci.return_value = self._mock_oci_response()

            response = self.client.post(
                "/extract",
                files={"file": ("invoice.pdf", b"%PDF", "application/pdf")}
            )

            self.assertIn("dataConfidence", response.json())

    
    def test_get_value_none(self):
        self.assertIsNone(get_value(None))

    def test_get_value_text(self):
        obj = MagicMock()
        obj.text = "Hello"
        self.assertEqual(get_value(obj), "Hello")

    def test_get_value_value(self):
        obj = MagicMock()
        del obj.text        # critical: avoid MagicMock shadowing
        obj.value = "World"
        self.assertEqual(get_value(obj), "World")




