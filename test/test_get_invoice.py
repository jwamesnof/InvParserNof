import unittest
from fastapi.testclient import TestClient
from app import app
from db_util import init_db, clean_db, save_inv_extraction


class TestGetInvoiceById(unittest.TestCase):
    """Integration tests for GET /invoice/{invoice_id}"""

    def setUp(self):
        init_db()
        self.client = TestClient(app)

        self.invoice_id = save_inv_extraction({
            "VendorName": "Test Vendor",
            "InvoiceNumber": "INV-001",
            "InvoiceDate": "2024-01-01",
            "TotalAmount": 100.0
        })

    def tearDown(self):
        clean_db()

    def test_get_existing_invoice(self):
        response = self.client.get(f"/invoice/{self.invoice_id}")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["VendorName"], "Test Vendor")
        self.assertEqual(data["InvoiceNumber"], "INV-001")

    def test_get_non_existing_invoice(self):
        response = self.client.get("/invoice/9999")
        self.assertEqual(response.status_code, 404)