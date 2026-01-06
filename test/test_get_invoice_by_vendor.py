import unittest
from fastapi.testclient import TestClient
from app import app
from db_util import init_db, clean_db, save_inv_extraction


class TestGetInvoicesByVendor(unittest.TestCase):
    """Integration tests for GET /invoice/vendor/{vendor_name}"""

    def setUp(self):
        init_db()
        self.client = TestClient(app)

        save_inv_extraction({
            "VendorName": "Amazon",
            "InvoiceNumber": "A-001",
            "InvoiceDate": "2024-01-01",
            "TotalAmount": 50.0
        })

        save_inv_extraction({
            "VendorName": "Amazon",
            "InvoiceNumber": "A-002",
            "InvoiceDate": "2024-01-02",
            "TotalAmount": 75.0
        })

    def tearDown(self):
        clean_db()

    def test_get_invoices_for_existing_vendor(self):
        response = self.client.get("/invoice/vendor/Amazon")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["VendorName"], "Amazon")
        self.assertEqual(len(data["Invoices"]), 2)

    def test_get_invoices_for_unknown_vendor(self):
        response = self.client.get("/invoice/vendor/Unknown")
        self.assertEqual(response.status_code, 404)