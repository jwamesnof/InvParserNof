import unittest
from fastapi.testclient import TestClient
from app import app
from db_util import init_db, clean_db, save_inv_extraction


class TestGetInvoicesByVendor(unittest.TestCase):
    """Integration tests for GET /invoices/vendor/{vendor_name}"""

    def setUp(self):
        init_db()
        self.client = TestClient(app)

        # Insert multiple invoices for same vendor
        save_inv_extraction({
            "confidence": 0.96,
            "data": {
                "VendorName": "Amazon",
                "InvoiceNumber": "A-001"
            },
            "dataConfidence": {"VendorName": 0.96},
            "predictionTime": 0.1
        })

        save_inv_extraction({
            "confidence": 0.97,
            "data": {
                "VendorName": "Amazon",
                "InvoiceNumber": "A-002"
            },
            "dataConfidence": {"VendorName": 0.97},
            "predictionTime": 0.1
        })

    def tearDown(self):
        clean_db()

    def test_get_invoices_for_existing_vendor(self):
        response = self.client.get("/invoices/vendor/Amazon")

        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["VendorName"], "Amazon")
        self.assertEqual(data["TotalInvoices"], 2)
        self.assertEqual(len(data["invoices"]), 2)

    def test_get_invoices_for_unknown_vendor(self):
        response = self.client.get("/invoices/vendor/UnknownVendor")

        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["VendorName"], "Unknown Vendor")
        self.assertEqual(data["TotalInvoices"], 0)
        self.assertEqual(data["invoices"], [])
