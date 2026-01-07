import unittest
from fastapi.testclient import TestClient
from app import app
from db_util import init_db, clean_db, save_inv_extraction


class TestGetInvoiceByVendor(unittest.TestCase):

    def setUp(self):
        init_db()
        self.client = TestClient(app)

        # ✅ MUST include InvoiceId or nothing is saved
        save_inv_extraction({
            "confidence": 0.95,
            "data": {
                "InvoiceId": 1,
                "VendorName": "Amazon"
            },
            "dataConfidence": {},
            "predictionTime": 0.1
        })

        save_inv_extraction({
            "confidence": 0.96,
            "data": {
                "InvoiceId": 2,
                "VendorName": "Amazon"
            },
            "dataConfidence": {},
            "predictionTime": 0.2
        })

        save_inv_extraction({
            "confidence": 0.97,
            "data": {
                "InvoiceId": 3,
                "VendorName": "Google"
            },
            "dataConfidence": {},
            "predictionTime": 0.3
        })

    def tearDown(self):
        clean_db()

    def test_vendor_success(self):
        response = self.client.get("/invoices/vendor/Amazon")
        self.assertEqual(response.status_code, 200)

    def test_vendor_invoice_count(self):
        response = self.client.get("/invoices/vendor/Amazon")
        self.assertEqual(response.json()["TotalInvoices"], 2)

    def test_vendor_name_echoed(self):
        response = self.client.get("/invoices/vendor/Amazon")
        self.assertEqual(response.json()["VendorName"], "Amazon")

    def test_vendor_not_found(self):
        response = self.client.get("/invoices/vendor/Apple")
        self.assertEqual(response.json()["TotalInvoices"], 0)
        self.assertEqual(response.json()["VendorName"], "Unknown Vendor")

    def test_vendor_invoices_is_list(self):
        response = self.client.get("/invoices/vendor/Amazon")
        self.assertIsInstance(response.json()["invoices"], list)
