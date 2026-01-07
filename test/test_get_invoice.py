import unittest
from fastapi.testclient import TestClient
from app import app
from db_util import init_db, clean_db, save_inv_extraction


class TestGetInvoice(unittest.TestCase):

    def setUp(self):
        init_db()
        self.client = TestClient(app)

        save_inv_extraction({
            "data": {
                "InvoiceId": 1,
                "VendorName": "Amazon"
            }
        })

    def tearDown(self):
        clean_db()

    def test_get_invoice_success(self):
        response = self.client.get("/invoice/1")
        self.assertEqual(response.status_code, 200)

    def test_get_invoice_not_found(self):
        response = self.client.get("/invoice/999")
        self.assertEqual(response.status_code, 404)

    def test_invoice_contains_vendor_name(self):
        response = self.client.get("/invoice/1")
        self.assertEqual(response.json()["VendorName"], "Amazon")

    def test_invoice_contains_invoice_id(self):
        response = self.client.get("/invoice/1")
        self.assertIn("InvoiceId", response.json())

    def test_invoice_response_is_dict(self):
        response = self.client.get("/invoice/1")
        self.assertIsInstance(response.json(), dict)

