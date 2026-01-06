import unittest
import sqlite3
from fastapi.testclient import TestClient
from app import app
from db_util import init_db, clean_db, save_inv_extraction


class TestGetInvoiceById(unittest.TestCase):

    def setUp(self):
        init_db()
        self.client = TestClient(app)

        save_inv_extraction({
            "confidence": 0.95,
            "data": {
                "VendorName": "Amazon",
                "InvoiceNumber": "INV-001"
            },
            "dataConfidence": {"VendorName": 0.95},
            "predictionTime": 0.12
        })

        # ✅ Fetch the inserted invoice ID from DB
        conn = sqlite3.connect("invoices.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM invoices ORDER BY id DESC LIMIT 1")
        self.invoice_id = cursor.fetchone()[0]
        conn.close()

    def tearDown(self):
        clean_db()

    def test_get_existing_invoice(self):
        response = self.client.get(f"/invoice/{self.invoice_id}")

        self.assertEqual(response.status_code, 200)

        invoice = response.json()
        self.assertEqual(invoice["data"]["VendorName"], "Amazon")
        self.assertEqual(invoice["data"]["InvoiceNumber"], "INV-001")

    def test_get_non_existing_invoice(self):
        response = self.client.get("/invoice/999999")

        self.assertEqual(response.status_code, 404)
        self.assertIn("Invoice not found", response.json()["detail"])


