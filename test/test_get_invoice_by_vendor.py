"""Test GET /invoices/vendor/{name} endpoint"""
import os
import unittest
from unittest.mock import patch

os.environ["DB_BACKEND"] = "sqlite"

from fastapi.testclient import TestClient
from app.main import app
from app.db.database import Base, engine, init_db


class TestGetInvoiceByVendor(unittest.TestCase):
    def setUp(self):
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        init_db()
        self.client = TestClient(app)

    def tearDown(self):
        Base.metadata.drop_all(bind=engine)

    @patch("app.controllers.invoice_controller.fetch_invoices_by_vendor_name")
    def test_get_invoices_by_vendor_success(self, mock_fetch):
        """Test getting invoices by vendor name"""
        from unittest.mock import MagicMock
        
        mock_invoice = MagicMock()
        mock_invoice.InvoiceId = "INV001"
        mock_invoice.VendorName = "Test Vendor"
        mock_invoice.InvoiceDate = None
        mock_invoice.BillingAddressRecipient = None
        mock_invoice.ShippingAddress = None
        mock_invoice.SubTotal = None
        mock_invoice.ShippingCost = None
        mock_invoice.InvoiceTotal = None
        
        mock_fetch.return_value = [mock_invoice]
        
        response = self.client.get("/invoices/vendor/Test%20Vendor")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body), 1)
        self.assertEqual(body[0]["InvoiceId"], "INV001")

    @patch("app.controllers.invoice_controller.fetch_invoices_by_vendor_name")
    def test_get_invoices_by_vendor_empty(self, mock_fetch):
        """Test getting invoices for vendor with no matches"""
        mock_fetch.return_value = []
        
        response = self.client.get("/invoices/vendor/NoMatch")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body), 0)


if __name__ == "__main__":
    unittest.main()
