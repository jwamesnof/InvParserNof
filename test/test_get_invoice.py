"""Test GET /invoice/{id} endpoint"""
import os
import unittest
from unittest.mock import patch, MagicMock

os.environ["DB_BACKEND"] = "sqlite"

from fastapi.testclient import TestClient
from app.main import app
from app.db.database import Base, engine, init_db


class TestGetInvoice(unittest.TestCase):
    def setUp(self):
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        init_db()
        self.client = TestClient(app)

    def tearDown(self):
        Base.metadata.drop_all(bind=engine)

    @patch("app.controllers.invoice_controller.fetch_invoice")
    def test_get_invoice_success(self, mock_fetch):
        """Test getting an existing invoice"""
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
        
        mock_fetch.return_value = {
            "invoice": mock_invoice,
            "items": [],
            "confidence": None
        }
        
        response = self.client.get("/invoice/INV001")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["InvoiceId"], "INV001")

    @patch("app.controllers.invoice_controller.fetch_invoice")
    def test_get_invoice_not_found(self, mock_fetch):
        """Test getting non-existent invoice returns 404"""
        mock_fetch.return_value = None
        
        response = self.client.get("/invoice/NOTEXIST")
        self.assertEqual(response.status_code, 404)


class TestInvoiceControllerEdgeCases(unittest.TestCase):
    """Test invoice controller edge cases for schema conversions"""
    
    def test_schema_conversions_edge_cases(self):
        """Test confidence, item, and invoice schema conversions with edge cases"""
        from app.controllers.invoice_controller import _to_confidence_schema, _to_item_schema, _to_invoice_schema
        
        # Test confidence schema with None
        result = _to_confidence_schema(None)
        self.assertIsNone(result)
        
        # Test item schema with missing attributes
        mock_item = MagicMock(spec=[])  # Empty spec = no attributes
        result = _to_item_schema(mock_item)
        self.assertIsNotNone(result)
        self.assertIsNone(result.Description)
        self.assertIsNone(result.Name)
        
        # Test invoice schema with empty items list
        mock_inv = MagicMock()
        mock_inv.InvoiceId = "INV001"
        mock_inv.VendorName = "Vendor"
        mock_inv.InvoiceDate = None
        mock_inv.BillingAddressRecipient = None
        mock_inv.ShippingAddress = None
        mock_inv.SubTotal = None
        mock_inv.ShippingCost = None
        mock_inv.InvoiceTotal = None
        
        result = _to_invoice_schema(mock_inv, [], None)
        self.assertEqual(result.InvoiceId, "INV001")
        self.assertEqual(result.Items, [])
        self.assertIsNone(result.Confidence)
        
        # Test invoice schema with multiple items
        mock_inv.InvoiceId = "INV002"
        mock_inv.InvoiceTotal = 1000.00
        mock_items = [
            MagicMock(Description="Item1", Name="Desc1", Quantity=1, UnitPrice=500, Amount=500),
            MagicMock(Description="Item2", Name="Desc2", Quantity=1, UnitPrice=500, Amount=500),
        ]
        
        result = _to_invoice_schema(mock_inv, mock_items, None)
        self.assertEqual(len(result.Items), 2)
        self.assertEqual(result.Items[0].Description, "Item1")


if __name__ == "__main__":
    unittest.main()
