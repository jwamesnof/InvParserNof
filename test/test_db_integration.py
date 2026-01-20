"""
Lean integration tests for repositories (11 tests total)
Covers CRUD for invoice/item/confidence and one cascade case.
"""
import os
import unittest
from unittest.mock import patch, MagicMock

os.environ["DB_BACKEND"] = "sqlite"

from app.db.database import Base, engine, SessionLocal
from app.models.invoice_model import Invoice
from app.models.item_model import Item
from app.models.confidence_model import Confidence
from app.repositories.invoice_repository import (
    create_invoice,
    get_invoice_by_id,
    update_invoice,
    delete_invoice,
    get_invoices_by_vendor,
)
from app.repositories.item_repository import (
    create_items,
    get_items_by_invoice,
    delete_items_for_invoice,
)
from app.repositories.confidence_repository import (
    create_confidence,
    get_confidence_by_invoice,
    delete_confidence_for_invoice,
)


class BaseDBTest(unittest.TestCase):
    def setUp(self):
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        self.db = SessionLocal()

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(bind=engine)


class TestInvoiceRepository(BaseDBTest):
    def test_create_invoice(self):
        invoice = Invoice(InvoiceId="INV001", VendorName="Vendor", InvoiceTotal=100.0)
        result = create_invoice(self.db, invoice)
        self.assertEqual(result.InvoiceId, "INV001")

    def test_get_invoice_by_id(self):
        invoice = Invoice(InvoiceId="INV002", VendorName="Vendor")
        create_invoice(self.db, invoice)
        result = get_invoice_by_id(self.db, "INV002")
        self.assertIsNotNone(result)

    def test_update_invoice(self):
        invoice = Invoice(InvoiceId="INV003", VendorName="Old")
        create_invoice(self.db, invoice)
        result = update_invoice(self.db, "INV003", {"VendorName": "New"})
        self.assertEqual(result.VendorName, "New")

    def test_delete_invoice(self):
        invoice = Invoice(InvoiceId="INV004", VendorName="Delete")
        create_invoice(self.db, invoice)
        deleted = delete_invoice(self.db, "INV004")
        self.assertTrue(deleted)

    def test_get_invoices_by_vendor(self):
        inv1 = Invoice(InvoiceId="INV001", VendorName="Vendor A")
        inv2 = Invoice(InvoiceId="INV002", VendorName="Vendor A")
        create_invoice(self.db, inv1)
        create_invoice(self.db, inv2)
        result = get_invoices_by_vendor(self.db, "Vendor A")
        self.assertEqual(len(result), 2)


class TestItemRepository(BaseDBTest):
    def setUp(self):
        super().setUp()
        self.invoice = Invoice(InvoiceId="INV005", VendorName="Vendor")
        create_invoice(self.db, self.invoice)

    def test_create_items(self):
        items = [Item(InvoiceId="INV005", Name="Item1", Amount=10.0)]
        create_items(self.db, items)
        self.assertEqual(len(get_items_by_invoice(self.db, "INV005")), 1)

    def test_get_items_by_invoice(self):
        create_items(self.db, [Item(InvoiceId="INV005", Name="Item1", Amount=10.0)])
        result = get_items_by_invoice(self.db, "INV005")
        self.assertEqual(len(result), 1)

    def test_delete_items_for_invoice(self):
        create_items(self.db, [Item(InvoiceId="INV005", Name="Item1", Amount=10.0)])
        deleted = delete_items_for_invoice(self.db, "INV005")
        self.assertEqual(deleted, 1)


class TestConfidenceRepository(BaseDBTest):
    def setUp(self):
        super().setUp()
        self.invoice = Invoice(InvoiceId="INV006", VendorName="Vendor")
        create_invoice(self.db, self.invoice)

    def test_create_confidence(self):
        conf = Confidence(InvoiceId="INV006", VendorName=0.95)
        result = create_confidence(self.db, conf)
        self.assertEqual(result.InvoiceId, "INV006")

    def test_get_confidence_by_invoice(self):
        conf = Confidence(InvoiceId="INV006", VendorName=0.90)
        create_confidence(self.db, conf)
        result = get_confidence_by_invoice(self.db, "INV006")
        self.assertIsNotNone(result)

    def test_delete_confidence_for_invoice(self):
        conf = Confidence(InvoiceId="INV006", VendorName=0.85)
        create_confidence(self.db, conf)
        deleted = delete_confidence_for_invoice(self.db, "INV006")
        self.assertTrue(deleted)


class TestExtractionServiceEdgeCases(unittest.TestCase):
    """Test extraction service edge cases"""
    
    @patch('app.services.extraction_service.create_items')
    @patch('app.services.extraction_service.create_confidence')
    @patch('app.services.extraction_service.create_invoice')
    def test_save_extracted_invoice_with_none_invoice_id(
        self, mock_create_invoice, mock_create_confidence, mock_create_items
    ):
        """Test save_extracted_invoice returns early when InvoiceId is None"""
        from app.services.extraction_service import save_extracted_invoice
        from sqlalchemy.orm import Session
        
        mock_db = MagicMock(spec=Session)
        
        # Call with data missing InvoiceId
        result = save_extracted_invoice(
            db=mock_db,
            data={"VendorName": "Test"},  # No InvoiceId
            confidences={},
            document_confidence=0.95
        )
        
        # Should not create anything
        mock_create_invoice.assert_not_called()
        self.assertIsNone(result)
    
    @patch('app.services.extraction_service.create_items')
    @patch('app.services.extraction_service.create_confidence')
    @patch('app.services.extraction_service.create_invoice')
    def test_save_extracted_invoice_with_no_items(
        self, mock_create_invoice, mock_create_confidence, mock_create_items
    ):
        """Test save_extracted_invoice with no items"""
        from app.services.extraction_service import save_extracted_invoice
        from sqlalchemy.orm import Session
        
        mock_db = MagicMock(spec=Session)
        
        result = save_extracted_invoice(
            db=mock_db,
            data={"InvoiceId": "INV001", "VendorName": "Test"},
            confidences={},
            document_confidence=0.95
        )
        
        # Should create invoice and confidence but not items
        mock_create_invoice.assert_called_once()
        mock_create_confidence.assert_called_once()
        mock_create_items.assert_not_called()
    
    @patch('app.services.extraction_service.create_items')
    @patch('app.services.extraction_service.create_confidence')
    @patch('app.services.extraction_service.create_invoice')
    def test_save_extracted_invoice_with_items(
        self, mock_create_invoice, mock_create_confidence, mock_create_items
    ):
        """Test save_extracted_invoice creates items when present"""
        from app.services.extraction_service import save_extracted_invoice
        from sqlalchemy.orm import Session
        
        mock_db = MagicMock(spec=Session)
        
        result = save_extracted_invoice(
            db=mock_db,
            data={
                "InvoiceId": "INV001",
                "VendorName": "Test",
                "Items": [
                    {"Description": "Item1", "Amount": "100"},
                    {"Description": "Item2", "Amount": "200"},
                ]
            },
            confidences={},
            document_confidence=0.95
        )
        
        # Should create items
        mock_create_items.assert_called_once()
        # Verify 2 items were passed
        call_args = mock_create_items.call_args
        items_arg = call_args[0][1]  # Second positional argument
        self.assertEqual(len(items_arg), 2)


if __name__ == "__main__":
    unittest.main()
