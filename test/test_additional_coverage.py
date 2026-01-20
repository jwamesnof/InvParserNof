"""Additional coverage tests - consolidated from coverage optimization files"""
import os
import importlib
import unittest
from contextlib import ExitStack
import sys
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from app.main import app
from app.db import database
from app.db.database import Base, engine, SessionLocal
from app.services import value_utils
from app.services.invoice_service import fetch_invoices_by_vendor_name, fetch_invoice
from app.repositories.invoice_repository import (
    update_invoice,
    delete_invoice,
    get_invoice_by_id,
    create_invoice,
)
from app.repositories.confidence_repository import delete_confidence_for_invoice
from app.models.invoice_model import Invoice
from app.models.confidence_model import Confidence
from app.schemas.invoice_response_schema import InvoiceResponseSchema
from app.services.confidence_service import create_confidence


class TestLifespan(unittest.TestCase):
    """Test app lifespan context"""
    
    def test_lifespan_runs(self):
        with TestClient(app) as client:
            response = client.get("/health")
            self.assertIn(response.status_code, (200, 404))


class TestValueUtils(unittest.TestCase):
    """Value formatting utilities - all edge cases"""
    
    def test_amount_format_all_cases(self):
        """Test amount_format with valid, invalid, and None inputs"""
        self.assertIsNone(value_utils.amount_format("abc"))
        self.assertIsNone(value_utils.amount_format("invalid"))
        self.assertEqual(value_utils.amount_format("$1,234.50"), 1234.50)
        self.assertEqual(value_utils.amount_format("$123.45"), 123.45)
        self.assertIsNone(value_utils.amount_format(None))

    def test_format_date_all_cases(self):
        """Test format_date with ISO, text, invalid, and None inputs"""
        iso = "2020-01-01T00:00:00Z"
        self.assertEqual(value_utils.format_date(iso), iso)
        
        text_date = "Jan 15 2024"
        result = value_utils.format_date(text_date)
        self.assertIn("2024-01-15", result)
        
        bad = "not-a-date"
        self.assertEqual(value_utils.format_date(bad), bad)
        
        self.assertIsNone(value_utils.format_date(None))
    
    def test_get_value_from_object(self):
        """Test get_value with object and None"""
        obj = MagicMock()
        obj.text = "test value"
        self.assertEqual(value_utils.get_value(obj), "test value")
        self.assertIsNone(value_utils.get_value(None))


class TestInvoiceResponseSchema(unittest.TestCase):
    """Invoice response schema validation"""
    
    def test_invoice_response_creation(self):
        schema = InvoiceResponseSchema(
            invoice={"InvoiceId": "INV001"},
            items=[],
            confidence=None
        )
        self.assertIsNotNone(schema)
        self.assertIsInstance(schema, InvoiceResponseSchema)


class TestHealthAndEndpoints(unittest.TestCase):
    """Health endpoint and basic routing"""
    
    def setUp(self):
        self.client = TestClient(app)
    
    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
    
    def test_invalid_endpoint_404(self):
        response = self.client.get("/invalid/path")
        self.assertEqual(response.status_code, 404)


class TestInvoiceService(unittest.TestCase):
    """Invoice service helper functions"""
    
    @patch("app.services.invoice_service.get_invoices_by_vendor")
    def test_fetch_invoices_by_vendor_name_passthrough(self, mock_repo):
        """Test fetch_invoices_by_vendor_name passthrough"""
        mock_repo.return_value = ["sentinel"]
        result = fetch_invoices_by_vendor_name(MagicMock(), "Vendor")
        self.assertEqual(result, ["sentinel"])
    
    @patch("app.services.invoice_service.get_invoice_by_id")
    def test_fetch_invoice_missing(self, mock_get):
        """Test fetch_invoice with missing invoice"""
        mock_get.return_value = None
        result = fetch_invoice(MagicMock(), "MISSING")
        self.assertIsNone(result)
    
    @patch("app.services.invoice_service.get_invoice_by_id")
    @patch("app.services.invoice_service.get_items_by_invoice")
    @patch("app.services.invoice_service.get_confidence_by_invoice")
    def test_fetch_invoice_success(self, mock_conf, mock_items, mock_inv):
        """Test fetch_invoice with valid invoice"""
        mock_inv.return_value = MagicMock(InvoiceId="INV300", VendorName="Vendor")
        mock_items.return_value = [MagicMock()]
        mock_conf.return_value = MagicMock()
        
        result = fetch_invoice(MagicMock(), "INV300")
        self.assertIsNotNone(result)
        self.assertEqual(result["invoice"].InvoiceId, "INV300")
    
    @patch("app.services.invoice_service.get_invoices_by_vendor")
    def test_fetch_invoices_by_vendor_name_success(self, mock_get):
        """Test fetch_invoices_by_vendor_name with results"""
        mock_get.return_value = [MagicMock(InvoiceId="INV400", VendorName="Vendor")]
        
        vendors = fetch_invoices_by_vendor_name(MagicMock(), "Vendor")
        self.assertEqual(len(vendors), 1)
        mock_get.assert_called_once()


class TestInvoiceRepositoryEdge(unittest.TestCase):
    """Invoice repository edge cases"""
    
    def setUp(self):
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        self.db = SessionLocal()

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(bind=engine)

    def test_missing_record_operations(self):
        """Test operations on missing records return expected None/False"""
        result = update_invoice(self.db, "MISSING", {"VendorName": "X"})
        self.assertIsNone(result)
        
        result = delete_invoice(self.db, "MISSING")
        self.assertFalse(result)
        
        self.assertIsNone(get_invoice_by_id(self.db, "NONE"))

    def test_get_invoice_by_id_present(self):
        """Test get_invoice_by_id with existing invoice"""
        inv = create_invoice(self.db, Invoice(InvoiceId="INVX", VendorName="V"))
        found = get_invoice_by_id(self.db, "INVX")
        self.assertEqual(found.InvoiceId, inv.InvoiceId)


class TestConfidenceRepositoryEdge(unittest.TestCase):
    """Confidence repository edge cases"""
    
    def setUp(self):
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        self.db = SessionLocal()

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(bind=engine)

    def test_delete_confidence_missing_returns_false(self):
        """Test delete_confidence_for_invoice with missing invoice"""
        result = delete_confidence_for_invoice(self.db, "NONE")
        self.assertFalse(result)


class TestConfidenceService(unittest.TestCase):
    """Confidence service wrapper"""
    
    @patch("app.services.confidence_service.confidence_repository.create_confidence")
    def test_create_confidence_wrapper(self, mock_create):
        """Test create_confidence service wrapper"""
        db = MagicMock()
        mock_created = MagicMock()
        mock_create.return_value = mock_created
        
        result = create_confidence(db, "INV500", vendor_name=0.9)
        mock_create.assert_called_once()
        self.assertEqual(result, mock_created)


class TestInvoiceControllerEndpoints(unittest.TestCase):
    """Invoice controller endpoints coverage"""

    def setUp(self):
        self.client = TestClient(app)

    @patch("app.controllers.invoice_controller.fetch_invoice")
    def test_get_invoice_endpoint_success(self, mock_fetch):
        """Test GET /invoice/{invoice_id} endpoint"""
        mock_fetch.return_value = {
            "invoice": MagicMock(
                InvoiceId="INV100",
                VendorName="Vendor",
                InvoiceDate=None,
                BillingAddressRecipient=None,
                ShippingAddress=None,
                SubTotal=None,
                ShippingCost=None,
                InvoiceTotal=None,
            ),
            "items": [],
            "confidence": MagicMock(VendorName=0.9),
        }
        response = self.client.get("/invoice/INV100")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["InvoiceId"], "INV100")

    @patch("app.controllers.invoice_controller.fetch_invoices_by_vendor_name")
    def test_get_invoices_by_vendor_endpoint(self, mock_fetch):
        """Test GET /invoices/vendor/{vendor_name} endpoint"""
        mock_invoice = MagicMock()
        mock_invoice.InvoiceId = "INV200"
        mock_invoice.VendorName = "Vendor"
        mock_invoice.InvoiceDate = None
        mock_invoice.BillingAddressRecipient = None
        mock_invoice.ShippingAddress = None
        mock_invoice.SubTotal = None
        mock_invoice.ShippingCost = None
        mock_invoice.InvoiceTotal = None
        
        mock_fetch.return_value = [mock_invoice]
        response = self.client.get("/invoices/vendor/Vendor")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body), 1)
        self.assertEqual(body[0]["InvoiceId"], "INV200")


class TestDatabaseConfig(unittest.TestCase):
    """Database configuration branching"""
    
    def test_sqlite_branch_default(self):
        """Test SQLite configuration branch"""
        sys.modules.pop("app.db.database", None)
        import app.db.database as db_mod
        with patch.dict(os.environ, {"DB_BACKEND": "sqlite"}, clear=True):
            importlib.reload(db_mod)
            self.assertIn("sqlite", str(db_mod.engine.url))
        importlib.reload(db_mod)
        import pathlib
        code = pathlib.Path(db_mod.__file__).read_text()
        exec(compile(code, db_mod.__file__, "exec"), {})
        db_mod.configure_engine()

    def test_postgres_branch_uses_url(self):
        """Test PostgreSQL configuration branch"""
        import app.db.database as db_mod

        orig_env = os.environ.get("DB_BACKEND")
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ, {
                    "DB_BACKEND": "postgres",
                    "POSTGRES_USER": "user",
                    "POSTGRES_PASSWORD": "pass",
                    "POSTGRES_HOST": "host",
                    "POSTGRES_PORT": "1234",
                    "POSTGRES_DB": "dbname",
                }))
                mock_engine = stack.enter_context(patch("sqlalchemy.create_engine"))

                importlib.reload(db_mod)

                mock_engine.assert_called_once()
                args, kwargs = mock_engine.call_args
                self.assertIn("postgresql://user:pass@host:1234/dbname", args[0])
        finally:
            if orig_env is None:
                os.environ.pop("DB_BACKEND", None)
            else:
                os.environ["DB_BACKEND"] = orig_env
            importlib.reload(db_mod)


if __name__ == "__main__":
    unittest.main()
