from sqlalchemy.orm import Session
from typing import Optional, List

from app.repositories.invoice_repository import get_invoice_by_id, get_invoices_by_vendor
from app.repositories.item_repository import get_items_by_invoice
from app.repositories.confidence_repository import get_confidence_by_invoice


def fetch_invoice(db: Session, invoice_id: str) -> Optional[dict]:
    invoice = get_invoice_by_id(db, invoice_id)
    if not invoice:
        return None

    items = get_items_by_invoice(db, invoice_id)
    confidence = get_confidence_by_invoice(db, invoice_id)

    return {
        "invoice": invoice,
        "items": items,
        "confidence": confidence
    }


def fetch_invoices_by_vendor_name(db: Session, vendor_name: str) -> List:
    return get_invoices_by_vendor(db, vendor_name)




