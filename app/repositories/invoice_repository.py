from sqlalchemy.orm import Session
from typing import Optional, Dict, Any, List
from app.models.invoice_model import Invoice


def create_invoice(db: Session, invoice: Invoice) -> Invoice:
    db.add(invoice)
    db.commit()
    db.refresh(invoice)
    return invoice


def get_invoice_by_id(db: Session, InvoiceId: str):
    return db.query(Invoice).filter(Invoice.InvoiceId == str(InvoiceId)).first()  # pragma: no cover

def get_invoices_by_vendor(db: Session, VendorName: str):
    return (
        db.query(Invoice)
        .filter(Invoice.VendorName == VendorName)
        .order_by(Invoice.InvoiceDate.desc())
        .all()
    )


def update_invoice(db: Session, InvoiceId: str, updates: Dict[str, Any]):
    """
    Update invoice fields by InvoiceId.
    IMPORTANT: `updates` keys must match ORM attribute names (PascalCase),
    e.g. VendorName, InvoiceDate, ShippingAddress, etc.
    """
    invoice = db.query(Invoice).filter(Invoice.InvoiceId == InvoiceId).first()
    if not invoice:
        return None

    for key, value in updates.items():
        if hasattr(invoice, key):
            setattr(invoice, key, value)

    db.add(invoice)
    db.commit()
    db.refresh(invoice)
    return invoice


def delete_invoice(db: Session, InvoiceId: str) -> bool:
    invoice = db.query(Invoice).filter(Invoice.InvoiceId == InvoiceId).first()
    if not invoice:
        return False

    db.delete(invoice)
    db.commit()
    return True
