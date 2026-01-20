from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional

from app.db.database import get_db
from app.services.invoice_service import fetch_invoice, fetch_invoices_by_vendor_name

from app.schemas.invoice_schema import InvoiceSchema
from app.schemas.item_schema import ItemSchema
from app.schemas.confidence_schema import ConfidenceSchema

router = APIRouter(tags=["Invoices"])


def _to_item_schema(it) -> ItemSchema:
    return ItemSchema(
        Description=getattr(it, "Description", None),
        Name=getattr(it, "Name", None),
        Quantity=getattr(it, "Quantity", None),
        UnitPrice=getattr(it, "UnitPrice", None),
        Amount=getattr(it, "Amount", None),
    )


def _to_confidence_schema(conf) -> Optional[ConfidenceSchema]:
    if conf is None:
        return None
    return ConfidenceSchema(
        VendorName=getattr(conf, "VendorName", None),
        InvoiceDate=getattr(conf, "InvoiceDate", None),
        BillingAddressRecipient=getattr(conf, "BillingAddressRecipient", None),
        ShippingAddress=getattr(conf, "ShippingAddress", None),
        SubTotal=getattr(conf, "SubTotal", None),
        ShippingCost=getattr(conf, "ShippingCost", None),
        InvoiceTotal=getattr(conf, "InvoiceTotal", None),
    )


def _to_invoice_schema(inv, items, conf) -> InvoiceSchema:
    return InvoiceSchema(
        InvoiceId=getattr(inv, "InvoiceId", None),
        VendorName=getattr(inv, "VendorName", None),
        InvoiceDate=getattr(inv, "InvoiceDate", None),
        BillingAddressRecipient=getattr(inv, "BillingAddressRecipient", None),
        ShippingAddress=getattr(inv, "ShippingAddress", None),
        SubTotal=getattr(inv, "SubTotal", None),
        ShippingCost=getattr(inv, "ShippingCost", None),
        InvoiceTotal=getattr(inv, "InvoiceTotal", None),
        Items=[_to_item_schema(x) for x in (items or [])],
        Confidence=_to_confidence_schema(conf),
    )


@router.get("/invoice/{invoice_id}", response_model=InvoiceSchema)
def get_invoice(invoice_id: str, db: Session = Depends(get_db)):
    result = fetch_invoice(db, invoice_id)
    if not result:
        raise HTTPException(status_code=404, detail="Invoice not found")

    return _to_invoice_schema(
        result["invoice"],
        result.get("items", []),
        result.get("confidence"),
    )


@router.get("/invoices/vendor/{vendor_name}", response_model=List[InvoiceSchema])
def get_invoices_by_vendor(vendor_name: str, db: Session = Depends(get_db)):
    invoices = fetch_invoices_by_vendor_name(db, vendor_name)

    # For list view, you may return empty Items/Confidence to keep it light
    # (still matches schema because Items defaults to [])
    return [
        InvoiceSchema(
            InvoiceId=getattr(inv, "InvoiceId", None),
            VendorName=getattr(inv, "VendorName", None),
            InvoiceDate=getattr(inv, "InvoiceDate", None),
            BillingAddressRecipient=getattr(inv, "BillingAddressRecipient", None),
            ShippingAddress=getattr(inv, "ShippingAddress", None),
            SubTotal=getattr(inv, "SubTotal", None),
            ShippingCost=getattr(inv, "ShippingCost", None),
            InvoiceTotal=getattr(inv, "InvoiceTotal", None),
            Items=[],
            Confidence=None,
        )
        for inv in invoices
    ]
