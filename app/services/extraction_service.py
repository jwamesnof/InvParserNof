from sqlalchemy.orm import Session
from typing import Dict, List

from app.models.invoice_model import Invoice
from app.models.item_model import Item
from app.models.confidence_model import Confidence

from app.repositories.invoice_repository import create_invoice
from app.repositories.item_repository import create_items
from app.repositories.confidence_repository import create_confidence


def save_extracted_invoice(
    db: Session,
    data: Dict,
    confidences: Dict,
    document_confidence: float
) -> None:
    invoice_id = data.get("InvoiceId")
    if invoice_id is None:
        # If no InvoiceId, we can’t persist correctly
        return

    invoice_id = str(invoice_id)

    invoice = Invoice(
        InvoiceId=invoice_id,
        VendorName=data.get("VendorName"),
        InvoiceDate=data.get("InvoiceDate"),
        BillingAddressRecipient=data.get("BillingAddressRecipient"),
        ShippingAddress=data.get("ShippingAddress"),
        SubTotal=data.get("SubTotal"),
        ShippingCost=data.get("ShippingCost"),
        InvoiceTotal=data.get("InvoiceTotal"),
    )

    create_invoice(db, invoice)

    confidence = Confidence(
        InvoiceId=invoice_id,
        VendorName=confidences.get("VendorName"),
        InvoiceDate=confidences.get("InvoiceDate"),
        BillingAddressRecipient=confidences.get("BillingAddressRecipient"),
        ShippingAddress=confidences.get("ShippingAddress"),
        SubTotal=confidences.get("SubTotal"),
        ShippingCost=confidences.get("ShippingCost"),
        InvoiceTotal=confidences.get("InvoiceTotal"),
    )

    create_confidence(db, confidence)

    items: List[Item] = []
    for item in data.get("Items", []) or []:
        items.append(
            Item(
                InvoiceId=invoice_id,
                Description=item.get("Description"),
                Name=item.get("Name"),
                Quantity=item.get("Quantity"),
                UnitPrice=item.get("UnitPrice"),
                Amount=item.get("Amount"),
            )
        )

    if items:
        create_items(db, items)
