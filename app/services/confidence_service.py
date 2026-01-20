"""
Confidence service - wrapper for confidence repository operations
Note: This is maintained for backward compatibility but extraction_service
directly uses confidence_repository.
"""
from sqlalchemy.orm import Session
from app.repositories import confidence_repository
from app.models.confidence_model import Confidence


def create_confidence(
    db: Session,
    invoice_id: str,
    vendor_name: float = None,
    invoice_date: float = None,
    billing_address_recipient: float = None,
    shipping_address: float = None,
    sub_total: float = None,
    shipping_cost: float = None,
    invoice_total: float = None,
    amount_due: float = None,
    vendor_name_logo: float = None,
) -> Confidence:
    """Create a confidence record for an invoice"""
    confidence = Confidence(
        InvoiceId=invoice_id,
        VendorName=vendor_name,
        InvoiceDate=invoice_date,
        BillingAddressRecipient=billing_address_recipient,
        ShippingAddress=shipping_address,
        SubTotal=sub_total,
        ShippingCost=shipping_cost,
        InvoiceTotal=invoice_total,
        AmountDue=amount_due,
        VendorNameLogo=vendor_name_logo,
    )
    return confidence_repository.create_confidence(db, confidence)
