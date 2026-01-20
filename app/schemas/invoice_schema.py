from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from app.schemas.item_schema import ItemSchema
from app.schemas.confidence_schema import ConfidenceSchema

class InvoiceSchema(BaseModel):
    InvoiceId: str
    VendorName: Optional[str] = None
    InvoiceDate: Optional[str] = None
    BillingAddressRecipient: Optional[str] = None
    ShippingAddress: Optional[str] = None
    SubTotal: Optional[float] = None
    ShippingCost: Optional[float] = None
    InvoiceTotal: Optional[float] = None
    Items: List[ItemSchema] = []
    Confidence: Optional[ConfidenceSchema] = None

    model_config = ConfigDict(from_attributes=True)
