from pydantic import BaseModel, ConfigDict
from typing import Optional

class ConfidenceSchema(BaseModel):
    VendorName: Optional[float] = None
    InvoiceDate: Optional[float] = None
    BillingAddressRecipient: Optional[float] = None
    ShippingAddress: Optional[float] = None
    SubTotal: Optional[float] = None
    ShippingCost: Optional[float] = None
    InvoiceTotal: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)

