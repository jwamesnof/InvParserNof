from typing import List, Optional
from pydantic import BaseModel
from app.schemas.item_schema import ItemSchema
from app.schemas.confidence_schema import ConfidenceSchema


class InvoiceResponseSchema(BaseModel):
    invoice: dict
    items: List[ItemSchema] = []
    confidence: Optional[ConfidenceSchema] = None
