from pydantic import BaseModel, ConfigDict
from typing import Optional

class ItemSchema(BaseModel):
    Description: Optional[str] = None
    Name: Optional[str] = None
    Quantity: Optional[float] = None
    UnitPrice: Optional[float] = None
    Amount: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)

