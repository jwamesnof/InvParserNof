from pydantic import BaseModel
from typing import Dict, Any, Optional

class ExtractResponseSchema(BaseModel):
    confidence: float
    data: Dict[str, Any]
    dataConfidence: Dict[str, Optional[float]]

