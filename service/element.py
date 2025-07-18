from pydantic import BaseModel
from typing import Any

class Element(BaseModel):
    type: str
    text: Any
