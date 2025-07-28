from pydantic import BaseModel
from typing import Any

# Element class declaration
class Element(BaseModel):
    type: str
    text: Any
