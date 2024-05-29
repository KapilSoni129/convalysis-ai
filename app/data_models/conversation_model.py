from pydantic import BaseModel
from typing import List

class Dialog(BaseModel):
    speaker: str
    message: str
    index: int | None = None

class Transcript(BaseModel):
    conversation: List[Dialog]
