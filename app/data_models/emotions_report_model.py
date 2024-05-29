from pydantic import BaseModel
from typing import List

class SpeakerEmotions(BaseModel):
    speaker: str
    primary_emotion: str | None
    secondary_emotion: str | None

class EmotionsReport(BaseModel):
    report : List[SpeakerEmotions]  
