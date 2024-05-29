from pydantic import BaseModel
from typing import List

class SpeakerRelationship(BaseModel):
    speaker1 : str
    speaker2 : str
    agreement_score : float

class SpeakerRelationshipsReport(BaseModel):
    report : List[SpeakerRelationship]