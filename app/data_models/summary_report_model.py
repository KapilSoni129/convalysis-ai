from pydantic import BaseModel

class TranscriptSummary(BaseModel):
    summary: str

class SummaryReport(BaseModel):
    report : TranscriptSummary