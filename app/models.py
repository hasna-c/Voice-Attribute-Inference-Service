from pydantic import BaseModel, Field
from typing import Literal, Optional

class AttributePrediction(BaseModel):
    prediction: str
    confidence: float = Field(..., ge=0.0, le=1.0)

class AudioAnalysisResponse(BaseModel):
    contact_id: str
    gender: AttributePrediction
    age_bracket: AttributePrediction
    language: Optional[AttributePrediction] = None
    processing_ms: int
    audio_quality: Literal["good", "degraded", "insufficient"]