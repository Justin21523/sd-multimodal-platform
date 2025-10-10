# backend/app/schemas/vqa.py

from pydantic import BaseModel
from typing import Optional


class VQARequest(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    question: str


class VQAResponse(BaseModel):
    answer: str
    model: str
    processing_time: float
