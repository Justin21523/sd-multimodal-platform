# backend/app/schemas/caption.py

from pydantic import BaseModel
from typing import Optional


class CaptionRequest(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None  # 二選一


class CaptionResponse(BaseModel):
    caption: str
    model: str
    processing_time: float
