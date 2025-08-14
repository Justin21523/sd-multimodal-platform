# app/schemas/__init__.py
"""
Pydantic schemas for SD Multi-Modal Platform API.
"""

from .requests import Txt2ImgRequest, ModelSwitchRequest, BatchTxt2ImgRequest

from .responses import (
    Txt2ImgResponse,
    Txt2ImgResponseData,
    ErrorResponse,
    HealthResponse,
    ModelStatusResponse,
    ModelListResponse,
    ModelSwitchResponse,
    APIInfoResponse,
    ImageInfo,
    ModelInfo,
    GenerationParams,
    GenerationResults,
)

__all__ = [
    # Request schemas
    "Txt2ImgRequest",
    "ModelSwitchRequest",
    "BatchTxt2ImgRequest",
    # Response schemas
    "Txt2ImgResponse",
    "Txt2ImgResponseData",
    "ErrorResponse",
    "HealthResponse",
    "ModelStatusResponse",
    "ModelListResponse",
    "ModelSwitchResponse",
    "APIInfoResponse",
    # Component schemas
    "ImageInfo",
    "ModelInfo",
    "GenerationParams",
    "GenerationResults",
]
