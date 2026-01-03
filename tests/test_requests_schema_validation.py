# tests/test_requests_schema_validation.py

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from app.schemas.requests import (
    CaptionRequest,
    ControlNetConfig,
    GenerationParams,
    Img2ImgRequest,
    InpaintRequest,
    PromptTemplate,
    VQARequest,
)
from utils.image_utils import create_test_image, pil_image_to_base64


@pytest.mark.unit
def test_prompt_template_apply_to_prompt_normalizes_whitespace():
    tpl = PromptTemplate(
        name="t",
        style="photo",
        positive_prefix="  foo  ",
        positive_suffix="  bar  ",
        negative_prompt="  bad  anatomy  ",
    )
    positive, negative = tpl.apply_to_prompt("  user   prompt  ")
    assert positive == "foo user prompt bar"
    assert negative == "bad anatomy"


@pytest.mark.unit
def test_generation_params_normalize_seed_and_dimensions():
    params = GenerationParams(width=513, height=513, seed=-1)
    assert params.width % 8 == 0
    assert params.height % 8 == 0
    assert params.seed is None


@pytest.mark.unit
def test_caption_and_vqa_require_image_source():
    with pytest.raises(ValueError):
        CaptionRequest(max_length=20)
    with pytest.raises(ValueError):
        VQARequest(question="what?", max_length=20)


@pytest.mark.unit
def test_controlnet_config_accepts_asset_id_and_rejects_multiple_sources():
    asset_id = str(uuid.uuid4())
    cfg = ControlNetConfig(type="canny", asset_id=asset_id)
    assert cfg.asset_id == asset_id

    img = create_test_image(8, 8, "RGB")
    b64 = pil_image_to_base64(img)
    with pytest.raises(ValueError):
        ControlNetConfig(type="canny", image=b64, asset_id=asset_id)


@pytest.mark.unit
def test_img2img_and_inpaint_accept_allowed_image_paths(mock_settings, tmp_path):
    outputs_root = Path(str(mock_settings.OUTPUT_PATH))
    outputs_root.mkdir(parents=True, exist_ok=True)

    src_path = outputs_root / "src.png"
    create_test_image(8, 8, "RGB").save(src_path, format="PNG")

    req = Img2ImgRequest(prompt="p", image_path=str(src_path), strength=0.5)
    assert req.image_path == str(src_path.resolve())

    mask_path = outputs_root / "mask.png"
    create_test_image(8, 8, "L", fill_color=255).save(mask_path, format="PNG")

    inpaint = InpaintRequest(
        prompt="p",
        image_path=str(src_path),
        mask_path=str(mask_path),
        strength=0.5,
    )
    assert inpaint.image_path == str(src_path.resolve())
    assert inpaint.mask_path == str(mask_path.resolve())

