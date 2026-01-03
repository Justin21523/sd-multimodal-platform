# utils/attention_utils.py
"""
Attention processor utilities for RTX 5080 optimization and memory management.
Handles attention mechanism setup, memory optimizations, and hardware compatibility.
"""

import logging
import torch
from typing import Dict, Any, Optional, Union, List
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
)  # Import SDPA processor
from utils.logging_utils import get_request_logger
logger = logging.getLogger(__name__)


def setup_attention_processor(pipeline, force_sdpa: bool = False) -> str:
    """
    Setup attention processor for the pipeline with RTX 5080 optimization.

    Args:
        pipeline: Diffusers pipeline to configure
        force_sdpa: Force use of SDPA regardless of xformers availability

    Returns:
        str: Type of attention processor used ("sdpa", "default")
    """
    try:
        # Force SDPA for RTX 5080 or if explicitly requested
        if force_sdpa:
            logger.info("Forcing SDPA attention processor")
            return _setup_sdpa_attention(pipeline)

        # Fall back to SDPA (recommended for RTX 5080)
        return _setup_sdpa_attention(pipeline)

    except Exception as e:
        logger.warning(f"Attention processor setup failed: {e}, using default")
        return "default"


def _setup_sdpa_attention(pipeline) -> str:
    """Setup PyTorch Scaled Dot Product Attention."""
    try:
        # Set SDPA processor for UNet
        if hasattr(pipeline, "unet"):
            pipeline.unet.set_attn_processor(AttnProcessor2_0())
            logger.info("✅ SDPA attention processor enabled for UNet")

        # Set SDPA processor for VAE if it has attention layers
        if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "set_attn_processor"):
            try:
                pipeline.vae.set_attn_processor(AttnProcessor2_0())
                logger.info("✅ SDPA attention processor enabled for VAE")
            except:
                logger.debug("VAE doesn't support attention processor setting")

        return "sdpa"

    except ImportError:
        logger.error("AttnProcessor2_0 not available, using default attention")
        return "default"
    except Exception as e:
        logger.error(f"SDPA setup failed: {e}")
        return "default"


def setup_memory_optimizations(
    pipeline,
    attention_type: str = "auto",
    enable_cpu_offload: bool = False,
    enable_attention_slicing: bool = True,
    enable_vae_slicing: bool = True,
) -> List[str]:
    """
    Apply memory optimizations to the pipeline.

    Args:
        pipeline: Diffusers pipeline to optimize
        attention_type: Type of attention processor being used
        enable_cpu_offload: Whether to enable CPU offload
        enable_attention_slicing: Whether to enable attention slicing
        enable_vae_slicing: Whether to enable VAE slicing

    Returns:
        List[str]: List of applied optimizations
    """
    applied_optimizations = []

    try:
        # CPU Offload (saves VRAM but slows inference)
        if enable_cpu_offload:
            try:
                # Sequential CPU offload for maximum VRAM savings
                pipeline.enable_sequential_cpu_offload()
                applied_optimizations.append("sequential_cpu_offload")
                logger.info("✅ Sequential CPU offload enabled")
            except Exception as e:
                logger.warning(f"CPU offload failed: {e}")

        # Attention Slicing (reduces peak VRAM usage)
        if enable_attention_slicing:
            try:
                if hasattr(pipeline, "enable_attention_slicing"):
                    pipeline.enable_attention_slicing()
                    applied_optimizations.append("attention_slicing")
                    logger.info("✅ Attention slicing enabled")
            except Exception as e:
                logger.warning(f"Attention slicing failed: {e}")

        # VAE Slicing (reduces VAE memory usage)
        if enable_vae_slicing:
            try:
                if hasattr(pipeline, "enable_vae_slicing"):
                    pipeline.enable_vae_slicing()
                    applied_optimizations.append("vae_slicing")
                    logger.info("✅ VAE slicing enabled")
            except Exception as e:
                logger.warning(f"VAE slicing failed: {e}")

        # Memory format optimization
        try:
            if hasattr(pipeline, "unet"):
                pipeline.unet.to(memory_format=torch.channels_last)
                applied_optimizations.append("channels_last_format")
                logger.debug("UNet converted to channels_last format")
        except Exception as e:
            logger.debug(f"Memory format optimization failed: {e}")

    except Exception as e:
        logger.error(f"Memory optimization setup failed: {e}")

    logger.info(f"Applied memory optimizations: {applied_optimizations}")
    return applied_optimizations


def get_attention_info() -> Dict[str, Any]:
    """
    Get information about available attention processors and recommendations.

    Returns:
        Dict containing attention processor information and recommendations
    """
    info = {
        "available_processors": [],
        "recommended": "sdpa",
        "reason": "Default recommendation",
        "hardware_specific": False,
    }

    # Check SDPA availability
    try:
        info["available_processors"].append("sdpa")
    except ImportError:
        pass

    # Always have default
    info["available_processors"].append("default")

    # Hardware-specific recommendations
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

        # RTX 5080 specific
        if "RTX 50" in gpu_name:
            info["recommended"] = "sdpa"
            info["reason"] = (
                "RTX 5080 (sm_120) - SDPA optimized, xFormers may not be compatible"
            )
            info["hardware_specific"] = True

        # RTX 4090/4080 - xFormers still beneficial
        elif "RTX 40" in gpu_name:
            if "xformers" in info["available_processors"]:
                info["recommended"] = "xformers"
                info["reason"] = "RTX 4090/4080 - xFormers provides optimal performance"
            else:
                info["recommended"] = "sdpa"
                info["reason"] = (
                    "RTX 4090/4080 - SDPA fallback (xFormers not available)"
                )
            info["hardware_specific"] = True

        # RTX 3090/3080 - xFormers beneficial
        elif "RTX 30" in gpu_name:
            if "xformers" in info["available_processors"]:
                info["recommended"] = "xformers"
                info["reason"] = (
                    "RTX 3090/3080 - xFormers recommended for memory efficiency"
                )
            else:
                info["recommended"] = "sdpa"
                info["reason"] = "RTX 3090/3080 - SDPA fallback"
            info["hardware_specific"] = True

        # Other NVIDIA GPUs
        elif "GTX" in gpu_name or "RTX" in gpu_name:
            info["recommended"] = "sdpa"
            info["reason"] = "General NVIDIA GPU - SDPA provides good compatibility"
            info["hardware_specific"] = True

        # AMD/Intel GPUs
        else:
            info["recommended"] = "default"
            info["reason"] = "Non-NVIDIA GPU - default attention for compatibility"
            info["hardware_specific"] = True

        info["gpu_name"] = gpu_name

    return info


def optimize_for_inference(pipeline) -> Dict[str, Any]:
    """
    Apply inference-specific optimizations to the pipeline.

    Args:
        pipeline: Diffusers pipeline to optimize

    Returns:
        Dict with optimization results
    """
    optimizations = {"applied": [], "failed": [], "recommendations": []}

    try:
        # Set to eval mode
        if hasattr(pipeline, "unet"):
            pipeline.unet.eval()
            optimizations["applied"].append("unet_eval_mode")

        if hasattr(pipeline, "vae"):
            pipeline.vae.eval()
            optimizations["applied"].append("vae_eval_mode")

        # Disable gradient computation
        for param in pipeline.unet.parameters():
            param.requires_grad_(False)
        optimizations["applied"].append("gradients_disabled")

        # Memory format optimization
        try:
            if hasattr(pipeline, "unet"):
                pipeline.unet = pipeline.unet.to(memory_format=torch.channels_last)
                optimizations["applied"].append("channels_last_unet")
        except Exception as e:
            optimizations["failed"].append(f"channels_last_format: {e}")

        # Compile with PyTorch 2.0 if available
        if hasattr(torch, "compile"):
            try:
                pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead")
                optimizations["applied"].append("torch_compile_unet")
            except Exception as e:
                optimizations["failed"].append(f"torch_compile: {e}")
                optimizations["recommendations"].append(
                    "Consider updating PyTorch for compilation support"
                )

    except Exception as e:
        optimizations["failed"].append(f"general_optimization: {e}")

    return optimizations


def clear_gpu_cache():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("GPU cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear GPU cache: {e}")


def get_memory_stats() -> Dict[str, Any]:
    """
    Get current GPU memory statistics.

    Returns:
        Dict with memory usage information
    """
    stats = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "memory_info": [],
    }

    if torch.cuda.is_available():
        stats["device_count"] = torch.cuda.device_count()

        for i in range(torch.cuda.device_count()):
            device_stats = {
                "device_id": i,
                "device_name": torch.cuda.get_device_name(i),
                "total_memory_gb": round(
                    torch.cuda.get_device_properties(i).total_memory / (1024**3), 2
                ),
                "allocated_gb": round(torch.cuda.memory_allocated(i) / (1024**3), 2),
                "reserved_gb": round(torch.cuda.memory_reserved(i) / (1024**3), 2),
                "free_gb": 0,
            }

            device_stats["free_gb"] = round(
                device_stats["total_memory_gb"] - device_stats["allocated_gb"], 2
            )

            stats["memory_info"].append(device_stats)

    return stats


def estimate_vram_usage(
    model_type: str,
    width: int = 1024,
    height: int = 1024,
    batch_size: int = 1,
    dtype: str = "float16",
) -> Dict[str, float]:
    """
    Estimate VRAM usage for given parameters.

    Args:
        model_type: Type of model (sdxl-base, sd-1.5, etc.)
        width: Image width
        height: Image height
        batch_size: Number of images in batch
        dtype: Data type (float16, float32)

    Returns:
        Dict with VRAM estimates in GB
    """
    # Base VRAM requirements (in GB) for 512x512, batch=1, float32
    base_requirements = {
        "sd-1.5": 3.5,
        "sd-2.1": 4.5,
        "sdxl-base": 6.5,
        "controlnet": 1.5,  # Additional for ControlNet
        "lora": 0.2,  # Additional for LoRA
    }

    base_vram = base_requirements.get(model_type, 6.5)

    # Resolution scaling
    base_pixels = 512 * 512
    current_pixels = width * height
    resolution_factor = current_pixels / base_pixels

    # Batch scaling (not linear due to some fixed costs)
    batch_factor = batch_size * 0.8 + 0.2

    # Data type scaling
    dtype_factor = 0.5 if dtype == "float16" else 1.0

    # Calculate estimates
    estimated_vram = base_vram * resolution_factor * batch_factor * dtype_factor

    # Add overhead estimates
    overhead = estimated_vram * 0.3  # 30% overhead for safety
    peak_usage = estimated_vram * 1.5  # Peak during generation

    return {
        "base_estimate_gb": round(estimated_vram, 2),
        "with_overhead_gb": round(estimated_vram + overhead, 2),
        "peak_usage_gb": round(peak_usage, 2),
        "minimum_required_gb": round(estimated_vram + overhead, 2),
    }


def validate_memory_requirements(
    model_type: str,
    width: int,
    height: int,
    batch_size: int = 1,
    dtype: str = "float16",
) -> Dict[str, Any]:
    """
    Validate if current system can handle the memory requirements.

    Returns:
        Dict with validation results and recommendations
    """
    result = {
        "can_run": False,
        "estimated_usage": {},
        "available_memory": {},
        "recommendations": [],
    }

    # Get memory estimates
    result["estimated_usage"] = estimate_vram_usage(
        model_type, width, height, batch_size, dtype
    )

    # Get current memory stats
    result["available_memory"] = get_memory_stats()

    if torch.cuda.is_available() and result["available_memory"]["memory_info"]:
        device_info = result["available_memory"]["memory_info"][0]
        available_gb = device_info["free_gb"]
        required_gb = result["estimated_usage"]["minimum_required_gb"]

        result["can_run"] = available_gb >= required_gb

        if not result["can_run"]:
            deficit = required_gb - available_gb
            result["recommendations"].extend(
                [
                    f"Need {deficit:.1f}GB more VRAM",
                    "Consider reducing resolution or batch size",
                    "Enable CPU offload to reduce VRAM usage",
                    "Use attention slicing for memory efficiency",
                ]
            )

            # Suggest specific optimizations
            if dtype == "float32":
                result["recommendations"].append("Switch to float16 to save ~50% VRAM")

            if batch_size > 1:
                result["recommendations"].append(
                    f"Reduce batch_size from {batch_size} to 1"
                )

            if width > 512 or height > 512:
                result["recommendations"].append(
                    "Reduce resolution (e.g., 512x512 instead of 1024x1024)"
                )

    else:
        result["recommendations"].append(
            "Running on CPU - expect much slower performance"
        )
        result["can_run"] = True  # CPU can run but slowly

    return result


def setup_pipeline_optimizations(pipeline, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup complete pipeline optimizations based on configuration.

    Args:
        pipeline: Diffusers pipeline to optimize
        config: Configuration dictionary with optimization settings

    Returns:
        Dict with results of all optimizations applied
    """
    results = {
        "attention_processor": "default",
        "memory_optimizations": [],
        "inference_optimizations": {},
        "total_optimizations": 0,
        "warnings": [],
    }

    try:
        # Setup attention processor
        attention_type = setup_attention_processor(
            pipeline,
            force_sdpa=config.get("use_sdpa", True),
        )
        results["attention_processor"] = attention_type

        # Apply memory optimizations
        memory_opts = setup_memory_optimizations(
            pipeline,
            attention_type=attention_type,
            enable_cpu_offload=config.get("enable_cpu_offload", False),
            enable_attention_slicing=config.get("use_attention_slicing", True),
            enable_vae_slicing=config.get("enable_vae_slicing", True),
        )
        results["memory_optimizations"] = memory_opts

        # Apply inference optimizations
        inference_opts = optimize_for_inference(pipeline)
        results["inference_optimizations"] = inference_opts

        # Count total optimizations
        results["total_optimizations"] = (
            len(memory_opts)
            + len(inference_opts.get("applied", []))
            + (1 if attention_type != "default" else 0)
        )

        # Collect warnings
        if inference_opts.get("failed"):
            results["warnings"].extend(inference_opts["failed"])

        logger.info(
            f"Pipeline optimization complete: {results['total_optimizations']} optimizations applied"
        )

    except Exception as e:
        logger.error(f"Pipeline optimization failed: {e}")
        results["warnings"].append(f"Optimization error: {e}")

    return results
