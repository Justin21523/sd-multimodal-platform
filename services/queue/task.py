# services/queue/tasks.py
"""
Celery task implementations for generation and post-processing
"""

import time
import asyncio
from typing import Dict, List, Optional, Any
from celery import Celery
from celery.exceptions import Retry
from celery.result import AsyncResult
from PIL import Image

from services.models.sd_models import get_model_manager
from services.postprocess.pipeline_manager import (
    get_pipeline_manager,
    create_standard_pipeline,
    create_fast_pipeline,
    create_quality_pipeline,
)
from services.queue.task_manager import get_task_manager, TaskStatus
from utils.logging_utils import get_generation_logger
from utils.image_utils import base64_to_pil_image, pil_image_to_base64
from utils.file_utils import save_generation_output
from utils.metadata_utils import save_generation_metadata

# Create Celery app instance
celery_app = Celery("sd_multimodal")


class BaseGenerationTask(celery_app.Task):
    """Base class for generation tasks with common functionality"""

    def on_start(self, task_id, args, kwargs):
        """Called when task starts"""
        self.update_state(
            state="STARTED", meta={"progress": 0.0, "stage": "initializing"}
        )

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails"""
        logger = get_generation_logger("task", "celery")
        logger.error(f"Task {task_id} failed: {str(exc)}")

    def on_success(self, retval, task_id, args, kwargs):
        """Called when task succeeds"""
        logger = get_generation_logger("task", "celery")
        logger.info(f"Task {task_id} completed successfully")


@celery_app.task(bind=True, base=BaseGenerationTask, name="tasks.generate_txt2img")
def generate_txt2img(
    self,
    task_id: str,
    generation_params: Dict[str, Any],
    postprocess_chain: Optional[List[str]] = None,
):
    """Generate image from text prompt"""

    logger = get_generation_logger("txt2img", "celery")
    logger.info(f"Starting txt2img generation for task {task_id}")

    try:
        # Update progress
        self.update_state(
            state="PROGRESS", meta={"progress": 0.1, "stage": "loading_model"}
        )

        # Get model manager
        model_manager = get_model_manager()

        # Auto-select model if needed
        target_model = generation_params.get("model_id")
        if not target_model or target_model == "auto":
            target_model = model_manager.auto_select_model(
                generation_params["prompt"], "txt2img"
            )

        # Switch model if necessary
        if model_manager.current_model != target_model:
            self.update_state(
                state="PROGRESS", meta={"progress": 0.2, "stage": "switching_model"}
            )

            success = asyncio.run(model_manager.switch_model(target_model))
            if not success:
                raise Exception(f"Failed to switch to model: {target_model}")

        # Update progress
        self.update_state(
            state="PROGRESS", meta={"progress": 0.3, "stage": "generating_image"}
        )

        # Generate image
        generation_result = asyncio.run(
            model_manager.generate_image(
                prompt=generation_params["prompt"],
                negative_prompt=generation_params.get("negative_prompt", ""),
                width=generation_params.get("width", 1024),
                height=generation_params.get("height", 1024),
                num_inference_steps=generation_params.get("steps", 25),
                guidance_scale=generation_params.get("cfg_scale", 7.5),
                seed=generation_params.get("seed"),  # type: ignore
            )
        )

        # Save generated images
        self.update_state(
            state="PROGRESS", meta={"progress": 0.7, "stage": "saving_images"}
        )

        saved_images = []
        for i, image in enumerate(generation_result["images"]):
            image_path = asyncio.run(save_generation_output(image, task_id, "txt2img"))
            saved_images.append(str(image_path))

        # Save metadata
        metadata = {
            **generation_result["generation_params"],
            "model_used": generation_result["model_used"],
            "generation_time": generation_result.get("generation_time", 0),
            "vram_used": generation_result.get("vram_used", "N/A"),
            "task_id": task_id,
            "image_paths": saved_images,
        }

        asyncio.run(save_generation_metadata(metadata, task_id, "txt2img"))

        # Run post-processing if requested
        final_images = saved_images.copy()
        if postprocess_chain:
            self.update_state(
                state="PROGRESS", meta={"progress": 0.8, "stage": "post_processing"}
            )

            final_images = []
            pipeline_manager = get_pipeline_manager()

            for image_path in saved_images:
                # Load image for post-processing
                image = Image.open(image_path)

                # Create post-processing pipeline
                pipeline = create_standard_pipeline()

                # Execute pipeline
                postprocess_result = asyncio.run(
                    pipeline.execute(image, f"{task_id}_postprocess")
                )

                # Save post-processed image
                processed_path = asyncio.run(
                    save_generation_output(
                        postprocess_result["processed_image"],
                        f"{task_id}_processed",
                        "txt2img",
                    )
                )
                final_images.append(str(processed_path))

        # Final progress update
        self.update_state(
            state="PROGRESS", meta={"progress": 1.0, "stage": "completed"}
        )

        # Return results
        result = {
            "task_id": task_id,
            "task_type": "txt2img",
            "images": final_images,
            "metadata": metadata,
            "postprocessed": bool(postprocess_chain),
        }

        logger.info(f"Completed txt2img generation for task {task_id}")
        return result

    except Exception as e:
        logger.error(f"txt2img generation failed for task {task_id}: {str(e)}")
        raise


@celery_app.task(bind=True, base=BaseGenerationTask, name="tasks.generate_img2img")
def generate_img2img(
    self,
    task_id: str,
    generation_params: Dict[str, Any],
    postprocess_chain: Optional[List[str]] = None,
):
    """Generate image from image and text prompt"""

    logger = get_generation_logger("img2img", "celery")
    logger.info(f"Starting img2img generation for task {task_id}")

    try:
        # Update progress
        self.update_state(
            state="PROGRESS", meta={"progress": 0.1, "stage": "loading_model"}
        )

        # Get model manager
        model_manager = get_model_manager()

        # Prepare input image
        init_image = base64_to_pil_image(generation_params["init_image"])

        # Auto-select model if needed
        target_model = generation_params.get("model_id")
        if not target_model or target_model == "auto":
            target_model = model_manager.auto_select_model(
                generation_params["prompt"], "img2img"
            )

        # Switch model if necessary
        if model_manager.current_model != target_model:
            self.update_state(
                state="PROGRESS", meta={"progress": 0.2, "stage": "switching_model"}
            )

            success = asyncio.run(model_manager.switch_model(target_model))
            if not success:
                raise Exception(f"Failed to switch to model: {target_model}")

        # Update progress
        self.update_state(
            state="PROGRESS", meta={"progress": 0.3, "stage": "generating_image"}
        )

        # Generate image
        generation_result = asyncio.run(
            model_manager.generate_img2img(
                prompt=generation_params["prompt"],
                image=init_image,
                negative_prompt=generation_params.get("negative_prompt", ""),
                strength=generation_params.get("strength", 0.75),
                num_inference_steps=generation_params.get("steps", 25),
                guidance_scale=generation_params.get("cfg_scale", 7.5),
                seed=generation_params.get("seed"),  # type: ignore
            )
        )

        # Save and process results (similar to txt2img)
        self.update_state(
            state="PROGRESS", meta={"progress": 0.7, "stage": "saving_images"}
        )

        saved_images = []
        for i, image in enumerate(generation_result["images"]):
            image_path = asyncio.run(save_generation_output(image, task_id, "img2img"))
            saved_images.append(str(image_path))

        # Save metadata
        metadata = {
            **generation_result["generation_params"],
            "model_used": generation_result["model_used"],
            "generation_time": generation_result.get("generation_time", 0),
            "vram_used": generation_result.get("vram_used", "N/A"),
            "task_id": task_id,
            "image_paths": saved_images,
        }

        asyncio.run(save_generation_metadata(metadata, task_id, "img2img"))

        # Post-processing if requested
        final_images = saved_images.copy()
        if postprocess_chain:
            self.update_state(
                state="PROGRESS", meta={"progress": 0.8, "stage": "post_processing"}
            )

            final_images = []
            pipeline_manager = get_pipeline_manager()

            for image_path in saved_images:
                image = Image.open(image_path)

                pipeline = create_standard_pipeline()
                postprocess_result = asyncio.run(
                    pipeline.execute(image, f"{task_id}_postprocess")
                )

                processed_path = asyncio.run(
                    save_generation_output(
                        postprocess_result["processed_image"],
                        f"{task_id}_processed",
                        "img2img",
                    )
                )
                final_images.append(str(processed_path))

        # Return results
        result = {
            "task_id": task_id,
            "task_type": "img2img",
            "images": final_images,
            "metadata": metadata,
            "postprocessed": bool(postprocess_chain),
        }

        logger.info(f"Completed img2img generation for task {task_id}")
        return result

    except Exception as e:
        logger.error(f"img2img generation failed for task {task_id}: {str(e)}")
        raise


@celery_app.task(bind=True, base=BaseGenerationTask, name="tasks.run_postprocess")
def run_postprocess(self, task_id: str, postprocess_params: Dict[str, Any]):
    """Run post-processing pipeline on existing images"""

    logger = get_generation_logger("postprocess", "celery")
    logger.info(f"Starting post-processing for task {task_id}")

    try:
        # Update progress
        self.update_state(
            state="PROGRESS", meta={"progress": 0.1, "stage": "loading_images"}
        )

        # Load input images
        input_images = []
        image_paths = postprocess_params.get("image_paths", [])

        if not image_paths:
            raise ValueError("No input images provided for post-processing")

        for path in image_paths:
            if path.startswith("data:image"):
                # Base64 encoded image
                image = base64_to_pil_image(path)
            else:
                # File path
                image = Image.open(path)
            input_images.append(image)

        # Update progress
        self.update_state(
            state="PROGRESS", meta={"progress": 0.2, "stage": "preparing_pipeline"}
        )

        # Create post-processing pipeline
        pipeline_manager = get_pipeline_manager()

        pipeline_type = postprocess_params.get("pipeline_type", "standard")
        if pipeline_type == "standard":
            pipeline = create_standard_pipeline()
        elif pipeline_type == "fast":
            pipeline = create_fast_pipeline()
        elif pipeline_type == "quality":
            pipeline = create_quality_pipeline()
        else:
            # Custom pipeline
            pipeline = get_pipeline_manager()
            steps = postprocess_params.get("steps", [])
            for step in steps:
                pipeline.add_step(step["type"], step["model"], **step.get("params", {}))

        # Process images
        processed_images = []
        total_images = len(input_images)

        for i, image in enumerate(input_images):
            self.update_state(
                state="PROGRESS",
                meta={
                    "progress": 0.3 + (0.6 * i / total_images),
                    "stage": f"processing_image_{i+1}_of_{total_images}",
                },
            )

            # Execute pipeline
            postprocess_result = asyncio.run(pipeline.execute(image, f"{task_id}_{i}"))

            # Save processed image
            processed_path = asyncio.run(
                save_generation_output(
                    postprocess_result["processed_image"],
                    f"{task_id}_processed_{i}",
                    "postprocess",
                )
            )

            processed_images.append(
                {
                    "path": str(processed_path),
                    "processing_time": postprocess_result["total_processing_time"],
                    "steps_executed": postprocess_result["steps_executed"],
                    "vram_used": postprocess_result["vram_used"],
                }
            )

        # Save metadata
        self.update_state(
            state="PROGRESS", meta={"progress": 0.9, "stage": "saving_metadata"}
        )

        metadata = {
            "task_id": task_id,
            "pipeline_type": pipeline_type,
            "input_images": len(input_images),
            "processed_images": processed_images,
            "total_processing_time": sum(
                img["processing_time"] for img in processed_images
            ),
        }

        asyncio.run(save_generation_metadata(metadata, task_id, "postprocess"))

        # Return results
        result = {
            "task_id": task_id,
            "task_type": "postprocess",
            "processed_images": [img["path"] for img in processed_images],
            "metadata": metadata,
        }

        logger.info(f"Completed post-processing for task {task_id}")
        return result

    except Exception as e:
        logger.error(f"Post-processing failed for task {task_id}: {str(e)}")
        raise


@celery_app.task(bind=True, name="tasks.coordinate_batch")
def coordinate_batch(self, batch_id: str, subtask_ids: List[str]):
    """Coordinate batch processing and aggregate results"""

    logger = get_generation_logger("batch", "celery")
    logger.info(f"Starting batch coordination for {batch_id}")

    try:
        # Monitor subtasks
        completed_tasks = []
        failed_tasks = []
        total_tasks = len(subtask_ids)

        while len(completed_tasks) + len(failed_tasks) < total_tasks:
            # Update progress
            progress = (len(completed_tasks) + len(failed_tasks)) / total_tasks
            self.update_state(
                state="PROGRESS",
                meta={
                    "progress": progress,
                    "completed": len(completed_tasks),
                    "failed": len(failed_tasks),
                    "total": total_tasks,
                },
            )

            # Check subtask status
            for subtask_id in subtask_ids:
                if subtask_id in completed_tasks or subtask_id in failed_tasks:
                    continue

                result = AsyncResult(subtask_id, app=celery_app)

                if result.state == "SUCCESS":
                    completed_tasks.append(subtask_id)
                elif result.state == "FAILURE":
                    failed_tasks.append(subtask_id)

            # Wait before next check
            time.sleep(2)

        # Aggregate results
        batch_results = {
            "batch_id": batch_id,
            "total_tasks": total_tasks,
            "completed_tasks": len(completed_tasks),
            "failed_tasks": len(failed_tasks),
            "success_rate": len(completed_tasks) / total_tasks,
            "subtask_results": {},
        }

        # Collect individual results
        for subtask_id in completed_tasks:
            result = AsyncResult(subtask_id, app=celery_app)
            batch_results["subtask_results"][subtask_id] = result.result

        logger.info(
            f"Batch {batch_id} completed: {len(completed_tasks)}/{total_tasks} successful"
        )
        return batch_results

    except Exception as e:
        logger.error(f"Batch coordination failed for {batch_id}: {str(e)}")
        raise
