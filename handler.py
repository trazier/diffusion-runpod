import os
import time
import base64
from io import BytesIO
import runpod
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

# Model cache to avoid reloading the model for every request
MODEL_CACHE = {}

def initialize_model():
    """Initialize and return the Stable Diffusion model."""
    if "model" not in MODEL_CACHE:
        print("Initializing Stable Diffusion model...")
        model_id = "runwayml/stable-diffusion-v1-5"  # A stable, reliable model
        
        # Load the model with optimizations - removed invalid revision parameter
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None  # Optional: Disable safety checker for speed
        )
        
        # Use DPMSolver++ scheduler for faster generation with good quality
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, 
            algorithm_type="dpmsolver++", 
            solver_order=2
        )
        
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()  # Memory optimization
        
        # Optional: Enable xformers for even better memory efficiency and speed
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            pipe.enable_xformers_memory_efficient_attention()
        
        MODEL_CACHE["model"] = pipe
        print("Model initialization complete")
    
    return MODEL_CACHE["model"]

def image_to_base64(image):
    """Convert a PIL Image to a base64 encoded string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def handler(event):
    """
    Handle incoming requests to generate images with Stable Diffusion.
    
    Expected input format:
    {
        "input": {
            "prompt": "a beautiful landscape with mountains",
            "negative_prompt": "blurry, bad quality",  # Optional
            "width": 512,  # Optional
            "height": 512,  # Optional
            "num_inference_steps": 30,  # Optional
            "guidance_scale": 7.5,  # Optional
            "seed": 42  # Optional
        }
    }
    """
    try:
        start_time = time.time()
        
        # Get the input parameters
        job_input = event["input"]
        
        # Extract parameters with default values
        prompt = job_input.get("prompt", "")
        if not prompt:
            return {"error": "Prompt is required"}
        
        negative_prompt = job_input.get("negative_prompt", "")
        width = job_input.get("width", 512)
        height = job_input.get("height", 512)
        num_inference_steps = job_input.get("num_inference_steps", 30)
        guidance_scale = job_input.get("guidance_scale", 7.5)
        seed = job_input.get("seed", None)
        
        # Initialize the model
        pipe = initialize_model()
        
        # Set the seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            generator = None
        
        # Generate the image
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
        
        # Convert the image to base64
        base64_image = image_to_base64(image)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return the result
        return {
            "image": base64_image,
            "processTime": processing_time,
            "parameters": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed
            }
        }
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error handling request: {error_trace}")
        return {"error": str(e)}

# Start the serverless handler
runpod.serverless.start({"handler": handler})