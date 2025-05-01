import os
import time
import base64
from io import BytesIO
import runpod
from PIL import Image

# Import model-related components
from diffusers import StableDiffusionPipeline
import torch

# Initialize the model (done only once when the container starts)
def init_model():
    print("Initializing Stable Diffusion model...")
    model_id = "runwayml/stable-diffusion-v1-5"  # A reliable, stable model
    
    # Load the pipeline with CUDA acceleration if available
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        print("CUDA not available, using CPU (this will be slow)")
    
    # Optional: Enable attention slicing for lower memory usage
    pipe.enable_attention_slicing()
    
    return pipe

# Load the model globally so it's only loaded once
MODEL = init_model()

def handler(event):
    """
    This is the handler function that processes the incoming API requests.
    
    Args:
        event (dict): JSON object with the following required fields:
            - prompt (str): Text prompt for image generation
            
            Optional fields:
            - negative_prompt (str): Text to avoid in the image
            - width (int): Image width (default: 512)
            - height (int): Image height (default: 512)
            - num_inference_steps (int): Number of denoising steps (default: 50)
            - guidance_scale (float): How closely to follow the prompt (default: 7.5)
            - seed (int): Random seed for reproducibility (default: random)
    
    Returns:
        dict: JSON object containing the generated image(s) and metadata
    """
    try:
        # Start timing for performance logging
        start_time = time.time()
        
        # Extract params from the event
        job_input = event["input"]
        
        # Required parameters
        prompt = job_input.get("prompt")
        if not prompt:
            return {"error": "No prompt provided"}
        
        # Optional parameters with defaults
        negative_prompt = job_input.get("negative_prompt", "")
        width = int(job_input.get("width", 512))
        height = int(job_input.get("height", 512))
        num_inference_steps = int(job_input.get("num_inference_steps", 50))
        guidance_scale = float(job_input.get("guidance_scale", 7.5))
        
        # Set seed for reproducibility if provided
        generator = None
        if "seed" in job_input:
            seed = int(job_input.get("seed"))
            generator = torch.Generator("cuda").manual_seed(seed)
            print(f"Using seed: {seed}")
        
        # Generate the image
        print(f"Generating image with prompt: {prompt}")
        with torch.inference_mode():
            result = MODEL(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                num_images_per_prompt=1,
            )
        
        # Check for safety issues
        if hasattr(result, "nsfw_content_detected") and result.nsfw_content_detected:
            return {"error": "NSFW content detected"}
        
        # Process and encode the image
        image = result.images[0]
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Measure execution time
        execution_time = time.time() - start_time
        
        # Return the results
        return {
            "image": img_str,
            "format": "PNG",
            "metadata": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed if "seed" in job_input else "random",
                "execution_time": f"{execution_time:.2f} seconds"
            }
        }
    
    except Exception as e:
        # Log the error and return it
        error_message = str(e)
        print(f"Error in handler: {error_message}")
        return {"error": error_message}

# Start the serverless handler
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})