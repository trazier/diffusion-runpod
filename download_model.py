import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def download_model():
    print("Downloading Stable Diffusion model...")
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # Download the model
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        revision="fp16",
        safety_checker=None
    )
    
    print("Model downloaded successfully!")

if __name__ == "__main__":
    download_model()