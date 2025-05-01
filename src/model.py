import torch
from diffusers import StableDiffusionPipeline

class ImageGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        """
        Initialize the image generation model.
        
        Args:
            model_id (str): The Hugging Face model ID to load
        """
        self.model_id = model_id
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
        
    def _load_model(self):
        """Load the model into memory"""
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_safetensors=True
        )
        
        self.pipe = self.pipe.to(self.device)
        
        # Enable optimizations
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
        
        print(f"Model loaded on {self.device}")
    
    def generate(self, 
                prompt, 
                negative_prompt="", 
                width=512, 
                height=512, 
                num_inference_steps=50, 
                guidance_scale=7.5,
                seed=None):
        """
        Generate an image based on the provided parameters.
        
        Args:
            prompt (str): The text prompt
            negative_prompt (str): Negative text prompt
            width (int): Image width
            height (int): Image height
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): How closely to follow the prompt
            seed (int): Random seed for reproducibility
            
        Returns:
            PIL.Image: The generated image
        """
        # Set up generator for reproducibility if seed is provided
        generator = None
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(seed)
        
        # Generate the image
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )
        
        return result.images[0]