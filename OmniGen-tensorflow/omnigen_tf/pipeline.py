"""OmniGen Pipeline for text-to-image generation."""

import os
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from omnigen_tf.model import OmniGen
from omnigen_tf.scheduler import OmniGenScheduler
from omnigen_tf.processor import OmniGenProcessor

# Enable mixed precision globally
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(f"GPU memory growth setting failed: {e}")

class OmniGenPipeline:
    """Pipeline for text-to-image generation using OmniGen."""
    
    def __init__(self, model, scheduler, processor, device=None):
        """Initialize pipeline."""
        self.model = model
        self.scheduler = scheduler
        self.processor = processor
        
        # Set device strategy
        if device is None:
            # Use GPU if available
            if tf.config.list_physical_devices('GPU'):
                device = '/GPU:0'
                print("Using GPU for inference")
            else:
                device = '/CPU:0'
                print("No GPU available, using CPU (this will be slow)")
        
        self.device = device
        self.strategy = tf.distribute.OneDeviceStrategy(device)
            
    @classmethod
    def from_pretrained(cls, model_name):
        """Load pretrained model."""
        if not os.path.exists(model_name):
            print(f"Model not found at {model_name}, downloading from HuggingFace...")
            cache_folder = os.getenv('HF_HUB_CACHE')
            model_name = snapshot_download(
                repo_id=model_name,
                cache_dir=cache_folder,
                ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5']
            )
            
        # Initialize components with GPU support
        with tf.device('/GPU:0'):
            model = OmniGen.from_pretrained(model_name)
            processor = OmniGenProcessor.from_pretrained(model_name)
            scheduler = OmniGenScheduler()
        
        return cls(model, scheduler, processor)

    def __call__(
        self,
        prompt,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=3.0,
        negative_prompt=None,
        **kwargs
    ):
        """Generate images from text prompt."""
        
        # Process inputs on GPU
        with tf.device(self.device):
            # Generate initial latents
            latents = tf.random.normal(
                [1, height // 8, width // 8, 4]
            )
            
            # Process text prompt
            text_inputs = self.processor.process_text(prompt)
            
            # Run diffusion process
            for t in range(num_inference_steps):
                # Convert timestep to tensor
                timestep = tf.convert_to_tensor([t])
                
                # Get model prediction
                noise_pred = self.model(
                    latents,
                    timestep,
                    text_inputs,
                    training=False
                )
                
                # Update latents
                latents = self.scheduler.step(
                    noise_pred,
                    timestep,
                    latents
                )
            
            # Decode latents to image
            images = self.decode_latents(latents)
            
        return self.numpy_to_pil(images)
    
    def decode_latents(self, latents):
        """Decode the latents into images."""
        with tf.device(self.device):
            # Scale and decode the image latents with vae
            latents = 1 / 0.18215 * latents
            
            # Decode with VAE
            images = self.model.vae.decode(latents)
            
            # Postprocess images
            images = (images / 2 + 0.5)
            images = tf.clip_by_value(images, 0.0, 1.0)
            images = tf.cast(images * 255, tf.uint8)
            
        return images
        
    def numpy_to_pil(self, images):
        """Convert a numpy image to a PIL image."""
        if images.ndim == 3:
            images = images[None, ...]
        
        pil_images = [Image.fromarray(image.numpy()) for image in images]
        return pil_images

    def generate_image(self, prompt, output_path=None, show_image=False):
        """Generate an image from a text prompt.
        
        Args:
            prompt (str): Text prompt to generate image from
            output_path (str, optional): Path to save generated image
            show_image (bool): Whether to display the image
            
        Returns:
            PIL.Image: Generated image
        """
        # Generate image
        image = self(
            prompt=prompt,
            height=128,  # Reduced height for faster generation
            width=128,   # Reduced width for faster generation
            num_inference_steps=50,
            guidance_scale=7.5
        )
        
        # Save image if output path provided
        if output_path:
            image.save(output_path)
            
        # Show image if requested
        if show_image:
            image.show()
            
        return image
