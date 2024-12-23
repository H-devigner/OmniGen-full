"""OmniGen Pipeline for image generation."""

import os
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from omnigen_tf.model import OmniGen
from omnigen_tf.scheduler import OmniGenScheduler
from omnigen_tf.processor import OmniGenProcessor

# Configure GPU memory growth before any other TensorFlow operations
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU memory growth setting failed: {e}")

class OmniGenPipeline:
    """Pipeline for text-to-image generation using OmniGen."""
    
    def __init__(self, model, processor, scheduler, device="/CPU:0"):
        """Initialize pipeline.
        
        Args:
            model: OmniGen model
            processor: Text processor
            scheduler: Diffusion scheduler
            device: Device to run on
        """
        self.model = model
        self.processor = processor
        self.scheduler = scheduler
        self.device = device
        self._model_on_cpu = False
        self._vae_on_cpu = False
        
    @classmethod
    def from_pretrained(cls, model_name, device="/CPU:0"):
        """Load pipeline from pretrained model.
        
        Args:
            model_name: Name or path of pretrained model
            device: Device to run on
        """
        # Download model if needed
        if not os.path.exists(model_name):
            cache_folder = os.getenv('HF_HUB_CACHE')
            model_name = snapshot_download(
                repo_id=model_name,
                cache_dir=cache_folder,
                ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5']
            )
        
        # Initialize components
        model = OmniGen.from_pretrained(model_name)
        processor = OmniGenProcessor.from_pretrained(model_name)
        scheduler = OmniGenScheduler()
        
        # Create pipeline
        pipeline = cls(model, processor, scheduler, device)
        
        # Move to device
        if device:
            pipeline._move_to_device(pipeline.model, device)
        
        return pipeline
        
    def _move_to_device(self, model, device):
        """Move model to specified device."""
        if not device:
            return
            
        # Handle device string
        if isinstance(device, str):
            if device.startswith('/'):
                device = device[1:]  # Remove leading slash
                
        # Check if it's a TensorFlow model
        if hasattr(model, 'variables'):
            print(f"\nMoving TensorFlow model to {device}")
            with tf.device(device):
                for var in model.variables:
                    if isinstance(var, tf.Variable):
                        var.assign(tf.identity(var))
        # PyTorch model
        elif hasattr(model, 'to'):
            print(f"\nMoving PyTorch model to {device}")
            if device.upper() == 'GPU:0':
                device = 'cuda'
            elif device.upper() == 'CPU:0':
                device = 'cpu'
            model.to(device)
        else:
            print(f"Warning: Unknown model type, cannot move to {device}")
            
    def __call__(self, prompt, height=512, width=512, num_inference_steps=50, guidance_scale=7.5):
        """Generate image from text prompt.
        
        Args:
            prompt: Text prompt
            height: Output image height
            width: Output image width
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            PIL.Image: Generated image
        """
        # Process prompt
        inputs = self.processor(prompt)
        
        # Get input shape
        batch_size = inputs["input_ids"].shape[0]
        
        # Initialize latents
        latents_shape = (batch_size, 4, height // 8, width // 8)
        latents = tf.random.normal(latents_shape)
        
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        
        # Denoising loop
        for t in timesteps:
            # Get model prediction
            with tf.device(self.device):
                noise_pred = self.model(
                    latents,
                    t,
                    inputs["input_ids"],
                    inputs["attention_mask"]
                )
            
            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents)
            
        # Decode latents to image
        with tf.device(self.device):
            image = self.model.decode_latents(latents)
            
        # Convert to PIL
        image = tf.cast((image + 1.0) * 127.5, tf.uint8)
        image = tf.transpose(image, [0, 2, 3, 1])
        image = Image.fromarray(image[0].numpy())
        
        return image
