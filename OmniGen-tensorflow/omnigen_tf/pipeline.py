"""OmniGen Pipeline for image generation."""

import os
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import json

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
    
    def __init__(
        self,
        model,
        processor,
        scheduler,
        device=None
    ):
        """Initialize pipeline."""
        self.model = model
        self.processor = processor
        self.scheduler = scheduler
        
        # Set device
        if device is None:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Configure GPU to use memory growth
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    self.device = "/GPU:0"
                    print("Using GPU for inference")
                except RuntimeError as e:
                    print(f"GPU error: {e}")
                    self.device = "/CPU:0"
            else:
                print("No GPU found, using CPU for inference. This will be slow!")
                self.device = "/CPU:0"
        else:
            self.device = device
            
        # Move model to device and set to eval mode
        self.model_to_device()
        
    def model_to_device(self):
        """Move model to device and set to eval mode."""
        # Enable mixed precision if using GPU
        if self.device == "/GPU:0":
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
        # Create a new model instance with the same config
        with tf.device(self.device):
            config = self.model.get_config()
            self.model = self.model.__class__.from_config(config)
            
            # Copy weights from original model
            self.model.set_weights(self.model.get_weights())

    def __call__(
        self,
        prompt,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
    ):
        """Generate image from text prompt using GPU acceleration."""
        with tf.device(self.device):
            # Process text
            inputs = self.processor(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="tf"
            )
            input_ids = inputs["input_ids"]
            
            # Initialize latents on GPU
            latents_shape = (1, height // 8, width // 8, 4)
            latents = tf.random.normal(latents_shape)
            
            # Set timesteps
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps
            
            # Prepare extra kwargs for the scheduler step
            extra_step_kwargs = {}
            
            # Denoising loop
            for i, t in enumerate(timesteps):
                # Expand latents for classifier-free guidance
                latent_model_input = tf.concat([latents] * 2, axis=0)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # Predict noise residual
                with tf.GradientTape() as tape:
                    noise_pred = self.model(
                        input_ids=input_ids,
                        latents=latent_model_input,
                        timestep=t,
                        return_dict=False
                    )[0]
                    
                # Perform guidance
                noise_pred_uncond, noise_pred_text = tf.split(noise_pred, num_or_size_splits=2, axis=0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Compute previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
                
            # Scale and decode the image latents
            latents = latents * 0.18215
            image = self.model.decode(latents)
            
            # Post-process image
            image = (image / 2 + 0.5) * 255
            image = tf.clip_by_value(image, 0, 255)
            image = tf.cast(image, tf.uint8)
            
            # Convert to PIL Image
            image = Image.fromarray(image[0].numpy())
            
            return image
            
    def decode_latents(self, latents):
        """Decode latents to image using GPU."""
        with tf.device(self.device):
            # Scale latents
            latents = 1 / 0.18215 * latents
            
            # Process on GPU
            image = tf.transpose(latents, [0, 3, 1, 2])
            image = ((image + 1) / 2) * 255
            image = tf.clip_by_value(image, 0, 255)
            image = tf.cast(image, tf.uint8)
            
            # Move to CPU for PIL conversion
            image = image[0].numpy()
            image = Image.fromarray(np.transpose(image, [1, 2, 0]))
            
            return image
            
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
            
        # Initialize components
        model = OmniGen.from_pretrained(model_name)  # Config will be loaded from model_name/config.json
        processor = OmniGenProcessor.from_pretrained(model_name)
        scheduler = OmniGenScheduler()
        
        # Enable memory optimizations by default
        model.enable_memory_efficient_inference()
        
        return cls(model, processor, scheduler)
