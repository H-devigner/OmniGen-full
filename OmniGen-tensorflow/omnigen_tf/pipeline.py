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
                # Enable memory growth to avoid OOM
                for gpu in tf.config.list_physical_devices('GPU'):
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except:
                        pass
                # Set mixed precision policy
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
            else:
                device = '/CPU:0'
        self.device = device
        
        # Create device strategy
        self.strategy = tf.distribute.OneDeviceStrategy(device)
        
    def __call__(
        self,
        prompt,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
    ):
        """Generate image from text prompt using GPU acceleration."""
        # Use distribution strategy for GPU operations
        with self.strategy.scope():
            # Process text
            inputs = self.processor(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="tf"
            )
            input_ids = tf.cast(inputs["input_ids"], tf.int32)
            attention_mask = tf.cast(inputs.get("attention_mask", None), tf.int32)
            
            # Initialize latents
            latent_height = height // 8
            latent_width = width // 8
            latents_shape = (1, latent_height, latent_width, 4)
            latents = tf.random.normal(latents_shape, dtype=tf.float16)
            
            # Set timesteps
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = tf.cast(self.scheduler.timesteps, tf.int32)
            
            # Denoising loop
            for i, t in enumerate(timesteps):
                timestep = tf.cast(t, tf.int32)
                
                # Process unconditional
                noise_pred_uncond = self.model(
                    latents=latents,
                    timestep=timestep,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    training=False
                )
                
                if isinstance(noise_pred_uncond, tuple):
                    noise_pred_uncond = noise_pred_uncond[0]
                elif isinstance(noise_pred_uncond, dict):
                    noise_pred_uncond = noise_pred_uncond["sample"]
                
                # Process conditional
                noise_pred_text = self.model(
                    latents=latents,
                    timestep=timestep,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    training=False
                )
                
                if isinstance(noise_pred_text, tuple):
                    noise_pred_text = noise_pred_text[0]
                elif isinstance(noise_pred_text, dict):
                    noise_pred_text = noise_pred_text["sample"]
                
                # Convert predictions
                noise_pred_uncond = tf.cast(self._convert_single_noise_pred(noise_pred_uncond, latents), tf.float16)
                noise_pred_text = tf.cast(self._convert_single_noise_pred(noise_pred_text, latents), tf.float16)
                
                # Apply guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Scheduler step
                latents = self.scheduler.step(noise_pred, timestep, latents)
                if isinstance(latents, dict):
                    latents = latents["prev_sample"]
                latents = tf.cast(latents, tf.float16)
            
            # Decode image
            latents = tf.cast(latents * 0.18215, tf.float16)
            image = self.model.decode(latents)
            
            # Resize if needed
            if image.shape[1:3] != (height, width):
                image = tf.image.resize(
                    image,
                    (height, width),
                    method=tf.image.ResizeMethod.BICUBIC
                )
            
            # Post-process image
            image = (image / 2 + 0.5)  # Normalize to [0, 1]
            image = tf.clip_by_value(image, 0, 1)  # Ensure values are in [0, 1]
            image = tf.cast(image * 255, tf.uint8)  # Scale to [0, 255]
            
            # Convert to PIL Image
            image_np = image[0].numpy()
            pil_image = Image.fromarray(image_np)
            
            return pil_image
            
    def _convert_single_noise_pred(self, noise_pred, latents):
        """Convert a single noise prediction to match latents shape."""
        # If noise_pred is from transformer output (B, seq_len, hidden_dim)
        if len(noise_pred.shape) == 3:
            # Reduce sequence length dimension
            noise_pred = tf.reduce_mean(noise_pred, axis=1)  # Now (B, hidden_dim)
            
            # Calculate target dimensions
            batch_size = latents.shape[0]
            height = latents.shape[1]
            width = latents.shape[2]
            channels = latents.shape[3]
            
            # For transformer output (3072 features), reshape to intermediate size
            if noise_pred.shape[-1] == 3072:
                # Reshape to 24x32x4 (3072 = 24*32*4)
                noise_pred = tf.reshape(noise_pred, (batch_size, 24, 32, channels))
            else:
                # For other sizes, try to maintain aspect ratio
                total_pixels = noise_pred.shape[-1] // channels
                side_length = int(tf.sqrt(float(total_pixels)))
                noise_pred = tf.reshape(noise_pred, (batch_size, side_length, -1, channels))
            
            # Resize to target dimensions using bicubic interpolation
            noise_pred = tf.image.resize(
                noise_pred,
                (height, width),
                method=tf.image.ResizeMethod.BICUBIC
            )
            
            # Ensure the output has the correct shape
            noise_pred.set_shape([batch_size, height, width, channels])
        
        return noise_pred

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
        model = OmniGen.from_pretrained(model_name)
        processor = OmniGenProcessor.from_pretrained(model_name)
        scheduler = OmniGenScheduler()
        
        # Enable memory optimizations by default
        model.enable_memory_efficient_inference()
        
        return cls(model, scheduler, processor)
pipeline = OmniGenPipeline.from_pretrained("Shitao/omnigen-v1")
image = pipeline(
    prompt="your prompt here",
    height=512,
    width=512,
    num_inference_steps=50,
    guidance_scale=7.5
)
