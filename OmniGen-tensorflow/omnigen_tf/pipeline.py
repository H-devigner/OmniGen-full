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
            
            # Initialize latents on GPU with smaller batch size
            latents_shape = (1, height // 8, width // 8, 4)
            latents = tf.random.normal(latents_shape, dtype=tf.float16)
            
            # Set timesteps
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps
            
            # Pre-compute static tensors on GPU
            timesteps = tf.cast(timesteps, tf.int32)
            
            # Denoising loop with memory-efficient processing
            for i, t in enumerate(timesteps):
                # Convert timestep to scalar
                timestep = tf.cast(t, tf.int32)
                
                # Process unconditional and conditional separately to save memory
                # First process unconditional (no prompt)
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
                
                # Then process conditional (with prompt)
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
                
                # Convert predictions to match latents shape
                noise_pred_uncond = self._convert_single_noise_pred(noise_pred_uncond, latents)
                noise_pred_text = self._convert_single_noise_pred(noise_pred_text, latents)
                
                # Perform guidance (keep on GPU)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Compute previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, timestep, latents)
                
                # If step returns a dict, extract the sample
                if isinstance(latents, dict):
                    latents = latents["prev_sample"]
            
            # Scale and decode the image latents
            latents = latents * 0.18215
            image = self.model.decode(latents)
            
            # Post-process image (keep on GPU until final conversion)
            image = (image / 2 + 0.5)  # Normalize to [0, 1]
            image = tf.clip_by_value(image, 0, 1)  # Ensure values are in [0, 1]
            image = tf.cast(image * 255, tf.uint8)  # Scale to [0, 255] and convert to uint8
            
            # Final conversion to CPU for PIL Image creation
            image_np = image[0].numpy()
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_np)
            
            return pil_image
            
    def _convert_single_noise_pred(self, noise_pred, latents):
        """Convert a single noise prediction to match latents shape."""
        # Debug print shapes
        print(f"Single noise_pred shape: {noise_pred.shape}")
        print(f"Target latents shape: {latents.shape}")
        
        # If noise_pred is from transformer output (1, 78, 3072)
        if len(noise_pred.shape) == 3 and noise_pred.shape[-1] == 3072:
            # Reduce sequence length and project to latent space
            noise_pred = tf.reduce_mean(noise_pred, axis=1)
            noise_pred = tf.reshape(
                noise_pred, 
                (latents.shape[0], latents.shape[1], latents.shape[2], -1)
            )
            
            # Ensure last dimension matches latents
            if noise_pred.shape[-1] != latents.shape[-1]:
                noise_pred = tf.image.resize(
                    noise_pred, 
                    (latents.shape[1], latents.shape[2]), 
                    method=tf.image.ResizeMethod.BILINEAR
                )
        
        # If shape still doesn't match, use resize
        if noise_pred.shape != latents.shape:
            noise_pred = tf.image.resize(
                noise_pred, 
                (latents.shape[1], latents.shape[2]), 
                method=tf.image.ResizeMethod.BILINEAR
            )
        
        # Ensure the last dimension matches
        if noise_pred.shape[-1] != latents.shape[-1]:
            noise_pred = noise_pred[..., :latents.shape[-1]]
        
        # Print final shapes for debugging
        print(f"Converted noise_pred shape: {noise_pred.shape}")
        
        return noise_pred

    def decode_latents(self, latents):
        """Decode latents to image using GPU."""
        with tf.device(self.device):
            # Scale latents
            latents = latents * 0.18215
            
            # Decode
            image = self.model.decode(latents)
            
            # Post-process image
            image = (image / 2 + 0.5)  # Normalize to [0, 1]
            image = tf.clip_by_value(image, 0, 1)  # Ensure values are in [0, 1]
            image = tf.cast(image * 255, tf.uint8)  # Scale to [0, 255] and convert to uint8
            
            # Convert to numpy array
            image_np = image[0].numpy()  # Remove batch dimension
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_np)
            
            return pil_image
            
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
        
        return cls(model, scheduler, processor)
