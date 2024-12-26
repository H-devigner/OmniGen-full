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
            attention_mask = inputs.get("attention_mask", None)
            
            # Initialize latents on GPU with smaller batch size
            latents_shape = (1, height // 8, width // 8, 4)
            latents = tf.random.normal(latents_shape, dtype=tf.float16)
            
            # Set timesteps
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps
            
            # Prepare extra kwargs for the scheduler step
            extra_step_kwargs = {}
            
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
                
                # Perform guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Free memory
                tf.keras.backend.clear_session()
                
                # Compute previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, timestep, latents)
                
                # If step returns a dict, extract the sample
                if isinstance(latents, dict):
                    latents = latents["prev_sample"]
            
            # Scale and decode the image latents
            latents = latents * 0.18215
            image = self.model.decode(latents)
            
            # Post-process image
            image = (image / 2 + 0.5)  # Normalize to [0, 1]
            image = tf.clip_by_value(image, 0, 1)  # Ensure values are in [0, 1]
            
            # Convert to numpy and scale to [0, 255]
            image = image.numpy()
            image = (image * 255).astype(np.uint8)
            
            # Convert to PIL Image
            image = Image.fromarray(image[0])
            
            return image
            
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
            
            # Convert to numpy and scale to [0, 255]
            image = image.numpy()
            image = (image * 255).astype(np.uint8)
            
            # Convert to PIL Image
            image = Image.fromarray(image[0])
            
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
