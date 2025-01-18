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
                print("Using GPU for inference")
                # Enable memory growth to avoid OOM
                for gpu in tf.config.list_physical_devices('GPU'):
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except:
                        pass
                # Use float32 for better precision during generation
                tf.keras.mixed_precision.set_global_policy('float32')
            else:
                device = '/CPU:0'
                print("Using CPU for inference")
        self.device = device
        
        # Create device strategy
        self.strategy = tf.distribute.OneDeviceStrategy(device)

    def denoise_step(self, latents, timestep, noise_pred, num_inference_steps):
        """Perform a single denoising step with regularization."""
        # Add small amount of random noise for regularization
        # Scale noise based on timestep progress
        progress = 1.0 - (float(timestep) / num_inference_steps)
        noise_scale = 0.01 * progress
        reg_noise = tf.random.normal(latents.shape, stddev=noise_scale, dtype=latents.dtype)
        latents = latents + reg_noise
        
        # Get scheduler step
        latents = self.scheduler.step(noise_pred, timestep, latents)
        if isinstance(latents, dict):
            latents = latents["prev_sample"]
            
        return tf.cast(latents, tf.float32)

    def process_noise_pred(self, noise_pred_uncond, noise_pred_text, guidance_scale):
        """Process noise predictions with clamped guidance."""
        # Calculate guidance difference
        guidance_diff = noise_pred_text - noise_pred_uncond
        
        # Clamp the guidance influence
        clamped_guidance = tf.clip_by_value(
            guidance_scale * guidance_diff,
            -3.0, 3.0  # Limit extreme values
        )
        
        # Combine predictions
        return noise_pred_uncond + clamped_guidance

    def post_process_image(self, image):
        """Apply post-processing to reduce noise and enhance quality."""
        # Convert to float32 for processing
        image = tf.cast(image, tf.float32)
        
        # Apply mild Gaussian blur to reduce noise
        kernel_size = 3
        sigma = 0.5
        image = tf.image.gaussian_blur(image, [kernel_size, kernel_size], sigma)
        
        # Enhance contrast slightly
        image = tf.image.adjust_contrast(image, 1.1)
        
        # Adjust saturation
        image = tf.image.adjust_saturation(image, 1.1)
        
        # Ensure values are in valid range
        image = tf.clip_by_value(image, 0.0, 255.0)
        
        return tf.cast(image, tf.uint8)
        
    def __call__(
        self,
        prompt,
        height=512,
        width=512,
        num_inference_steps=100,  # Increased for better quality
        guidance_scale=4.0,  # Reduced for more stable results
        use_post_processing=True  # Enable post-processing by default
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
            latents = tf.random.normal(latents_shape, dtype=tf.float32)  # Use float32
            
            # Set timesteps
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = tf.cast(self.scheduler.timesteps, tf.int32)
            
            # Denoising loop with improved processing
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
                noise_pred_uncond = tf.cast(self._convert_single_noise_pred(noise_pred_uncond, latents), tf.float32)
                noise_pred_text = tf.cast(self._convert_single_noise_pred(noise_pred_text, latents), tf.float32)
                
                # Apply improved noise prediction processing
                noise_pred = self.process_noise_pred(noise_pred_uncond, noise_pred_text, guidance_scale)
                
                # Improved denoising step
                latents = self.denoise_step(latents, timestep, noise_pred, num_inference_steps)
            
            # Scale latents
            latents = tf.cast(latents * 0.18215, tf.float32)
            image = self.model.decode(latents)
            
            # Resize if needed
            if image.shape[1:3] != (height, width):
                image = tf.image.resize(
                    image,
                    (height, width),
                    method=tf.image.ResizeMethod.BICUBIC
                )
            
            # Post-process image
            image = (image / 2 + 0.5) * 255.0
            
            if use_post_processing:
                image = self.post_process_image(image)
            else:
                image = tf.cast(tf.clip_by_value(image, 0.0, 255.0), tf.uint8)
            
            # Convert to PIL Image
            image_np = image[0].numpy()
            pil_image = Image.fromarray(image_np)
            
            return pil_image
            
    def _convert_single_noise_pred(self, noise_pred, latents):
        """Convert a single noise prediction to match latents shape."""
        # Debug print shapes
        print(f"Single noise_pred shape: {noise_pred.shape}")
        print(f"Target latents shape: {latents.shape}")
        
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
        
        # If shape still doesn't match the target latents shape
        if noise_pred.shape[1:3] != latents.shape[1:3]:
            noise_pred = tf.image.resize(
                noise_pred,
                (latents.shape[1], latents.shape[2]),
                method=tf.image.ResizeMethod.BICUBIC
            )
        
        # Ensure the last dimension matches
        if noise_pred.shape[-1] != latents.shape[-1]:
            noise_pred = noise_pred[..., :latents.shape[-1]]
        
        # Print final shape for debugging
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
