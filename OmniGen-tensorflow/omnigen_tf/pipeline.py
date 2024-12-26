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
            
            # Calculate position IDs
            seq_length = tf.shape(input_ids)[1]
            position_ids = tf.range(seq_length, dtype=tf.int32)[tf.newaxis, :]
            
            # Initialize latents on GPU with smaller batch size
            latent_height = height // 8
            latent_width = width // 8
            latents_shape = (1, latent_height, latent_width, 4)
            latents = tf.random.normal(latents_shape, dtype=tf.float16)
            
            # Set timesteps
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = tf.cast(self.scheduler.timesteps, tf.int32)
            
            # Prepare for classifier-free guidance
            unconditional_input_ids = tf.zeros_like(input_ids)
            unconditional_attention_mask = tf.zeros_like(attention_mask)
            
            # Denoising loop with memory-efficient processing
            for i, t in enumerate(timesteps):
                # Convert timestep to scalar
                timestep = tf.cast(t, tf.int32)
                timestep_tensor = tf.fill([1], timestep)
                
                # Prepare latent input
                latent_model_input = tf.concat([latents] * 2, axis=0)
                
                # Process unconditional and conditional in one pass
                noise_pred = self.model(
                    latent_model_input,
                    timestep=timestep_tensor,
                    input_ids=tf.concat([unconditional_input_ids, input_ids], axis=0),
                    attention_mask=tf.concat([unconditional_attention_mask, attention_mask], axis=0),
                    position_ids=tf.concat([position_ids, position_ids], axis=0),
                    training=False
                )
                
                # Split predictions
                noise_pred_uncond, noise_pred_text = tf.split(noise_pred, num_or_size_splits=2, axis=0)
                
                # Perform guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Compute previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, timestep, latents)
                
                # If step returns a dict, extract the sample
                if isinstance(latents, dict):
                    latents = latents["prev_sample"]
                
                # Ensure latents stay in float16
                latents = tf.cast(latents, tf.float16)
            
            # Scale and decode the image latents
            latents = tf.cast(latents * 0.18215, tf.float16)
            image = self.model.decode(latents)
            
            # Resize to requested dimensions if needed
            if image.shape[1:3] != (height, width):
                image = tf.image.resize(
                    image,
                    (height, width),
                    method=tf.image.ResizeMethod.BICUBIC
                )
            
            # Post-process image (keep on GPU until final conversion)
            image = (image / 2 + 0.5)  # Normalize to [0, 1]
            image = tf.clip_by_value(image, 0, 1)  # Ensure values are in [0, 1]
            image = tf.cast(image * 255, tf.uint8)  # Scale to [0, 255] and convert to uint8
            
            # Final conversion to CPU for PIL Image creation
            image_np = image[0].numpy()
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_np)
            
            # Print final image size for verification
            print(f"Final image size: {pil_image.size}")
            
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
