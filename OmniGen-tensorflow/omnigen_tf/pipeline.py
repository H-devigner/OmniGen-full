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
        
        return cls(model, scheduler, processor)

    def __call__(
        self,
        prompt,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
    ):
        """Generate image from text prompt using GPU acceleration."""
        # Process text prompt
        text_inputs = self.processor(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="tf"
        )
        
        # Create latents
        latents_shape = (1, height // 8, width // 8, 4)
        latents = tf.random.normal(latents_shape, dtype=tf.float32)
        
        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        
        # Prepare text embeddings
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            max_length = text_inputs["input_ids"].shape[1]
            uncond_input = self.processor(
                [""] * 2,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="tf",
            )
            uncond_embeddings = uncond_input["input_ids"]
            text_embeddings = tf.concat([uncond_embeddings, text_inputs["input_ids"]], axis=0)
            attention_mask = tf.concat([uncond_input["attention_mask"], text_inputs["attention_mask"]], axis=0)
        else:
            text_embeddings = text_inputs["input_ids"]
            attention_mask = text_inputs["attention_mask"]
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            # Expand latents for classifier free guidance
            latent_model_input = tf.repeat(latents, 2, axis=0) if do_classifier_free_guidance else latents
            
            # Predict noise residual
            with tf.device(self.device):
                noise_pred = self.model(
                    latent_model_input,
                    t,
                    input_ids=text_embeddings,
                    attention_mask=attention_mask,
                    training=False
                )
            
            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = tf.split(noise_pred, 2, axis=0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
        # Scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        
        # Convert to image
        image = self.decode_latents(latents)
        
        return image

    def decode_latents(self, latents):
        """Decode latents to image."""
        with tf.device(self.device):
            # Ensure proper shape and scaling
            image = tf.transpose(latents, [0, 3, 1, 2])  # NHWC -> NCHW
            image = image / 2 + 0.5  # Scale to [0, 1]
            image = tf.clip_by_value(image, 0, 1)
            image = tf.cast(image * 255, tf.uint8)
            image = tf.transpose(image, [0, 2, 3, 1])  # NCHW -> NHWC
            
            # Convert to PIL Image
            image = image[0].numpy()  # Remove batch dimension
            return Image.fromarray(image)

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
