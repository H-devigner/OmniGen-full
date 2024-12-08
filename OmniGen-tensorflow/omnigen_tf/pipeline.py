"""OmniGen pipeline for text-to-image generation."""
from __future__ import annotations

import os
import time
from typing import Optional, List, Union, Dict, Any
import tensorflow as tf
import torch
from PIL import Image
import numpy as np
from huggingface_hub import snapshot_download

from .model import OmniGenTF
from .processor import OmniGenProcessor
from .vae import AutoencoderKL

class OmniGenPipeline:
    """Pipeline for text-to-image generation using OmniGen."""
    
    def __init__(
        self,
        model: OmniGenTF,
        processor: OmniGenProcessor,
        vae: AutoencoderKL,
        device: str = "gpu",
        **kwargs
    ):
        """Initialize pipeline."""
        self.model = model
        self.processor = processor
        self.vae = vae
        self.device = device
        
        # Set up device strategy
        if device == "gpu" and tf.config.list_physical_devices('GPU'):
            self.strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
        else:
            self.strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        vae_path: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: int = 1,
        mixed_precision: bool = False,
        **kwargs
    ) -> "OmniGenPipeline":
        """Load pipeline from pretrained model."""
        for attempt in range(max_retries):
            try:
                # Download model if needed
                if not os.path.isdir(model_name):
                    model_path = snapshot_download(
                        model_name,
                        allow_patterns=["*.safetensors", "*.json", "*.bin"]
                    )
                else:
                    model_path = model_name
                    
                # Load model components
                model = OmniGenTF.from_pretrained(
                    model_path,
                    use_mixed_precision=mixed_precision,
                    **kwargs
                )
                
                processor = OmniGenProcessor.from_pretrained(model_path)
                
                # Load VAE
                if vae_path is None:
                    vae_path = model_path
                vae = AutoencoderKL.from_pretrained(vae_path)
                
                return cls(
                    model=model,
                    processor=processor,
                    vae=vae,
                    device="gpu" if tf.config.list_physical_devices('GPU') else "cpu"
                )
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        **kwargs
    ) -> List[Image.Image]:
        """Generate images from text prompt."""
        # Process inputs
        inputs = self.processor(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            **kwargs
        )
        
        # Generate images
        with self.strategy.scope():
            # Generate latents
            latents = self.model(
                inputs["input_ids"],
                inputs["timesteps"],
                attention_mask=inputs["attention_mask"],
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
            
            # Scale latents
            latents = latents * self.vae.scaling_factor
            
            # Decode latents
            images = self.vae.decode_from_latents(latents)
            
        # Post-process images
        images = self.processor.postprocess_images(images)
        
        return images
