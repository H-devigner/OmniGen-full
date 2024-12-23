"""OmniGen TensorFlow Pipeline

This module provides the TensorFlow implementation of the OmniGen pipeline,
matching the PyTorch version's functionality while leveraging TensorFlow-specific optimizations.
"""

import os
import gc
import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Union
from PIL import Image
from safetensors import safe_open
from huggingface_hub import snapshot_download
from diffusers import AutoencoderKL
from diffusers.utils import logging

from .processor import OmniGenProcessor
from .scheduler import OmniGenScheduler
from .model import OmniGen

logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from omnigen_tf import OmniGenPipeline
        >>> pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")
        >>> prompt = "A woman holds a bouquet of flowers and faces the camera"
        >>> image = pipe(
        ...     prompt,
        ...     guidance_scale=2.5,
        ...     num_inference_steps=50,
        ... ).images[0]
        >>> image.save("t2i.png")
        ```
"""


class OmniGenPipeline:
    """Pipeline for text-to-image generation using OmniGen."""
    
    def __init__(
        self,
        vae: AutoencoderKL,
        model: OmniGen,
        processor: OmniGenProcessor,
        device: str = None,
    ):
        """Initialize OmniGen pipeline.
        
        Args:
            vae: VAE model for encoding/decoding images
            model: OmniGen model
            processor: Text and image processor
            device: Device to place models on ('CPU', 'GPU', or None for auto-detect)
        """
        self.vae = vae
        self.model = model
        self.processor = processor
        
        # Set device
        if device is None:
            if tf.config.list_physical_devices('GPU'):
                device = '/GPU:0'
            else:
                print("No GPU found, using CPU. This may be slow!")
                device = '/CPU:0'
        self.device = device
        
        # Enable mixed precision
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        # Memory optimization flags
        self._model_on_cpu = False
        self._vae_on_cpu = False
        
    def _move_to_device(self, model, device):
        """Move model to specified device."""
        with tf.device(device):
            for layer in model.layers:
                for weight in layer.weights:
                    weight.assign(tf.identity(weight))
                    
    def enable_cpu_offload(self):
        """Move models to CPU to save memory."""
        if not self._model_on_cpu:
            self._move_to_device(self.model, '/CPU:0')
            self._model_on_cpu = True
            
        if not self._vae_on_cpu:
            self._move_to_device(self.vae, '/CPU:0')
            self._vae_on_cpu = True
            
    def disable_cpu_offload(self):
        """Move models back to GPU."""
        if self._model_on_cpu and tf.config.list_physical_devices('GPU'):
            self._move_to_device(self.model, '/GPU:0')
            self._model_on_cpu = False
            
        if self._vae_on_cpu and tf.config.list_physical_devices('GPU'):
            self._move_to_device(self.vae, '/GPU:0')
            self._vae_on_cpu = False
            
    @tf.function(jit_compile=True)
    def _process_batch(self, prompt_embeds, timesteps):
        """Process a single batch with XLA optimization."""
        # Move to device
        with tf.device(self.device):
            # Generate latents
            latents = self.model(
                prompt_embeds,
                timesteps,
                training=False
            )
            
            # Decode latents
            images = self.vae.decode(latents).sample
            
            return images
            
    def __call__(self, prompt, **kwargs):
        """Generate images with memory optimization."""
        try:
            # Enable CPU offload
            self.enable_cpu_offload()
            
            # Process prompt
            text_inputs = self.processor(prompt)
            
            # Get batch size and process in chunks
            batch_size = text_inputs["input_ids"].shape[0]
            max_batch_size = 4  # Adjust based on memory
            
            all_images = []
            for i in range(0, batch_size, max_batch_size):
                end_idx = min(i + max_batch_size, batch_size)
                
                # Get batch slice
                batch_inputs = {
                    k: v[i:end_idx] for k, v in text_inputs.items()
                }
                
                # Process batch
                images = self._process_batch(
                    batch_inputs["input_ids"],
                    batch_inputs["timesteps"]
                )
                
                all_images.append(images)
                
                # Force cleanup
                tf.keras.backend.clear_session()
                
            # Combine results
            images = tf.concat(all_images, axis=0)
            
            return images
            
        finally:
            # Disable CPU offload
            self.disable_cpu_offload()
            
    @classmethod
    def from_pretrained(cls, model_name, vae_path=None, device=None):
        """Load pipeline from pretrained models.
        
        Args:
            model_name: Name or path of the pretrained model
            vae_path: Optional path to VAE model
            device: Device to place models on ('CPU', 'GPU', or None for auto-detect)
            
        Returns:
            Initialized pipeline
        """
        if not os.path.exists(model_name):
            print(f"Downloading model from {model_name}...")
            model_name = snapshot_download(model_name)
            print(f"Downloaded model to {model_name}")
            
        # Load models with specified device
        model = OmniGen.from_pretrained(
            model_name, 
            device=device,
        )
        processor = OmniGenProcessor.from_pretrained(model_name)
        
        # Load or download VAE
        if vae_path is None:
            vae_path = "stabilityai/sd-vae-ft-mse"
            
        if not os.path.exists(vae_path):
            print(f"Downloading VAE from {vae_path}...")
            vae_path = snapshot_download(vae_path)
            print(f"Downloaded VAE to {vae_path}")
            
        vae = AutoencoderKL.from_pretrained(vae_path)
        
        return cls(
            model=model,
            processor=processor,
            vae=vae,
            device=device
        )
        
    def vae_encode(self, x: tf.Tensor, dtype: tf.dtypes.DType) -> tf.Tensor:
        """Encode images using VAE.
        
        Args:
            x: Input tensor
            dtype: Output data type
            
        Returns:
            Encoded tensor
        """
        with tf.device(self.device):
            if hasattr(self.vae.config, 'shift_factor'):
                x = self.vae.encode(x).sample()
                x = (x - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            else:
                x = self.vae.encode(x).sample() * self.vae.config.scaling_factor
            return tf.cast(x, dtype)

    def merge_lora(self, lora_path: str):
        """Merge LoRA weights into the base model.
        
        This method loads a LoRA checkpoint and merges its weights into the base model,
        effectively applying the fine-tuned adaptations permanently.
        
        Args:
            lora_path: Path to the LoRA checkpoint
        """
        # Load LoRA weights
        if not os.path.exists(lora_path):
            lora_path = snapshot_download(
                repo_id=lora_path,
                cache_dir=os.getenv('HF_HUB_CACHE'),
                allow_patterns="*.safetensors"
            )
            
        # Let the model handle LoRA merging using PeftAdapterMixin
        self.model.merge_adapter(lora_path)
        print(f"Successfully merged LoRA weights from {lora_path}")
