"""OmniGen TensorFlow Pipeline

This module provides the TensorFlow implementation of the OmniGen pipeline,
matching the PyTorch version's functionality while leveraging TensorFlow-specific optimizations.
"""

import os
import gc
from typing import Any, Callable, Dict, List, Optional, Union

import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from diffusers.models import AutoencoderKL
from diffusers.utils import logging

from .model import OmniGen
from .processor import OmniGenProcessor
from .scheduler import OmniGenScheduler

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
        mixed_precision: bool = True
    ):
        """Initialize OmniGen pipeline.
        
        Args:
            vae: VAE model for encoding/decoding images
            model: OmniGen model
            processor: Text and image processor
            device: Device to place models on ('CPU', 'GPU', or None for auto-detect)
            mixed_precision: Whether to use mixed precision
        """
        # Set up device strategy
        if device is None:
            if len(tf.config.list_physical_devices('GPU')) > 0:
                device = 'GPU'
                print("Using GPU for inference")
            else:
                device = 'CPU'
                print("No GPU detected, using CPU. This may be slow!")
                
        self.device = device
        
        # Enable mixed precision if requested and on GPU
        if mixed_precision and device == 'GPU':
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Using mixed precision")
            
        with tf.device(f'/{device}:0'):
            self.vae = vae
            self.model = model
            self.processor = processor
            
            # Set models to eval mode
            self.model.trainable = False
            self.vae.trainable = False
            
    @classmethod
    def from_pretrained(cls, model_name: str, vae_path: str = None, device: str = None, mixed_precision: bool = True):
        """Load pipeline from pretrained model.
        
        Args:
            model_name: Name or path of pretrained model
            vae_path: Optional path to VAE model
            device: Device to place models on ('CPU', 'GPU', or None for auto-detect)
            mixed_precision: Whether to use mixed precision
            
        Returns:
            OmniGenPipeline instance
        """
        if not os.path.exists(model_name):
            print(f"Downloading model from {model_name}")
            cache_folder = os.getenv('HF_HUB_CACHE')
            model_name = snapshot_download(
                repo_id=model_name,
                cache_dir=cache_folder,
                ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'model.pt']
            )
            print(f"Downloaded model to {model_name}")
            
        # Load models with specified device and precision settings
        model = OmniGen.from_pretrained(model_name, device=device, mixed_precision=mixed_precision)
        processor = OmniGenProcessor.from_pretrained(model_name)
        
        # Load or download VAE
        if os.path.exists(os.path.join(model_name, "vae")):
            vae = AutoencoderKL.from_pretrained(os.path.join(model_name, "vae"))
        elif vae_path is not None:
            vae = AutoencoderKL.from_pretrained(vae_path)
        else:
            print("No VAE found, downloading stabilityai/sdxl-vae")
            vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
            
        return cls(vae, model, processor, device=device, mixed_precision=mixed_precision)
        
    def __call__(
        self,
        prompt: Union[str, List[str]],
        input_images: Union[List[str], List[List[str]]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        use_img_guidance: bool = True,
        img_guidance_scale: float = 1.6,
        max_input_image_size: int = 1024,
        separate_cfg_infer: bool = True,
        offload_model: bool = False,
        use_kv_cache: bool = True,
        offload_kv_cache: bool = True,
        use_input_image_size_as_output: bool = False,
        dtype: tf.dtypes.DType = tf.float32,
        seed: int = None,
        output_type: str = "pil",
    ) -> Dict[str, Any]:
        """Generate images from text prompt.
        
        Args:
            prompt: Text prompt(s)
            input_images: Optional input image paths
            height: Output height
            width: Output width
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            use_img_guidance: Whether to use image guidance
            img_guidance_scale: Image guidance scale
            max_input_image_size: Maximum input image size
            separate_cfg_infer: Whether to do separate inference for different guidance
            offload_model: Whether to offload model to CPU
            use_kv_cache: Whether to use key-value cache
            offload_kv_cache: Whether to offload cache to CPU
            use_input_image_size_as_output: Whether to use input size as output
            dtype: Data type
            seed: Random seed
            output_type: Output type ("pil" or "tensor")
            
        Returns:
            Dictionary with generated images
        """
        try:
            # Clear any existing cached tensors
            tf.keras.backend.clear_session()
            gc.collect()
            
            with tf.device(f'/{self.device}:0'):
                # Set random seed if provided
                if seed is not None:
                    tf.random.set_seed(seed)

                # Enable CPU offloading if requested
                if offload_model:
                    self.enable_model_cpu_offload()

                # Process inputs
                if not isinstance(prompt, list):
                    prompt = [prompt]
                
                batch_size = len(prompt)

                # Process text and images
                model_inputs = self.processor(
                    prompt, 
                    input_images=input_images,
                    max_input_image_size=max_input_image_size
                )

                # Move inputs to device
                model_inputs = {k: self.move_to_device(v) for k, v in model_inputs.items()}

                # Initialize latents
                latents = tf.random.normal(
                    [batch_size, self.vae.config.latent_channels, height // 8, width // 8],
                    dtype=dtype
                )

                # Setup scheduler
                scheduler = OmniGenScheduler(num_inference_steps)

                # Run inference
                images = self.model.generate(
                    latents,
                    model_inputs,
                    scheduler,
                    guidance_scale=guidance_scale,
                    use_img_guidance=use_img_guidance,
                    img_guidance_scale=img_guidance_scale,
                    separate_cfg_infer=separate_cfg_infer,
                    use_kv_cache=use_kv_cache,
                    offload_kv_cache=offload_kv_cache,
                    dtype=dtype
                )

                # Decode images
                images = self.vae.decode(images / self.vae.config.scaling_factor).sample
                images = (images + 1) / 2
                images = tf.clip_by_value(images, 0, 1)
                images = tf.transpose(images, [0, 2, 3, 1])

            # Convert to output format
            if output_type == "pil":
                images = [Image.fromarray((img.numpy() * 255).astype(np.uint8)) for img in images]

            # Cleanup
            if offload_model:
                self.disable_model_cpu_offload()

            # Clean up after inference
            tf.keras.backend.clear_session()
            gc.collect()
            
            return {"images": images}
            
        except Exception as e:
            print(f"Error in pipeline: {str(e)}")
            raise

    def to(self, device: str):
        """Move pipeline to specified device."""
        with tf.device(device):
            self.model = tf.identity(self.model)
            self.vae = tf.identity(self.vae)
        self.device = device

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

    def move_to_device(self, data: Union[tf.Tensor, List[tf.Tensor]]) -> Union[tf.Tensor, List[tf.Tensor]]:
        """Move data to device."""
        with tf.device(self.device):
            if isinstance(data, list):
                return [tf.identity(x) for x in data]
            return tf.identity(data)

    def enable_model_cpu_offload(self):
        """Enable CPU offloading for memory savings."""
        self.model_cpu_offload = True
        with tf.device("CPU:0"):
            self.model = tf.identity(self.model)
            self.vae = tf.identity(self.vae)
        tf.keras.backend.clear_session()
        gc.collect()

    def disable_model_cpu_offload(self):
        """Disable CPU offloading."""
        self.model_cpu_offload = False
        with tf.device(self.device):
            self.model = tf.identity(self.model)
            self.vae = tf.identity(self.vae)
