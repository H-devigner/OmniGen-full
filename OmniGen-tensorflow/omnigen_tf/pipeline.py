"""TensorFlow implementation of OmniGen pipeline."""

import os
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import gc

import numpy as np
import tensorflow as tf
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from safetensors import safe_open

from .model import OmniGen
from .processor import OmniGenProcessor
from .scheduler import OmniGenScheduler
from .utils import logging

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
        model: OmniGen,
        processor: OmniGenProcessor,
        scheduler: OmniGenScheduler,
        tokenizer: Optional[Any] = None,
        device: str = None,
        **kwargs
    ):
        """Initialize OmniGen pipeline.
        
        Args:
            model: OmniGen model
            processor: OmniGen processor
            scheduler: OmniGen scheduler
            tokenizer: Optional tokenizer
            device: Device to use (e.g. 'CPU', 'GPU')
            **kwargs: Additional arguments
        """
        self.model = model
        self.processor = processor
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        
        # Configure device
        self.device = device or ("GPU:0" if tf.config.list_physical_devices('GPU') else "CPU:0")
        print(f"Pipeline using device: {self.device}")
        
        # Enable mixed precision
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Configure GPU memory growth
        for gpu in tf.config.list_physical_devices('GPU'):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except:
                pass

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load pipeline from pretrained model.
        
        Args:
            pretrained_model_name_or_path: Path or name of pretrained model
            **kwargs: Additional arguments
            
        Returns:
            OmniGenPipeline: Loaded pipeline instance
        """
        try:
            # Configure device
            device = kwargs.pop('device', None)
            if device is None:
                device = "GPU:0" if tf.config.list_physical_devices('GPU') else "CPU:0"
            
            print(f"Loading pipeline on device: {device}")
            
            # Download model if needed
            if not os.path.isdir(pretrained_model_name_or_path):
                pretrained_model_name_or_path = snapshot_download(pretrained_model_name_or_path)
            
            # Load components
            model = OmniGen.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="model",
                device=device
            )
            
            processor = OmniGenProcessor.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="processor"
            )
            
            scheduler = OmniGenScheduler.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="scheduler"
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="tokenizer",
                use_fast=True
            )
            
            # Create pipeline
            pipeline = cls(
                model=model,
                processor=processor,
                scheduler=scheduler,
                tokenizer=tokenizer,
                device=device,
                **kwargs
            )
            
            return pipeline
            
        except Exception as e:
            logger.error(f"Error loading pipeline: {str(e)}")
            tf.keras.backend.clear_session()
            raise
            
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Any] = None,
        latents: Optional[tf.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, tf.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        """Generate images from text prompt.
        
        Args:
            prompt: The prompt to generate images from
            height: Height of output images
            width: Width of output images
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            negative_prompt: Optional negative prompt
            num_images_per_prompt: Number of images to generate per prompt
            eta: Eta parameter for scheduler
            generator: Random number generator
            latents: Pre-generated latents
            output_type: Output format ("pil" or "np")
            return_dict: Whether to return dict
            callback: Optional callback function
            callback_steps: Number of steps between callbacks
            **kwargs: Additional arguments
            
        Returns:
            Generated images
        """
        
        # Input validation
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"Height and width must be divisible by 8 but are {height} and {width}")
            
        if isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        else:
            batch_size = len(prompt)
            
        if negative_prompt is None:
            negative_prompt = [""] * batch_size
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size
            
        # Process inputs
        with tf.device(self.device):
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="tf",
            )
            
            text_input_ids = text_inputs.input_ids
            
            if negative_prompt is not None:
                uncond_inputs = self.tokenizer(
                    negative_prompt,
                    padding="max_length", 
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="tf",
                )
                uncond_input_ids = uncond_inputs.input_ids
            else:
                uncond_input_ids = None
                
            # Set timesteps
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps
            
            # Generate initial latents
            latents_shape = (batch_size * num_images_per_prompt, self.model.in_channels, height // 8, width // 8)
            if latents is None:
                latents = tf.random.normal(latents_shape, dtype=tf.float32)
                
            latents = latents * self.scheduler.init_noise_sigma
            
            # Prepare extra step kwargs
            extra_step_kwargs = {}
            if "eta" in inspect.signature(self.scheduler.step).parameters:
                extra_step_kwargs["eta"] = eta
                
            # Denoising loop
            for i, t in enumerate(timesteps):
                # Expand latents for classifier-free guidance
                latent_model_input = tf.concat([latents] * 2) if guidance_scale > 1.0 else latents
                
                # Predict noise residual
                noise_pred = self.model(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_input_ids,
                    uncond_encoder_hidden_states=uncond_input_ids,
                    return_dict=False,
                )[0]
                
                # Perform guidance
                if guidance_scale > 1.0:
                    noise_pred_uncond, noise_pred_text = tf.split(noise_pred, 2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                # Compute previous noisy sample
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                
                # Call callback if needed
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
                    
            # Decode latents
            images = self.model.decode_latents(latents)
            
            # Convert to PIL
            if output_type == "pil":
                images = self.numpy_to_pil(images)
                
            # Offload last model to CPU
            if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                self.final_offload_hook.offload()
                
            if not return_dict:
                return (images,)
                
            return {"images": images}
            
    def numpy_to_pil(self, images):
        """Convert numpy images to PIL images."""
        images = (images * 255).round().astype("uint8")
        if images.ndim == 3:
            images = images[None, ...]
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

# pipeline = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")
# image = pipeline(
#     prompt="your prompt here",
#     height=512,
#     width=512,
#     num_inference_steps=50,
#     guidance_scale=7.5
# )
