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
        self.device = device
        
        # Track model state
        self.model_cpu_offload = False
        self._model_on_cpu = False
        self._vae_on_cpu = False
        
        # Set mixed precision policy
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        if device is None:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Memory growth needs to be the same across GPUs
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    self.device = '/GPU:0'
                except RuntimeError as e:
                    print(e)
            else:
                print("No GPU found, using CPU instead. This may take a long time!")
                self.device = '/CPU:0'
                
        # Set models to eval mode
        self.model.trainable = False
        self.vae.trainable = False
        
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
        
    def enable_model_cpu_offload(self):
        """Move model weights to CPU to save GPU memory."""
        if not self._model_on_cpu:
            with tf.device('/CPU:0'):
                self.model.save_weights('model_weights_temp')
                self.vae.save_weights('vae_weights_temp')
            self._model_on_cpu = True
            self._vae_on_cpu = True
            tf.keras.backend.clear_session()
            gc.collect()
            
    def disable_model_cpu_offload(self):
        """Move model weights back to device."""
        if self._model_on_cpu:
            with tf.device(self.device):
                self.model.load_weights('model_weights_temp')
                self.vae.load_weights('vae_weights_temp')
            self._model_on_cpu = False
            self._vae_on_cpu = False
            # Clean up temp files
            if os.path.exists('model_weights_temp.index'):
                os.remove('model_weights_temp.index')
            if os.path.exists('vae_weights_temp.index'):
                os.remove('vae_weights_temp.index')
                
    def to(self, device):
        """Move models to specified device."""
        self.device = device
        with tf.device(device):
            if not self._model_on_cpu:
                self.model.set_weights(self.model.get_weights())
            if not self._vae_on_cpu:
                self.vae.set_weights(self.vae.get_weights())
                
    def move_to_device(self, data):
        """Move data to current device."""
        if isinstance(data, list):
            return [self._move_tensor(x) for x in data]
        return self._move_tensor(data)
    
    def _move_tensor(self, tensor):
        """Helper to move a single tensor to device."""
        if self._model_on_cpu:
            return tf.identity(tensor)
        with tf.device(self.device):
            return tf.identity(tensor)
            
    def _cleanup_memory(self):
        """Cleanup GPU memory."""
        tf.keras.backend.clear_session()
        gc.collect()
        
    def _process_batch(
        self,
        prompt,
        input_images=None,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=3,
        use_img_guidance=True,
        img_guidance_scale=1.6,
        max_input_image_size=1024,
        use_kv_cache=True,
        offload_kv_cache=True,
        use_input_image_size_as_output=False,
    ):
        # Process text and images
        model_inputs = self.processor(
            prompt,
            input_images=input_images,
            max_input_image_size=max_input_image_size
        )

        # Move inputs to device and cast to dtype
        model_inputs = {
            k: tf.cast(self.move_to_device(v), tf.bfloat16)
            for k, v in model_inputs.items()
        }

        # Initialize latents
        latents = tf.random.normal(
            [len(prompt), self.vae.config.latent_channels, height // 8, width // 8],
            dtype=tf.bfloat16
        )

        # Setup scheduler
        scheduler = OmniGenScheduler(num_inference_steps)

        # Run inference on batch
        images = self.model.memory_efficient_forward(
            latents,
            model_inputs=model_inputs,
            scheduler=scheduler,
            guidance_scale=guidance_scale,
            use_img_guidance=use_img_guidance,
            img_guidance_scale=img_guidance_scale,
            dtype=tf.bfloat16
        )

        # Decode images
        images = self.vae.decode(images / self.vae.config.scaling_factor).sample
        images = (images + 1) / 2
        images = tf.clip_by_value(images, 0, 1)
        images = tf.transpose(images, [0, 2, 3, 1])

        # Convert to output format
        images = [Image.fromarray((img.numpy() * 255).astype(np.uint8)) for img in images]

        return images

    def __call__(
        self,
        prompt,
        input_images=None,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=3,
        use_img_guidance=True,
        img_guidance_scale=1.6,
        max_input_image_size=1024,
        separate_cfg_infer=True,  # Enable by default to save memory
        offload_model=True,      # Enable by default
        use_kv_cache=True,
        offload_kv_cache=True,
        use_input_image_size_as_output=False,
        output_type="pil",
    ):
        # Enable CPU offload if requested
        if offload_model:
            self.enable_model_cpu_offload()
            
        try:
            # Process in smaller batches if using separate CFG inference
            if separate_cfg_infer:
                batch_size = 1
            else:
                batch_size = 2 if guidance_scale > 1 else 1
                
            # Process images in chunks to save memory
            results = []
            for i in range(0, len(prompt) if isinstance(prompt, list) else 1, batch_size):
                batch_prompt = prompt[i:i+batch_size] if isinstance(prompt, list) else prompt
                batch_images = input_images[i:i+batch_size] if input_images is not None else None
                
                # Move models to device for this batch
                if offload_model:
                    self.disable_model_cpu_offload()
                    
                # Process batch
                result = self._process_batch(
                    batch_prompt,
                    batch_images,
                    height,
                    width,
                    num_inference_steps,
                    guidance_scale,
                    use_img_guidance,
                    img_guidance_scale,
                    max_input_image_size,
                    use_kv_cache,
                    offload_kv_cache,
                    use_input_image_size_as_output,
                )
                results.extend(result)
                
                # Move models back to CPU after batch
                if offload_model:
                    self.enable_model_cpu_offload()
                    
                # Cleanup after each batch
                self._cleanup_memory()
                
            return {"images": results}
            
        finally:
            # Ensure models are moved back to device
            if offload_model:
                self.disable_model_cpu_offload()
                
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
