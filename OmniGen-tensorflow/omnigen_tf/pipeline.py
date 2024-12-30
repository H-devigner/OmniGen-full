"""OmniGen Pipeline for text-to-image generation."""

import os
import gc
import numpy as np
import tensorflow as tf
from PIL import Image
from typing import List, Optional, Union
from huggingface_hub import snapshot_download
import json
import torch

from omnigen_tf.model import OmniGen
from omnigen_tf.scheduler import OmniGenScheduler
from omnigen_tf.processor import OmniGenProcessor
from diffusers import AutoencoderKL

# Enable mixed precision globally
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(f"GPU memory growth setting failed: {e}")

class OmniGenPipeline:
    """OmniGen pipeline for text-to-image generation."""
    
    def __init__(self, vae=None, model=None, processor=None, device=None):
        """Initialize pipeline with memory optimizations.
        
        Args:
            vae: VAE model instance
            model: OmniGen model instance
            processor: OmniGenProcessor instance
            device: Device to use (CPU/GPU)
        """
        self.vae = vae
        self.model = model
        self.processor = processor
        self.scheduler = OmniGenScheduler()
        
        # Set device strategy
        if device is None:
            if tf.config.list_physical_devices('GPU'):
                device = '/GPU:0'
                print("Using GPU for inference")
            else:
                print("No GPU available, using CPU (this will be slow)")
                device = '/CPU:0'
        self.device = device
        
        # Initialize model in mixed precision
        if tf.config.list_physical_devices('GPU'):
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        self.model_cpu_offload = False
            
    @classmethod
    def from_pretrained(cls, model_name, vae_path=None):
        """Load pipeline from pretrained model with optimized memory usage."""
        if not os.path.exists(model_name):
            print(f"Model not found, downloading {model_name}...")
            cache_folder = os.getenv('HF_HUB_CACHE')
            try:
                model_name = snapshot_download(
                    repo_id=model_name,
                    cache_dir=cache_folder,
                    local_files_only=False,
                    resume_download=True,
                    ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5', 'model.pt']
                )
                print(f"Downloaded model to {model_name}")
            except Exception as e:
                print(f"Error downloading model: {str(e)}")
                raise
            
        # Load processor with optimized settings
        processor = OmniGenProcessor.from_pretrained(model_name)
        
        # Load model with optimized settings
        model = OmniGen.from_pretrained(model_name)
        
        # Load or download VAE
        if os.path.exists(os.path.join(model_name, "vae")):
            print("Loading VAE from model directory...")
            vae = AutoencoderKL.from_pretrained(
                os.path.join(model_name, "vae"),
                use_safetensors=True,
                low_cpu_mem_usage=True
            )
        elif vae_path is not None:
            print(f"Loading VAE from {vae_path}...")
            vae = AutoencoderKL.from_pretrained(
                vae_path,
                use_safetensors=True,
                low_cpu_mem_usage=True
            )
        else:
            print("No VAE found, downloading stabilityai/sdxl-vae...")
            vae = AutoencoderKL.from_pretrained(
                "stabilityai/sdxl-vae",
                use_safetensors=True,
                low_cpu_mem_usage=True
            )
        
        # Clear memory after loading
        if tf.config.list_physical_devices('GPU'):
            tf.keras.backend.clear_session()
        gc.collect()
        
        return cls(vae=vae, model=model, processor=processor)
            
    def enable_model_cpu_offload(self):
        """Enable model CPU offloading to save GPU memory."""
        self.model_cpu_offload = True
        with tf.device('/CPU:0'):
            # Move model weights to CPU
            for layer in self.model.layers:
                if 'layers' in layer.name and 'layers.0' not in layer.name:
                    layer.set_weights([w.numpy() for w in layer.weights])
        tf.keras.backend.clear_session()
        gc.collect()
        
    def disable_model_cpu_offload(self):
        """Disable model CPU offloading."""
        self.model_cpu_offload = False
        with tf.device(self.device):
            # Move model weights back to device
            for layer in self.model.layers:
                if hasattr(layer, 'weights'):
                    layer.set_weights(layer.get_weights())
        
    def vae_encode(self, x, dtype=tf.float16):
        """Encode images using VAE with memory optimization."""
        if self.model_cpu_offload:
            with tf.device(self.device):
                encoded = self.vae.encode(x)
            # Clear GPU memory
            tf.keras.backend.clear_session()
            gc.collect()
        else:
            encoded = self.vae.encode(x)
        return tf.cast(encoded, dtype)
        
    def generate_latents(
        self,
        text_embeddings,
        height,
        width,
        num_inference_steps,
        guidance_scale,
        use_img_guidance=True,
        img_guidance_scale=1.6,
        use_kv_cache=True,
        offload_kv_cache=True,
    ):
        """Generate latents with optional image guidance."""
        
        # Initialize noise
        latents = tf.random.normal(
            [1, height // 8, width // 8, 4],
            dtype=tf.float16
        )
        
        # Setup timesteps
        timesteps = self.scheduler.timesteps
        
        # Initialize KV cache if enabled
        kv_cache = {} if use_kv_cache else None
        
        for t in timesteps:
            # Move model to GPU if offloaded
            if self.model_cpu_offload:
                with tf.device(self.device):
                    self.model.to_gpu()
            
            # Get model prediction
            if separate_cfg_infer:
                noise_pred = []
                for i in range(0, len(latent_model_input), num_cfg):
                    chunk = latent_model_input[i:i + num_cfg]
                    chunk_input_ids = input_data['input_ids'][i:i + num_cfg]
                    chunk_attention_mask = input_data.get('attention_mask')[i:i + num_cfg] if input_data.get('attention_mask') is not None else None
                    
                    # Get chunk prediction
                    chunk_result = self.model(
                        chunk,
                        timesteps[i:i + num_cfg],
                        input_ids=chunk_input_ids,
                        attention_mask=chunk_attention_mask,
                        guidance_scale=guidance_scale,
                        training=False
                    )
                    noise_pred.append(chunk_result)
                noise_pred = tf.concat(noise_pred, axis=0)
            else:
                # Single pass prediction
                noise_pred = self.model(
                    latent_model_input,
                    timesteps,
                    input_ids=input_data['input_ids'],
                    attention_mask=input_data.get('attention_mask'),
                    guidance_scale=guidance_scale,
                    training=False
                )
                
            # Perform guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = tf.split(noise_pred, num_or_size_splits=2, axis=0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
            # Compute previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, timesteps, latents)
            
            # Offload KV cache if enabled
            if use_kv_cache and offload_kv_cache:
                with tf.device('/CPU:0'):
                    for k, v in kv_cache.items():
                        kv_cache[k] = tf.identity(v)
            
            # Move model back to CPU if offloaded
            if self.model_cpu_offload:
                with tf.device('/CPU:0'):
                    self.model.to_cpu()
                tf.keras.backend.clear_session()
        
        return latents

    def __call__(
        self,
        prompt: Union[str, List[str]],
        input_images: Optional[Union[List[str], List[List[str]]]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3,
        use_img_guidance: bool = True,
        img_guidance_scale: float = 1.6,
        max_input_image_size: int = 1024,
        separate_cfg_infer: bool = True,
        offload_model: bool = False,
        use_kv_cache: bool = True,
        offload_kv_cache: bool = True,
        use_input_image_size_as_output: bool = False,
        seed: Optional[int] = None,
        output_type: str = "pil"
    ):
        """Generate images with memory optimizations."""
        # Input validation
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError("Height and width must be multiples of 16")
            
        if input_images is None:
            use_img_guidance = False
            
        if isinstance(prompt, str):
            prompt = [prompt]
            input_images = [input_images] if input_images is not None else None
            
        # Process inputs
        input_data = self.processor.process_text(prompt)
        
        if seed is not None:
            tf.random.set_seed(seed)
            
        # Setup model offloading if requested
        if offload_model:
            self.enable_model_cpu_offload()
        else:
            self.disable_model_cpu_offload()
            
        # Generate initial latents
        num_prompt = len(prompt)
        num_cfg = 2 if guidance_scale > 1.0 else 1
        latent_size_h, latent_size_w = height // 8, width // 8
        
        # Generate on device then move to CPU if needed
        with tf.device(self.device):
            # Generate latents with proper dtype
            latents = tf.random.normal(
                [num_prompt, latent_size_h, latent_size_w, 4],
                dtype=self.model.dtype  # Use model's dtype
            )
            # Cast scheduler init_noise_sigma to model dtype
            init_noise_sigma = tf.cast(self.scheduler.init_noise_sigma, self.model.dtype)
            latents = latents * init_noise_sigma
            
        # Prepare unconditional input
        if guidance_scale > 1.0:
            uncond_input = self.processor.process_text([""] * num_prompt)
            input_data = {
                "input_ids": tf.concat([uncond_input["input_ids"], input_data["input_ids"]], axis=0),
                "attention_mask": tf.concat([uncond_input["attention_mask"], input_data["attention_mask"]], axis=0)
            }
            
        # Denoising loop
        timesteps = self.scheduler.timesteps
        for t in timesteps:
            # Expand for classifier free guidance
            latent_model_input = tf.repeat(latents, num_cfg, axis=0) if guidance_scale > 1.0 else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Get model prediction
            if separate_cfg_infer:
                noise_pred = []
                for i in range(0, len(latent_model_input), num_cfg):
                    chunk = latent_model_input[i:i + num_cfg]
                    chunk_input_ids = input_data['input_ids'][i:i + num_cfg]
                    chunk_attention_mask = input_data.get('attention_mask')[i:i + num_cfg] if input_data.get('attention_mask') is not None else None
                    
                    # Get chunk prediction
                    chunk_result = self.model(
                        chunk,
                        timesteps[i:i + num_cfg],
                        input_ids=chunk_input_ids,
                        attention_mask=chunk_attention_mask,
                        guidance_scale=guidance_scale,
                        training=False
                    )
                    noise_pred.append(chunk_result)
                noise_pred = tf.concat(noise_pred, axis=0)
            else:
                # Single pass prediction
                noise_pred = self.model(
                    latent_model_input,
                    timesteps,
                    input_ids=input_data['input_ids'],
                    attention_mask=input_data.get('attention_mask'),
                    guidance_scale=guidance_scale,
                    training=False
                )
                
            # Perform guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = tf.split(noise_pred, num_or_size_splits=2, axis=0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
            # Compute previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, timesteps, latents)
            
        # Scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        
        # Convert TensorFlow tensor to PyTorch tensor
        latents_np = latents.numpy()
        latents_torch = torch.from_numpy(latents_np).to(self.vae.device)
        
        # Decode with VAE
        if separate_cfg_infer:
            samples = []
            for i in range(0, len(latents_torch), num_cfg):
                chunk = latents_torch[i:i + num_cfg]
                chunk_result = self.vae.decode(chunk).sample
                samples.append(chunk_result)
            samples = torch.cat(samples)
        else:
            samples = self.vae.decode(latents_torch).sample
            
        # Convert back to numpy/tensorflow if needed
        samples = samples.cpu().numpy()
        samples = np.transpose(samples, (0, 2, 3, 1))
        
        # Convert to output format
        if output_type == "pil":
            # Denormalize and convert to PIL
            samples = (samples * 255).round().astype("uint8")
            samples = [Image.fromarray(sample) for sample in samples]
            
        return samples
        
    def decode_latents(self, latents):
        """Decode the latents into images."""
        with tf.device(self.device):
            # Scale and decode the image latents with vae
            latents = 1 / 0.18215 * latents
            
            # Convert TensorFlow tensor to PyTorch tensor
            latents_np = latents.numpy()
            latents_torch = torch.from_numpy(latents_np).to(self.vae.device)
            
            # Decode with VAE
            samples = self.vae.decode(latents_torch).sample
            
            # Convert back to numpy/tensorflow if needed
            samples = samples.cpu().numpy()
            samples = np.transpose(samples, (0, 2, 3, 1))
            
            # Postprocess images
            images = (samples / 2 + 0.5)
            images = tf.clip_by_value(images, 0.0, 1.0)
            images = tf.cast(images * 255, tf.uint8)
            
        return images
        
    def numpy_to_pil(self, images):
        """Convert a numpy image to a PIL image."""
        if images.ndim == 3:
            images = images[None, ...]
        
        pil_images = [Image.fromarray(image.numpy()) for image in images]
        return pil_images[0] if len(pil_images) == 1 else pil_images
