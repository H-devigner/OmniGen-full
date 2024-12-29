"""OmniGen Pipeline for text-to-image generation."""

import os
import gc
import numpy as np
import tensorflow as tf
from PIL import Image
from typing import List, Optional, Union
from huggingface_hub import snapshot_download
import json

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
            model_output = self.model(
                latents,
                t,
                text_embeddings,
                kv_cache=kv_cache,
                training=False
            )
            
            # Apply guidance
            if guidance_scale > 1.0:
                uncond_embeddings, cond_embeddings = text_embeddings
                uncond_output = model_output[0]
                cond_output = model_output[1]
                
                # Classifier-free guidance
                noise_pred = uncond_output + guidance_scale * (cond_output - uncond_output)
                
                # Image guidance if enabled
                if use_img_guidance and img_guidance_scale > 0:
                    img_output = model_output[2]
                    noise_pred = noise_pred + img_guidance_scale * (img_output - uncond_output)
            
            # Scheduler step
            latents = self.scheduler.step(
                noise_pred,
                t,
                latents
            )
            
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
        
        # Setup model offloading if requested
        if offload_model:
            self.enable_model_cpu_offload()
        else:
            self.disable_model_cpu_offload()
            
        # Generate initial latents
        num_prompt = len(prompt)
        num_cfg = 2 if use_img_guidance else 1
        latent_size_h, latent_size_w = height // 8, width // 8
        
        if seed is not None:
            tf.random.set_seed(seed)
            
        # Generate on device then move to CPU if needed
        with tf.device(self.device):
            latents = tf.random.normal(
                [num_prompt, latent_size_h, latent_size_w, 4],
                dtype=tf.float16
            )
            latents = tf.repeat(latents, 1 + num_cfg, axis=0)
            
        # Process input images with memory optimization
        if input_images is not None:
            input_img_latents = []
            
            # Move VAE to device temporarily
            if self.model_cpu_offload:
                with tf.device(self.device):
                    for img_batch in input_images:
                        img_latents = []
                        for img in (img_batch if isinstance(img_batch, list) else [img_batch]):
                            # Process image
                            if isinstance(img, str):
                                img = Image.open(img).convert('RGB')
                            img_tensor = tf.convert_to_tensor(np.array(img))
                            img_tensor = tf.cast(img_tensor, tf.float32) / 127.5 - 1.0
                            
                            # Encode with VAE
                            img_latent = self.vae_encode(img_tensor)
                            img_latents.append(img_latent)
                            
                        input_img_latents.append(img_latents)
                
                # Clear GPU memory
                tf.keras.backend.clear_session()
                gc.collect()
                
        # Generate images with memory-optimized inference
        with tf.device(self.device):
            # Run inference in chunks if separate_cfg_infer is True
            if separate_cfg_infer:
                samples = []
                for i in range(0, len(latents), num_cfg):
                    chunk = latents[i:i + num_cfg]
                    chunk_result = self.model(
                        chunk,
                        input_data['input_ids'],
                        input_data['attention_mask'],
                        guidance_scale=guidance_scale,
                        img_guidance_scale=img_guidance_scale if use_img_guidance else None,
                        use_kv_cache=use_kv_cache
                    )
                    samples.append(chunk_result)
                    
                    # Clear memory after each chunk
                    if offload_kv_cache:
                        tf.keras.backend.clear_session()
                        gc.collect()
                        
                samples = tf.concat(samples, axis=0)
            else:
                samples = self.model(
                    latents,
                    input_data['input_ids'],
                    input_data['attention_mask'],
                    guidance_scale=guidance_scale,
                    img_guidance_scale=img_guidance_scale if use_img_guidance else None,
                    use_kv_cache=use_kv_cache
                )
            
            # Post-process samples
            samples = tf.cast(samples, tf.float32)
            samples = samples / self.vae.config.scaling_factor
            
            if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor is not None:
                samples = samples + self.vae.config.shift_factor
                
            # Decode with VAE
            if self.model_cpu_offload:
                with tf.device(self.device):
                    samples = self.vae.decode(samples)
            else:
                samples = self.vae.decode(samples)
                
            # Final processing
            samples = tf.clip_by_value(samples * 0.5 + 0.5, 0, 1)
            
        # Clear GPU memory
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Return results in requested format
        if output_type == "tf":
            return samples
        else:
            samples = tf.cast(samples * 255, tf.uint8)
            samples = samples.numpy()
            return [Image.fromarray(img) for img in samples]
        
    def decode_latents(self, latents):
        """Decode the latents into images."""
        with tf.device(self.device):
            # Scale and decode the image latents with vae
            latents = 1 / 0.18215 * latents
            
            # Decode with VAE
            images = self.vae.decode(latents)
            
            # Postprocess images
            images = (images / 2 + 0.5)
            images = tf.clip_by_value(images, 0.0, 1.0)
            images = tf.cast(images * 255, tf.uint8)
            
        return images
        
    def numpy_to_pil(self, images):
        """Convert a numpy image to a PIL image."""
        if images.ndim == 3:
            images = images[None, ...]
        
        pil_images = [Image.fromarray(image.numpy()) for image in images]
        return pil_images[0] if len(pil_images) == 1 else pil_images
