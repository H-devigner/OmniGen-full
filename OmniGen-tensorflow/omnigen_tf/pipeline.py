"""OmniGen Pipeline for text-to-image generation."""

import os
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import gc

from omnigen_tf.model import OmniGen
from omnigen_tf.scheduler import OmniGenScheduler
from omnigen_tf.processor import OmniGenProcessor

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
            else:
                device = '/CPU:0'
                print("No GPU available, using CPU (this will be slow)")
        
        self.device = device
        self.strategy = tf.distribute.OneDeviceStrategy(device)
        self.model_cpu_offload = False
            
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
            
        # Initialize components with GPU support
        with tf.device('/GPU:0'):
            model = OmniGen.from_pretrained(model_name)
            processor = OmniGenProcessor.from_pretrained(model_name)
            scheduler = OmniGenScheduler()
        
        return cls(model, scheduler, processor)

    def enable_model_cpu_offload(self):
        """Enable CPU offloading to save GPU memory."""
        self.model_cpu_offload = True
        with tf.device('/CPU:0'):
            self.model.to_cpu()
        tf.keras.backend.clear_session()
        gc.collect()

    def disable_model_cpu_offload(self):
        """Disable CPU offloading."""
        self.model_cpu_offload = False
        with tf.device(self.device):
            self.model.to_gpu()

    @tf.function(jit_compile=True)
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
        prompt,
        input_images=None,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=3.0,
        use_img_guidance=True,
        img_guidance_scale=1.6,
        max_input_image_size=1024,
        separate_cfg_infer=True,
        use_kv_cache=True,
        offload_kv_cache=True,
        use_input_image_size_as_output=False,
        seed=None,
        output_type="pil",
        **kwargs
    ):
        """Generate images from text prompt with advanced features."""
        
        if seed is not None:
            tf.random.set_seed(seed)
        
        # Process input size
        if use_input_image_size_as_output and input_images is not None:
            if len(input_images) == 1:
                img = Image.open(input_images[0])
                height, width = img.size
        
        # Process inputs
        text_inputs = self.processor.process_text(prompt)
        image_inputs = None
        if input_images is not None:
            image_inputs = self.processor.process_images(
                input_images,
                max_size=max_input_image_size
            )
        
        # Generate latents
        with tf.device(self.device):
            latents = self.generate_latents(
                text_inputs,
                height,
                width,
                num_inference_steps,
                guidance_scale,
                use_img_guidance,
                img_guidance_scale,
                use_kv_cache,
                offload_kv_cache
            )
            
            # Decode latents
            images = self.decode_latents(latents)
        
        # Convert to output format
        if output_type == "pil":
            images = self.numpy_to_pil(images)
        elif output_type == "pt":
            images = tf.convert_to_tensor(images)
            
        return images
    
    def decode_latents(self, latents):
        """Decode the latents into images."""
        with tf.device(self.device):
            # Scale and decode the image latents with vae
            latents = 1 / 0.18215 * latents
            
            # Decode with VAE
            images = self.model.vae.decode(latents)
            
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
