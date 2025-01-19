"""OmniGen Pipeline for image generation."""

import os
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import json
import torch
import gc

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
    """Memory-efficient pipeline for OmniGen model."""
    
    def __init__(self, model, tokenizer=None, device=None):
        """Initialize pipeline."""
        self.model = model
        self.tokenizer = tokenizer
        
        # Set up device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Pipeline using device: {self.device}")
        
        # Configure GPU settings if available
        if torch.cuda.is_available():
            # Enable TF32 for better performance
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            
            # Configure TensorFlow GPU memory growth
            for device in tf.config.list_physical_devices('GPU'):
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                except:
                    pass
        
        # Enable mixed precision
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load pipeline from a pretrained model.
        
        Args:
            pretrained_model_name_or_path (str): Path or name of pretrained model
            **kwargs: Additional arguments to pass to components
            
        Returns:
            OmniGenPipeline: Loaded pipeline instance
        """
        try:
            # Configure device
            device = kwargs.pop('device', None)
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            print(f"Loading pipeline on device: {device}")
            
            # Enable mixed precision
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="tokenizer",
                use_fast=True
            )
            
            # Load scheduler
            scheduler = OmniGenScheduler.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="scheduler"
            )
            
            # Load model
            model = OmniGen.from_pretrained(
                "Shitao/OmniGen-v1",
                subfolder="model",
                device=device
            )
            
            # Initialize processor
            processor = OmniGenProcessor(
                tokenizer=tokenizer,
                device=device
            )
            
            # Create pipeline instance
            pipeline = cls(
                model=model,
                tokenizer=tokenizer,
                device=device,
                **kwargs
            )
            
            return pipeline
            
        except Exception as e:
            print(f"Error loading pipeline: {str(e)}")
            # Clean up on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            tf.keras.backend.clear_session()
            raise

    def prepare_inputs(self, prompt, negative_prompt=None, input_images=None, height=1024, width=1024, num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=1, img_guidance_scale=1.5, seed=None, use_img_guidance=True):
        """Prepare inputs with memory optimization."""
        try:
            # Set random seed if provided
            if seed is not None:
                tf.random.set_seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            # Process text inputs
            text_inputs = []
            if negative_prompt is None:
                negative_prompt = ""
                
            # Handle prompts efficiently
            if not isinstance(prompt, list):
                prompt = [prompt]
            if not isinstance(negative_prompt, list):
                negative_prompt = [negative_prompt]
                
            # Process prompts in batches
            for p, n in zip(prompt * num_images_per_prompt, negative_prompt * num_images_per_prompt):
                text_inputs.extend([p, n] if use_img_guidance else [p])
                
            # Tokenize efficiently
            tokenized = self.tokenizer(
                text_inputs,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="tf",
            )
            
            # Process input images if provided
            if input_images is not None:
                if not isinstance(input_images, list):
                    input_images = [input_images]
                    
                processed_images = []
                for image in input_images:
                    # Convert PIL images to numpy arrays
                    if isinstance(image, Image.Image):
                        image = np.array(image)
                    
                    # Ensure correct shape and dtype
                    if image.dtype != np.float32:
                        image = image.astype(np.float32) / 255.0
                    if len(image.shape) == 3:
                        image = image[None, ...]
                        
                    processed_images.append(image)
                    
                # Stack images efficiently
                input_images = np.concatenate(processed_images, axis=0)
                input_images = tf.convert_to_tensor(input_images, dtype=tf.float32)
                
            return {
                "input_ids": tokenized.input_ids,
                "attention_mask": tokenized.attention_mask,
                "input_images": input_images,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": num_images_per_prompt,
                "img_guidance_scale": img_guidance_scale,
                "use_img_guidance": use_img_guidance
            }
            
        except Exception as e:
            print(f"Error in prepare_inputs: {str(e)}")
            raise
            
    def __call__(self, prompt, negative_prompt=None, input_images=None, height=1024, width=1024, num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=1, img_guidance_scale=1.5, seed=None):
        """Memory-efficient pipeline execution."""
        try:
            # Prepare inputs
            inputs = self.prepare_inputs(
                prompt=prompt,
                negative_prompt=negative_prompt,
                input_images=input_images,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                img_guidance_scale=img_guidance_scale,
                seed=seed
            )
            
            # Run model with memory optimization
            with torch.cuda.amp.autocast():
                with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                    images = self.model(**inputs)
            
            # Process outputs
            if isinstance(images, list):
                images = [Image.fromarray((img * 255).numpy().astype(np.uint8)) for img in images]
            else:
                images = Image.fromarray((images[0] * 255).numpy().astype(np.uint8))
                
            return images
            
        except Exception as e:
            print(f"Error in pipeline execution: {str(e)}")
            # Clean up on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            tf.keras.backend.clear_session()
            raise
            
        finally:
            # Final cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            tf.keras.backend.clear_session()

pipeline = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")
image = pipeline(
    prompt="your prompt here",
    height=512,
    width=512,
    num_inference_steps=50,
    guidance_scale=7.5
)
