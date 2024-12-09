"""OmniGen pipeline for text-to-image generation."""

import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer
from PIL import Image
import logging
from .model import OmniGen  # Importing OmniGenTF

logger = logging.getLogger(__name__)

class OmniGenPipeline:
    def __init__(self, model, tokenizer=None):
        self.model = model
        if tokenizer is None:
            # Initialize the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
            # Set the padding token if not already defined
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token  # Use eos_token as pad_token
        else:
            self.tokenizer = tokenizer
    
    @classmethod
    def from_pretrained(cls, model_name_or_path, local_files_only=False):
        """Load the pipeline from a pretrained model."""
        logger.info(f"Loading pipeline from {model_name_or_path}")
        
        # Load the model
        model = OmniGenTF.from_pretrained(model_name_or_path, local_files_only=local_files_only)
        
        return cls(model=model)
    
    @staticmethod
    def prepare_image_tensor(batch_size=1, channels=4, height=64, width=64):
        """Prepare initial image tensor"""
        return tf.random.normal([batch_size, channels, height, width])
    
    def encode_prompt(self, prompt):
        """Encode text prompt to input tensors"""
        # Tokenize text
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        
        # Convert to TensorFlow tensors
        input_ids = tf.convert_to_tensor(text_inputs.input_ids.numpy(), dtype=tf.int32)
        attention_mask = tf.convert_to_tensor(text_inputs.attention_mask.numpy(), dtype=tf.int32)
        
        # Create position IDs
        batch_size = tf.shape(input_ids)[0]
        seq_length = tf.shape(input_ids)[1]
        position_ids = tf.range(seq_length)
        position_ids = tf.repeat(position_ids[None], batch_size, axis=0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }
    
    def prepare_timesteps(self, num_inference_steps=50):
        """Prepare diffusion timesteps"""
        timesteps = tf.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
        return timesteps
    
    def denoise_latent(self, latent, timesteps, text_embeddings):
        """Denoise the image latents"""
        for t in timesteps:
            # Expand dimensions for batch processing
            timestep = tf.fill([tf.shape(latent)[0]], t)
            
            # Model inputs
            model_inputs = {
                "x": latent,
                "timestep": timestep,
                **text_embeddings
            }
            
            # Get model prediction
            noise_pred = self.model(model_inputs)
            
            # Update latent
            latent = self.scheduler_step(noise_pred, t, latent)
        
        return latent
    
    def scheduler_step(self, model_output, timestep, sample):
        """Simple DDIM-style scheduler step"""
        prev_timestep = tf.maximum(timestep - 1.0/50.0, 0.0)
        alpha_prod_t = timestep
        alpha_prod_t_prev = prev_timestep
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        pred_original_sample = (sample - tf.sqrt(beta_prod_t) * model_output) / tf.sqrt(alpha_prod_t)
        pred_sample_direction = tf.sqrt(beta_prod_t_prev) * model_output
        prev_sample = tf.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
        
        return prev_sample
    
    def decode_latents(self, latents):
        """Convert latents to image"""
        # Scale and transpose latents
        latents = 1 / 0.18215 * latents
        latents = tf.transpose(latents, [0, 2, 3, 1])
        
        # Clip values
        image = tf.clip_by_value(latents, -1, 1)
        image = (image + 1) / 2
        image = tf.clip_by_value(image * 255, 0, 255)
        image = tf.cast(image, tf.uint8)
        
        return image
    
    def __call__(self,
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        negative_prompt=None,
        height=512,
        width=512,
    ):
        """Generate image from text prompt"""
        # Encode prompt
        text_embeddings = self.encode_prompt(prompt)
        
        # Prepare latents
        latents = self.prepare_image_tensor()
        
        # Prepare timesteps
        timesteps = self.prepare_timesteps(num_inference_steps)
        
        # Denoise latents
        latents = self.denoise_latent(latents, timesteps, text_embeddings)
        
        # Decode latents to image
        image = self.decode_latents(latents)
        
        # Convert to PIL Image
        image = Image.fromarray(image[0].numpy())
        
        return image
