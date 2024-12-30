"""OmniGen TensorFlow Model Implementation

This module contains the TensorFlow implementation of the OmniGen model, 
which is a diffusion model with a Transformer backbone. The implementation
closely follows the PyTorch version while utilizing TensorFlow-specific optimizations.
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import math
from typing import Dict, Optional, List, Union
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from diffusers.loaders import PeftAdapterMixin
import json
from dataclasses import fields
import gc
import huggingface_hub

from omnigen_tf.transformer import Phi3Config, Phi3Transformer
from diffusers.models import AutoencoderKL

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(f"GPU memory growth setting failed: {e}")

# Enable mixed precision globally
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

@tf.function(jit_compile=True)
def modulate(x, shift, scale):
    """Apply adaptive layer normalization modulation."""
    return x * (1 + tf.expand_dims(scale, 1)) + tf.expand_dims(shift, 1)


class TimestepEmbedder(layers.Layer):
    """Timestep embedder layer."""
    
    def __init__(self, hidden_size=2048, **kwargs):
        """Initialize layer."""
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        
        # Initialize MLP layers
        self.mlp = tf.keras.Sequential([
            layers.Dense(hidden_size * 4, activation="gelu", name="mlp_1"),
            layers.Dense(hidden_size, name="mlp_2")
        ], name="mlp")
        
    def call(self, timesteps):
        """Forward pass."""
        # Convert timesteps to float32
        timesteps = tf.cast(timesteps, tf.float32)
        
        # Project timesteps through MLP
        time_embed = self.mlp(timesteps)
        
        return time_embed


class PatchEmbed(layers.Layer):
    """Patch embedding layer."""
    
    def __init__(self, patch_size, hidden_size, in_channels=4, **kwargs):
        """Initialize layer."""
        super().__init__(**kwargs)
        
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        
        # Initialize projection layer
        self.proj = layers.Conv2D(
            filters=hidden_size,
            kernel_size=patch_size,
            strides=patch_size,
            name="proj"
        )
        
    def call(self, x):
        """Forward pass."""
        # Project patches
        x = self.proj(x)
        
        # Reshape to [batch_size, num_patches, hidden_size]
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, -1, self.hidden_size])
        
        return x


class FinalLayer(layers.Layer):
    """Final projection layer."""
    
    def __init__(self, hidden_size, out_channels, **kwargs):
        """Initialize layer."""
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        
        # Use two smaller projections instead of one large one
        self.time_proj = layers.Dense(hidden_size, name="time_proj")
        self.intermediate = layers.Dense(hidden_size * 2, name="intermediate")
        self.out_proj = layers.Dense(out_channels, name="out_proj")
        
    def call(self, x, time_emb):
        """Forward pass."""
        # Project time embeddings
        time_emb = self.time_proj(time_emb)  # [batch_size, hidden_size]
        
        # Add time dimension for broadcasting
        time_emb = tf.expand_dims(time_emb, 1)  # [batch_size, 1, hidden_size]
        
        # Add time embeddings
        x = x + time_emb
        
        # Project through smaller layers
        x = self.intermediate(x)
        x = tf.nn.gelu(x)
        x = self.out_proj(x)
        
        return x


class OmniGen(tf.keras.Model):
    """OmniGen model."""
    
    def __init__(
        self,
        transformer_config,
        patch_size=16,
        in_channels=4,
        pe_interpolation=True,
        pos_embed_max_size=128,
        **kwargs
    ):
        """Initialize model."""
        super().__init__(**kwargs)
        
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.pe_interpolation = pe_interpolation
        self.pos_embed_max_size = pos_embed_max_size
        
        # Initialize components
        self.t_embedder = TimestepEmbedder(
            hidden_size=transformer_config.hidden_size,
            name="t_embedder"
        )
        self.time_token = TimestepEmbedder(
            hidden_size=transformer_config.hidden_size,
            name="time_token"
        )
        self.x_embedder = PatchEmbed(
            patch_size=patch_size,
            hidden_size=transformer_config.hidden_size,
            in_channels=in_channels,
            name="x_embedder"
        )
        
        # Initialize transformer
        self.transformer = Phi3Transformer(transformer_config, name="transformer")
        self.transformer.config.use_cache = False
        
        # Initialize final layer
        self.final_layer = FinalLayer(
            hidden_size=transformer_config.hidden_size,
            out_channels=patch_size * patch_size * in_channels,
            name="final_layer"
        )
        
        # Initialize output channels
        self.out_channels = in_channels

    def call(
        self,
        inputs,
        timestep,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        guidance_scale=None,
        training=False,
        **kwargs
    ):
        """Forward pass with classifier-free guidance support."""
        # Get batch size
        batch_size = tf.shape(inputs)[0]
        
        # Handle classifier-free guidance
        if guidance_scale is not None and guidance_scale > 1.0:
            # Duplicate inputs for classifier-free guidance
            inputs = tf.concat([inputs] * 2, axis=0)
            
            # Ensure timestep is 1D before duplicating
            timestep = tf.reshape(timestep, [batch_size, -1])
            timestep = tf.tile(timestep, [2, 1])
            
            # Duplicate other inputs if provided
            if input_ids is not None:
                input_ids = tf.concat([input_ids] * 2, axis=0)
            if attention_mask is not None:
                attention_mask = tf.concat([attention_mask] * 2, axis=0)
            if position_ids is not None:
                position_ids = tf.concat([position_ids] * 2, axis=0)
        
        # Regular forward pass
        outputs = self.forward_pass(
            inputs,
            timestep,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            training=training
        )
        
        # Apply classifier-free guidance
        if guidance_scale is not None and guidance_scale > 1.0:
            # Split unconditional and conditional outputs
            uncond_output, cond_output = tf.split(outputs, num_or_size_splits=2, axis=0)
            
            # Apply classifier-free guidance formula
            return uncond_output + guidance_scale * (cond_output - uncond_output)
            
        return outputs

    def forward_pass(
        self,
        latents,
        timestep,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        training=False,
    ):
        """Model forward pass."""
        # Get batch size and dimensions
        batch_size = tf.shape(latents)[0]
        h = tf.shape(latents)[1]
        w = tf.shape(latents)[2]
        
        # Debug prints
        tf.print("Input shapes:")
        tf.print("latents:", tf.shape(latents))
        tf.print("timestep:", tf.shape(timestep))
        
        # Get input dtype
        input_dtype = latents.dtype
        
        # Embed timesteps
        t_emb = tf.cast(self.t_embedder(timestep), input_dtype)
        time_token = tf.cast(self.time_token(timestep), input_dtype)  # Shape: [batch_size, embed_dim]
        
        tf.print("After embedding:")
        tf.print("t_emb:", tf.shape(t_emb))
        tf.print("time_token:", tf.shape(time_token))
        
        # Patch and embed input latents
        x = self.x_embedder(latents)  # Shape: [batch_size, num_patches, embed_dim]
        tf.print("x after embedding:", tf.shape(x))
        
        # Add positional embeddings
        if position_ids is None:
            # Use default positional embeddings
            pos_embed = get_2d_sincos_pos_embed(
                self.transformer.config.hidden_size,
                h // self.patch_size,
                w // self.patch_size,
                cls_token=False
            )
            pos_embed = tf.expand_dims(tf.convert_to_tensor(pos_embed, dtype=input_dtype), 0)
            pos_embed = tf.tile(pos_embed, [batch_size, 1, 1])
        else:
            # Use provided position IDs
            pos_embed = tf.cast(tf.gather(self.pos_embed, position_ids), input_dtype)
            
        tf.print("pos_embed:", tf.shape(pos_embed))
        x = x + pos_embed
        
        # Add time token - reshape to match batch dimension
        time_token = tf.reshape(time_token, [batch_size, 1, -1])  # Shape: [batch_size, 1, embed_dim]
        tf.print("time_token reshaped:", tf.shape(time_token))
        tf.print("x before concat:", tf.shape(x))
        x = tf.concat([time_token, x], axis=1)
        tf.print("x after concat:", tf.shape(x))
        
        # Process text input if provided
        if input_ids is not None:
            # Add batch dimension if needed
            if len(tf.shape(input_ids)) == 1:
                input_ids = tf.expand_dims(input_ids, 0)
                
            # Get text embeddings from transformer
            text_outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                training=training
            )
            text_embeds = tf.cast(text_outputs.last_hidden_state, input_dtype)
            tf.print("text_embeds:", tf.shape(text_embeds))
            
            # Ensure batch dimension is consistent
            if len(tf.shape(text_embeds)) == 2:
                text_embeds = tf.expand_dims(text_embeds, 0)
            if tf.shape(text_embeds)[0] != batch_size:
                text_embeds = tf.tile(text_embeds, [batch_size, 1, 1])
                
            # Concatenate with image embeddings
            x = tf.concat([text_embeds, x], axis=1)
            tf.print("x after text concat:", tf.shape(x))
            
            # Update attention mask if needed
            if attention_mask is not None:
                # Add batch dimension if needed
                if len(tf.shape(attention_mask)) == 1:
                    attention_mask = tf.expand_dims(attention_mask, 0)
                # Create full attention mask
                image_mask = tf.ones((batch_size, tf.shape(x)[1] - tf.shape(text_embeds)[1]))
                attention_mask = tf.concat([attention_mask, image_mask], axis=1)
        
        # Pass through transformer blocks
        hidden_states = x
        for block in self.transformer.blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                training=training
            )
            # Ensure batch dimension is maintained
            if len(tf.shape(hidden_states)) == 2:
                hidden_states = tf.expand_dims(hidden_states, 0)
                
        hidden_states = tf.cast(self.transformer.ln_f(hidden_states), input_dtype)
        
        tf.print("Final hidden states shape:", tf.shape(hidden_states))
        
        # Process output
        if input_ids is not None:
            # Remove text embeddings
            hidden_states = hidden_states[:, tf.shape(text_embeds)[1]:]
            
        # Remove time token
        hidden_states = hidden_states[:, 1:]
        
        # Final layer
        output = self.final_layer(hidden_states, t_emb)
        
        # Reshape to image dimensions
        output = tf.reshape(output, [batch_size, h, w, self.out_channels])
        
        return output

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """Load model from pretrained weights with optimized memory usage."""
        print(f"Loading model weights...")
        
        # Handle local path vs HuggingFace repo
        if os.path.exists(model_name_or_path):
            weights_file = os.path.join(model_name_or_path, "model.safetensors")
            if not os.path.exists(weights_file):
                raise ValueError(f"Could not find model.safetensors in {model_name_or_path}")
        else:
            # Download from HuggingFace
            try:
                weights_file = huggingface_hub.hf_hub_download(
                    repo_id=model_name_or_path,
                    filename="model.safetensors",
                    cache_dir=os.getenv("HF_HUB_CACHE", None)
                )
            except Exception as e:
                print(f"Error downloading model: {str(e)}")
                raise
        
        # Initialize model with default config
        model = cls(transformer_config=Phi3Config(), **kwargs)
        
        # Load safetensors metadata first
        with open(weights_file, 'rb') as f:
            header = f.read(8)
            header_size = int.from_bytes(header, 'little')
            metadata = f.read(header_size).decode('utf-8')
            metadata = json.loads(metadata)
            
        # Define dtype mapping
        dtype_map = {
            'F32': np.float32,
            'F16': np.float16,
            'BF16': np.float32,  # Convert bfloat16 to float32
            'I32': np.int32,
            'I64': np.int64
        }
        
        # Load weights in chunks with memory cleanup
        total_tensors = len(metadata)
        for i, (tensor_name, tensor_info) in enumerate(metadata.items(), 1):
            if hasattr(model, tensor_name):
                print(f"\rLoading tensor {i}/{total_tensors}: {tensor_name}", end="")
                offset = tensor_info['data_offsets'][0]
                length = tensor_info['data_offsets'][1] - offset
                dtype_str = tensor_info['dtype']
                dtype = dtype_map.get(dtype_str, np.float32)  # Default to float32 if unknown
                shape = tensor_info['shape']
                
                with open(weights_file, 'rb') as f:
                    f.seek(8 + header_size + offset)
                    tensor_bytes = f.read(length)
                    tensor = np.frombuffer(tensor_bytes, dtype=dtype).reshape(shape)
                    setattr(model, tensor_name, tf.convert_to_tensor(tensor))
                
                # Clear memory after each tensor
                del tensor_bytes
                del tensor
                gc.collect()
        print("\nModel weights loaded successfully!")
        
        # Final memory cleanup
        if tf.config.list_physical_devices('GPU'):
            tf.keras.backend.clear_session()
        gc.collect()
        
        return model

def get_2d_sincos_pos_embed(
    embed_dim,
    grid_size_h,
    grid_size_w=None,
    cls_token=False,
    interpolation_scale=1.0,
    base_size=16
):
    """Get 2D sine-cosine positional embeddings.
    
    Args:
        embed_dim: Output dimension for each position
        grid_size_h: Number of patches in height
        grid_size_w: Number of patches in width (default: same as height)
        cls_token: If True, add a classification token
        interpolation_scale: Scale factor for interpolation
        base_size: Base size for scaling calculations
        
    Returns:
        pos_embed: Position embeddings, shape (H*W, D) or (1+H*W, D)
    """
    if grid_size_w is None:
        grid_size_w = grid_size_h
        
    # No interpolation scaling for now
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # Here we reverse the order
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    
    # Get positional embeddings
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Get 2D sine-cosine positional embeddings from grid."""
    assert embed_dim % 2 == 0
    
    # Use half the dimensions for each grid
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Get 1D sine-cosine positional embeddings from grid."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)
    
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
    
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
