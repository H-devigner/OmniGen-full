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


class TimestepEmbedder(Model):
    """Embeds scalar timesteps into vector representations."""
    
    def __init__(self, hidden_size, frequency_embedding_size=256, **kwargs):
        super().__init__(**kwargs)
        self.mlp = tf.keras.Sequential([
            layers.Dense(hidden_size, use_bias=True),
            layers.Activation('silu'),
            layers.Dense(hidden_size, use_bias=True)
        ])
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    @tf.function(jit_compile=True)
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = tf.exp(
            -math.log(max_period) * tf.range(half, dtype=t.dtype) / half
        )
        args = tf.cast(t[:, None], dtype=freqs.dtype) * freqs[None]
        embedding = tf.concat([tf.cos(args), tf.sin(args)], axis=-1)
        if dim % 2:
            embedding = tf.concat([embedding, tf.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def call(self, t, dtype=tf.float32):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = tf.cast(t_freq, dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class PatchEmbed(layers.Layer):
    """2D Image to Patch Embedding."""
    
    def __init__(self, embed_dim=768, patch_size=16, in_channels=3, **kwargs):
        """Initialize patch embedding layer."""
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        # Initialize projection layer
        self.proj = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding='valid',
            name='proj'
        )
        
    def call(self, x):
        """Forward pass."""
        # Rearrange input to NCHW format
        if x.shape[-1] == self.in_channels:  # NHWC format
            x = tf.transpose(x, [0, 3, 1, 2])
            
        B, C, H, W = x.shape
        
        # Ensure input dimensions are compatible with patch size
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Input image dimensions ({H}, {W}) must be divisible by "
                f"patch size ({self.patch_size})"
            )
            
        # Convert back to NHWC for Conv2D
        x = tf.transpose(x, [0, 2, 3, 1])
        
        # Apply patch embedding
        x = self.proj(x)
        
        # Reshape to (B, N, C)
        x = tf.reshape(x, [B, -1, self.embed_dim])
        
        return x
        
    def get_num_patches(self, h, w):
        """Get number of patches for given input dimensions."""
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f"Input image dimensions ({h}, {w}) must be divisible by "
                f"patch size ({self.patch_size})"
            )
        return (h // self.patch_size) * (w // self.patch_size)


class TimeToken(tf.keras.layers.Layer):
    """Time token embedding layer."""
    
    def __init__(self, embed_dim=2048, **kwargs):
        """Initialize layer."""
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        
        # Create MLP for time embedding
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(embed_dim, activation='gelu'),
            tf.keras.layers.Dense(embed_dim)
        ])
        
    def call(self, t):
        """Forward pass."""
        # Reshape timestep to [batch_size, 1]
        if len(tf.shape(t)) == 0:
            t = tf.expand_dims(t, 0)  # Add batch dimension
        if len(tf.shape(t)) == 1:
            t = tf.expand_dims(t, 1)  # Add feature dimension
            
        # Convert to float32
        t = tf.cast(t, tf.float32)
        
        return self.mlp(t)
        
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim
        })
        return config


class FinalLayer(layers.Layer):
    """Final layer for image generation."""
    
    def __init__(self, patch_size, in_channels, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Initialize projection layer
        self.proj = layers.Dense(patch_size * patch_size * in_channels, name="proj")
        
    def call(self, x, time_emb):
        """Forward pass."""
        # Add time embedding
        x = x + time_emb[:, None, :]
        
        # Project to patch space
        x = self.proj(x)
        
        return x


class OmniGen(tf.keras.Model):
    """OmniGen model implementation."""
    
    def __init__(
        self,
        transformer_config,
        patch_size=2,
        in_channels=4,
        pe_interpolation: float = 1.0,
        pos_embed_max_size: int = 192,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.pos_embed_max_size = pos_embed_max_size
        
        # Convert dict to Phi3Config if needed
        if isinstance(transformer_config, dict):
            transformer_config = Phi3Config(**transformer_config)
        elif not isinstance(transformer_config, Phi3Config):
            raise ValueError("transformer_config must be either a dict or Phi3Config instance")
        
        hidden_size = transformer_config.hidden_size
        
        # Initialize embedders with mixed precision
        self.x_embedder = PatchEmbed(
            embed_dim=hidden_size,
            patch_size=patch_size,
            in_channels=in_channels,
            name="x_embedder"
        )
        self.input_x_embedder = PatchEmbed(
            embed_dim=hidden_size,
            patch_size=patch_size,
            in_channels=in_channels,
            name="input_x_embedder"
        )
        
        # Initialize time embedders
        self.time_token = TimestepEmbedder(hidden_size, name="time_token")
        self.t_embedder = TimestepEmbedder(hidden_size, name="t_embedder")
        
        # Initialize positional embedding
        self.pe_interpolation = pe_interpolation
        pos_embed = get_2d_sincos_pos_embed(
            hidden_size, 
            pos_embed_max_size, 
            interpolation_scale=self.pe_interpolation, 
            base_size=64
        )
        self.pos_embed = tf.Variable(
            initial_value=tf.expand_dims(tf.convert_to_tensor(pos_embed, dtype=tf.float32), 0),
            trainable=False,
            name="pos_embed"
        )
        
        # Initialize transformer
        self.transformer = Phi3Transformer(transformer_config, name="transformer")
        self.transformer.config.use_cache = False
        
        # Initialize final layer
        self.final_layer = FinalLayer(
            hidden_size, 
            patch_size, 
            self.out_channels
        )

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
        # Handle classifier-free guidance
        if guidance_scale is not None and guidance_scale > 1.0:
            # Duplicate input for classifier-free guidance
            inputs = tf.concat([inputs] * 2, axis=0)
            timestep = tf.concat([timestep] * 2, axis=0)
            
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
        
        # Embed timesteps
        t_emb = self.t_embedder(timestep)
        time_token = self.time_token(timestep)
        
        # Patch and embed input latents
        x = self.x_embedder(latents)
        
        # Add positional embeddings
        if position_ids is None:
            # Use default positional embeddings
            pos_embed = get_2d_sincos_pos_embed(
                self.transformer.config.hidden_size,
                h // self.patch_size,
                w // self.patch_size,
                cls_token=False
            )
            pos_embed = tf.expand_dims(tf.convert_to_tensor(pos_embed, dtype=tf.float32), 0)
            pos_embed = tf.tile(pos_embed, [batch_size, 1, 1])
        else:
            # Use provided position IDs
            pos_embed = tf.gather(self.pos_embed, position_ids)
            
        x = x + pos_embed
        
        # Add time token
        x = tf.concat([time_token[:, None, :], x], axis=1)
        
        # Process text input if provided
        if input_ids is not None:
            # Get text embeddings from transformer
            text_outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                training=training
            )
            text_embeds = text_outputs.last_hidden_state
            
            # Concatenate with image embeddings
            x = tf.concat([text_embeds, x], axis=1)
            
            # Update attention mask if needed
            if attention_mask is not None:
                # Create full attention mask
                image_mask = tf.ones((batch_size, tf.shape(x)[1] - tf.shape(text_embeds)[1]))
                attention_mask = tf.concat([attention_mask, image_mask], axis=1)
        
        # Pass through transformer
        outputs = self.transformer(
            inputs_embeds=x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            training=training
        )
        hidden_states = outputs.last_hidden_state
        
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
