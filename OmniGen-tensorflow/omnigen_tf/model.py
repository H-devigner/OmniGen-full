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

from omnigen_tf.transformer import Phi3Config, Phi3Transformer

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

        # Set compute dtype to float16 for better GPU performance
        self.compute_dtype = tf.float16
        self.variable_dtype = tf.float32

        hidden_size = transformer_config.hidden_size
        
        # Initialize embedders
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
        
        # Initialize transformer
        self.transformer = Phi3Transformer(transformer_config, name="transformer")
        
    def call(
        self,
        inputs,
        timestep,
        input_ids=None,
        attention_mask=None,
        training=False,
    ):
        """Forward pass with automatic mixed precision."""
        # Cast inputs to compute dtype (float16)
        inputs = tf.cast(inputs, self.compute_dtype)
        
        # Run forward pass in float16
        outputs = self.forward_pass(inputs, timestep, input_ids, attention_mask, training)
        
        # Cast outputs back to variable dtype (float32) for stability
        outputs = tf.cast(outputs, self.variable_dtype)
        
        return outputs

    def forward_pass(
        self,
        latents,
        timestep,
        input_ids=None,
        attention_mask=None,
        training=False,
    ):
        """Model forward pass."""
        # Cast inputs to float32
        latents = tf.cast(latents, tf.float32)
        timestep = tf.cast(timestep, tf.int32)
        
        # Get batch size and dimensions
        batch_size = tf.shape(latents)[0]
        h = tf.shape(latents)[1]
        w = tf.shape(latents)[2]
        
        # Embed latents
        x = self.x_embedder(latents)  # [B, H*W, C]
        
        # Create position IDs
        position_ids = tf.range(tf.shape(x)[1], dtype=tf.int32)[None]
        position_ids = tf.tile(position_ids, [batch_size, 1])
        
        # Get timestep embeddings
        t_emb = self.t_embedder(timestep[None])  # [1, C]
        time_tokens = self.time_token(timestep[None])  # [1, C]
        
        # Add time tokens to beginning of sequence
        time_tokens = tf.tile(time_tokens, [batch_size, 1])  # [B, C]
        time_tokens = time_tokens[:, None, :]  # [B, 1, C]
        x = tf.concat([time_tokens, x], axis=1)  # [B, 1+H*W, C]
        
        # Update position IDs for time token
        position_ids = tf.pad(position_ids, [[0, 0], [1, 0]])
        
        # Process through transformer
        x = self.transformer(
            x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            training=training
        )
        
        # Remove time token
        x = x[:, 1:]  # [B, H*W, C]
        
        # Reshape to image dimensions
        x = self.unpatchify(x, h, w)  # [B, H, W, C]
        
        return x
        
    def unpatchify(self, x, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        p = self.patch_size
        c = self.in_channels
        
        # Calculate dimensions
        h_unpatched = h * p
        w_unpatched = w * p
        
        # Reshape to image dimensions
        x = tf.reshape(x, [-1, h, w, self.out_channels])  # [B, H, W, C]
        
        return x

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load pretrained model from HuggingFace Hub or local path."""
        import os
        import json
        from huggingface_hub import snapshot_download
        from safetensors.tensorflow import load_file
        
        # Get model path
        model_path = pretrained_model_name_or_path
        if not os.path.exists(model_path):
            cache_folder = os.getenv('HF_HUB_CACHE', None)
            model_path = snapshot_download(
                repo_id=pretrained_model_name_or_path,
                cache_dir=cache_folder,
                ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5']
            )
            
        # Load config
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            config_dict = {}
            
        # Update config with kwargs
        config_dict.update(kwargs)
        
        # Create model with mixed precision
        model = cls(transformer_config=config_dict)
        
        # Load weights
        weights_file = os.path.join(model_path, "model.safetensors")
        if os.path.exists(weights_file):
            print("Loading weights from safetensors...")
            
            # Load weights
            state_dict = load_file(weights_file)
            
            # Map weights to TensorFlow format
            for name, weight in state_dict.items():
                try:
                    # Convert weight to float16 for GPU operations
                    weight = tf.cast(weight, tf.float16)
                    
                    # Find corresponding layer in TF model
                    if name.startswith('transformer.'):
                        tf_name = name.replace('transformer.', 'transformer/')
                    else:
                        tf_name = name.replace('.', '/')
                        
                    # Get layer by name
                    layer = model.get_layer(tf_name)
                    if layer is not None:
                        # Assign weights
                        if 'weight' in name:
                            layer.kernel.assign(weight)
                        elif 'bias' in name:
                            layer.bias.assign(weight)
                except Exception as e:
                    print(f"Warning: Could not load weight {name}: {str(e)}")
                    
            print("Weights loaded successfully!")
        else:
            print(f"No weights found at {weights_file}")
            
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
