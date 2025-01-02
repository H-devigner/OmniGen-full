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
import json
from dataclasses import fields

from omnigen_tf.transformer import Phi3Config, Phi3Transformer


@tf.function(jit_compile=True)
def modulate(x, shift, scale):
    """Apply adaptive layer normalization modulation."""
    return x * (1 + tf.expand_dims(scale, 1)) + tf.expand_dims(shift, 1)


class TimestepEmbedder(Model):
    """Embeds scalar timesteps into vector representations."""
    
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = tf.keras.Sequential([
            layers.Dense(hidden_size, use_bias=True, name="mlp_0"),
            layers.Activation('silu'),
            layers.Dense(hidden_size, use_bias=True, name="mlp_2")
        ])
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    @tf.function(jit_compile=True)
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = tf.exp(
            -math.log(max_period) * tf.range(half, dtype=t.dtype) / half
        )
        args = tf.cast(t[:, None], dtype=freqs.dtype) * freqs[None]
        embedding = tf.concat([tf.cos(args), tf.sin(args)], axis=-1)
        if dim % 2:
            embedding = tf.concat([embedding, tf.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def call(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class PatchEmbed(layers.Layer):
    """2D Image to Patch Embedding."""
    
    def __init__(self, patch_size=2, in_channels=4, embed_dim=768, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        self.proj = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding='valid',
            use_bias=True,
            name='proj'
        )
        
    def call(self, x):
        """Forward pass."""
        # Handle NHWC format
        if x.shape[-1] == self.in_channels:
            x = tf.transpose(x, [0, 3, 1, 2])  # NHWC -> NCHW
            
        B, C, H, W = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        
        # Ensure input dimensions are compatible with patch size
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Input image dimensions ({H}, {W}) must be divisible by "
                f"patch size ({self.patch_size})"
            )
            
        # Convert to NHWC for Conv2D
        x = tf.transpose(x, [0, 2, 3, 1])  # NCHW -> NHWC
        
        # Apply patch embedding
        x = self.proj(x)
        
        # Reshape to (B, N, C)
        x = tf.reshape(x, [B, -1, self.embed_dim])
        
        return x


class FinalLayer(layers.Layer):
    """The final layer of DiT."""
    
    def __init__(self, hidden_size, patch_size, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.norm_final = layers.LayerNormalization(epsilon=1e-6, center=False, scale=False)
        self.linear = layers.Dense(patch_size * patch_size * out_channels, use_bias=True)
        self.adaLN_modulation = tf.keras.Sequential([
            layers.Activation('silu'),
            layers.Dense(2 * hidden_size, use_bias=True)
        ])
        
    def call(self, x, c):
        """Forward pass."""
        shift, scale = tf.split(self.adaLN_modulation(c), 2, axis=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, interpolation_scale=1.0, base_size=1):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # Convert all inputs to float32 numpy arrays
    interpolation_scale = np.float32(interpolation_scale)
    base_size = np.float32(base_size)
    
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)
    
    # Convert grid sizes to float32
    grid_h_size = np.float32(grid_size[0])
    grid_w_size = np.float32(grid_size[1])
    
    # Create grid arrays
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    
    # Perform divisions with float32 values
    grid_h = grid_h / (grid_h_size / base_size) / interpolation_scale
    grid_w = grid_w / (grid_w_size / base_size) / interpolation_scale
    
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim], dtype=np.float32), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Create 2D positional embeddings from a grid.
    
    Args:
        embed_dim: Embedding dimension (must be even)
        grid: 2D grid of positions
        
    Returns:
        Array of shape [H*W, D] containing positional embeddings
    """
    assert embed_dim % 2 == 0

    # Create embeddings for height and width dimensions
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    # Combine height and width embeddings
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Create 1D sinusoidal positional embeddings.
    
    This function generates sinusoidal embeddings for a 1D position array,
    using alternating sine and cosine functions at different frequencies.
    
    Args:
        embed_dim: Output dimension for each position (must be even)
        pos: Array of positions to encode, shape (M,)
        
    Returns:
        Array of shape (M, D) containing positional embeddings
    """
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



class OmniGen(Model):
    """Diffusion model with a Transformer backbone."""
    
    def __init__(
        self,
        transformer_config: Phi3Config,
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

        hidden_size = transformer_config.hidden_size

        self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size)
        self.input_x_embedder = PatchEmbed(patch_size, in_channels, hidden_size)

        self.time_token = TimestepEmbedder(hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        self.pe_interpolation = pe_interpolation
        pos_embed = get_2d_sincos_pos_embed(hidden_size, pos_embed_max_size, interpolation_scale=self.pe_interpolation, base_size=64)
        self.pos_embed = tf.Variable(initial_value=pos_embed[None, ...], trainable=False, dtype=tf.float32)

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        # Ensure the configuration is compatible with the model type
        if transformer_config.model_type != 'phi3':
            raise ValueError(f"Configuration model type {transformer_config.model_type} is incompatible with Phi3 model.")

        # Ensure the positional embedding size matches the configuration
        if self.pos_embed_max_size != transformer_config.pos_embed_max_size:
            raise ValueError(f"Positional embedding max size {self.pos_embed_max_size} does not match configuration {transformer_config.pos_embed_max_size}.")

        self.llm = Phi3Transformer(config=transformer_config)
        self.llm.config.use_cache = False

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights."""
        # Build patch embedders by calling them with a dummy input
        dummy_input = tf.zeros((1, self.pos_embed_max_size, self.pos_embed_max_size, self.in_channels))
        self.x_embedder(dummy_input)
        self.input_x_embedder(dummy_input)

        # Initialize patch embedders with Xavier uniform initialization
        for embedder in [self.x_embedder, self.input_x_embedder]:
            w = embedder.proj.kernel
            w_flat = tf.reshape(w, [w.shape[0] * w.shape[1] * w.shape[2], w.shape[3]])
            limit = tf.sqrt(6 / float(w_flat.shape[0] + w_flat.shape[1]))
            embedder.proj.kernel.assign(tf.random.uniform(w.shape, -limit, limit))
            if embedder.proj.use_bias:
                embedder.proj.bias.assign(tf.zeros_like(embedder.proj.bias))

        # Initialize timestep embedding MLPs with normal distribution
        for embedder in [self.t_embedder, self.time_token]:
            for layer in embedder.mlp.layers:
                if isinstance(layer, layers.Dense):
                    layer.kernel.assign(tf.random.normal(layer.kernel.shape, stddev=0.02))

        # Zero-out output layers
        self.final_layer.adaLN_modulation.layers[-1].kernel.assign(
            tf.zeros_like(self.final_layer.adaLN_modulation.layers[-1].kernel)
        )
        self.final_layer.adaLN_modulation.layers[-1].bias.assign(
            tf.zeros_like(self.final_layer.adaLN_modulation.layers[-1].bias)
        )
        self.final_layer.linear.kernel.assign(tf.zeros_like(self.final_layer.linear.kernel))
        self.final_layer.linear.bias.assign(tf.zeros_like(self.final_layer.linear.bias))

    def unpatchify(self, x, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        B = tf.shape(x)[0]

        x = tf.reshape(x, [B, h//self.patch_size, w//self.patch_size, self.patch_size, self.patch_size, c])
        x = tf.transpose(x, [0, 5, 1, 3, 2, 4])  # [B, C, H', P, W', P]
        imgs = tf.reshape(x, [B, c, h, w])
        return imgs

    def call(self, x, timestep, input_ids=None, input_img_latents=None, input_image_sizes=None, 
            attention_mask=None, position_ids=None, padding_latent=None, past_key_values=None, 
            return_past_key_values=True, training=False):
        """Model forward pass."""
        
        # Embed patches
        x = self.x_embedder(x)
        
        # Add time embeddings
        t = self.t_embedder(timestep)
        x = x + t[:, None, :]
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Process through transformer
        x = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=x,
            use_cache=return_past_key_values,
            training=training
        )
        
        # Apply final layer
        if return_past_key_values:
            x, past_key_values = x
            
        x = self.final_layer(x, t)
        
        if return_past_key_values:
            return x, past_key_values
        return x

    @classmethod
    def from_pretrained(cls, model_name):
        """Load pretrained model."""
        if not os.path.exists(model_name):
            cache_folder = os.getenv('HF_HUB_CACHE')
            model_name = snapshot_download(
                repo_id=model_name,
                cache_dir=cache_folder,
                ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5']
            )
            
        # Load configuration
        config_path = os.path.join(model_name, 'config.json')
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        # Determine configuration type
        if config_dict.get('model_type') == 'phi3':
            config = Phi3Config.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported model type: {config_dict.get('model_type')}")

        model = cls(config)
        
        # Load weights
        if os.path.exists(os.path.join(model_name, 'model.safetensors')):
            print("Loading safetensors")
            weights = load_file(os.path.join(model_name, 'model.safetensors'))
            model.load_weights_from_torch(weights)
        else:
            weights_path = os.path.join(model_name, 'model.h5')
            if not os.path.exists(weights_path):
                raise ValueError(f"No weights found at {weights_path}")
            model.load_weights(weights_path)
            
        return model
