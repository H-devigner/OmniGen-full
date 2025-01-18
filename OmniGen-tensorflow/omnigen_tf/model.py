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


@tf.function(jit_compile=True)
def modulate(x, shift, scale):
    """Apply adaptive layer normalization modulation."""
    return x * (1 + tf.expand_dims(scale, 1)) + tf.expand_dims(shift, 1)


class TimestepEmbedder(layers.Layer):
    """Embeds scalar timesteps into vector representations."""
    
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=tf.float32):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = tf.keras.Sequential([
            layers.Dense(hidden_size, use_bias=True, dtype=dtype),
            layers.Activation('silu'),
            layers.Dense(hidden_size, use_bias=True, dtype=dtype)
        ])

    @tf.function(reduce_retracing=True)
    def timestep_embedding(self, t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings efficiently."""
        half = dim // 2
        freqs = tf.exp(
            -math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half
        )
        args = tf.cast(t[:, None], tf.float32) * freqs[None]
        embedding = tf.concat([tf.cos(args), tf.sin(args)], axis=-1)
        if dim % 2:
            embedding = tf.pad(embedding, [[0, 0], [0, 1]])
        return tf.cast(embedding, self.dtype)

    @tf.function(reduce_retracing=True)
    def call(self, t):
        if len(tf.shape(t)) == 0:
            t = t[None]
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class PatchEmbed(layers.Layer):
    """2D Image to Patch Embedding."""
    
    def __init__(self, embed_dim=768, patch_size=16, in_channels=3, dtype=tf.float32, **kwargs):
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
            dtype=dtype,
            name='proj'
        )
        
    def call(self, x):
        """Forward pass."""
        # Handle NHWC format
        x = tf.cast(x, self.dtype)  # Cast input to float16
        x = self.proj(x)  # Shape: [B, H', W', C]
        
        # Reshape to [B, H*W, C]
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, -1, self.embed_dim])
        
        return x
        
    def get_num_patches(self, h, w):
        """Get number of patches for given input dimensions."""
        return (h // self.patch_size) * (w // self.patch_size)


class FinalLayer(layers.Layer):
    """Final layer for image generation."""
    
    def __init__(self, patch_size, in_channels, embed_dim, dtype=tf.float32, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Initialize layers with proper dtype
        self.norm_final = layers.LayerNormalization(
            epsilon=1e-6, 
            center=False, 
            scale=False, 
            dtype=dtype
        )
        self.proj = layers.Dense(
            patch_size * patch_size * in_channels,
            dtype=dtype,
            name="proj"
        )
        self.adaLN_modulation = tf.keras.Sequential([
            layers.Activation('silu'),
            layers.Dense(2 * embed_dim, dtype=dtype)
        ])
        
    @tf.function(reduce_retracing=True)
    def call(self, x, time_emb):
        """Forward pass with efficient memory usage."""
        # Apply AdaLN modulation
        shift, scale = tf.split(self.adaLN_modulation(time_emb), 2, axis=-1)
        x = self.norm_final(x)
        x = x * (1 + tf.expand_dims(scale, 1)) + tf.expand_dims(shift, 1)
        x = self.proj(x)
        return x


class OmniGen(Model):
    """OmniGen model implementation."""
    
    def __init__(
        self,
        transformer_config,
        patch_size=16,
        in_channels=4,
        pe_interpolation='bicubic',
        pos_embed_max_size=1024,
        chunk_size=128,
        enable_checkpointing=False,
        **kwargs
    ):
        """Initialize model."""
        # Set compute dtype to float16 for mixed precision
        kwargs['dtype'] = tf.float16
        super().__init__(**kwargs)
        
        # Set default chunk size if not provided
        self.chunk_size = chunk_size
        self.enable_checkpointing = enable_checkpointing
        
        # Initialize transformer with config
        if not isinstance(transformer_config, Phi3Config):
            transformer_config = Phi3Config(**transformer_config)
            
        self.transformer = Phi3Transformer(transformer_config)
        self.transformer_config = transformer_config
        
        # Save configuration
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.pe_interpolation = pe_interpolation
        self.pos_embed_max_size = pos_embed_max_size
        
        # Initialize components with float16
        self.x_embedder = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=transformer_config.hidden_size,
            dtype=tf.float16
        )
        
        self.input_x_embedder = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=transformer_config.hidden_size,
            dtype=tf.float16
        )
        
        # Initialize timestep embedders
        self.time_token = TimestepEmbedder(
            hidden_size=transformer_config.hidden_size,
            dtype=tf.float16
        )
        
        self.t_embedder = TimestepEmbedder(
            hidden_size=transformer_config.hidden_size,
            dtype=tf.float16
        )
        
        self.final_layer = FinalLayer(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=transformer_config.hidden_size,
            dtype=tf.float16
        )
        
        # Initialize positional embeddings efficiently
        pos_embed = get_2d_sincos_pos_embed(
            transformer_config.hidden_size,
            pos_embed_max_size,
            interpolation_scale=1.0,
            base_size=64
        )
        self.pos_embed = tf.Variable(
            initial_value=tf.expand_dims(tf.cast(pos_embed, tf.float16), 0),
            trainable=False,
            name="pos_embed"
        )
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights efficiently."""
        # Initialize patch embedders
        for embedder in [self.x_embedder, self.input_x_embedder]:
            w = embedder.proj.kernel
            tf.keras.initializers.GlorotUniform()(w.shape).assign(w)
            tf.keras.initializers.Zeros()(embedder.proj.bias.shape).assign(embedder.proj.bias)
        
        # Initialize timestep embedders
        for embedder in [self.time_token, self.t_embedder]:
            for layer in embedder.mlp.layers:
                if isinstance(layer, layers.Dense):
                    tf.keras.initializers.RandomNormal(stddev=0.02)(layer.kernel.shape).assign(layer.kernel)
        
        # Zero-out final layer efficiently
        for layer in self.final_layer.adaLN_modulation.layers:
            if isinstance(layer, layers.Dense):
                tf.keras.initializers.Zeros()(layer.kernel.shape).assign(layer.kernel)
                if layer.bias is not None:
                    tf.keras.initializers.Zeros()(layer.bias.shape).assign(layer.bias)
        
        tf.keras.initializers.Zeros()(self.final_layer.proj.kernel.shape).assign(self.final_layer.proj.kernel)
        tf.keras.initializers.Zeros()(self.final_layer.proj.bias.shape).assign(self.final_layer.proj.bias)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def unpatchify(self, x, h, w):
        """Efficient unpatchify operation."""
        c = self.in_channels
        batch_size = tf.shape(x)[0]
        
        # Reshape efficiently
        x = tf.reshape(x, [
            batch_size,
            h // self.patch_size,
            w // self.patch_size,
            self.patch_size,
            self.patch_size,
            c
        ])
        
        # Use efficient transpose
        x = tf.transpose(x, [0, 5, 1, 3, 2, 4])
        imgs = tf.reshape(x, [batch_size, c, h, w])
        imgs = tf.transpose(imgs, [0, 2, 3, 1])  # NCHW -> NHWC
        return imgs

    @tf.function(reduce_retracing=True)
    def _forward(self, latents, timestep, input_ids, attention_mask=None, training=False):
        """Memory-efficient forward pass."""
        batch_size = tf.shape(latents)[0]
        h, w = tf.shape(latents)[1], tf.shape(latents)[2]
        
        # Process inputs efficiently
        x = self.x_embedder(latents)
        
        # Get time embeddings
        t = tf.fill([batch_size], timestep)
        time_token = self.time_token(t)
        time_emb = self.t_embedder(t)
        
        # Get text embeddings efficiently
        text_embeds = self.transformer.wte(input_ids)
        text_embeds = tf.cast(text_embeds, self.dtype)
        text_embeds = tf.repeat(text_embeds, batch_size, axis=0)
        
        # Combine embeddings efficiently
        hidden_states = tf.concat([
            text_embeds,
            tf.expand_dims(time_token, 1),
            x
        ], axis=1)
        
        # Handle attention mask efficiently
        if attention_mask is not None:
            attention_mask = tf.repeat(attention_mask, batch_size, axis=0)
            image_attention = tf.ones((batch_size, tf.shape(x)[1]), dtype=attention_mask.dtype)
            combined_attention = tf.concat([attention_mask, image_attention], axis=1)
        
        # Run transformer efficiently
        output = self.transformer.transformer(
            hidden_states,
            attention_mask=combined_attention,
            training=training
        )
        
        if isinstance(output, tuple):
            output = output[0]
        
        # Process final layer efficiently
        num_tokens = tf.shape(x)[1]
        image_embedding = output[:, -num_tokens:]
        x = self.final_layer(image_embedding, time_emb)
        
        # Unpatchify efficiently
        return self.unpatchify(x, h, w)

    def _chunked_forward(self, latents, timestep, input_ids, attention_mask=None, training=False):
        """Forward pass with chunking for memory efficiency."""
        # Process in chunks
        chunk_size = self.chunk_size
        chunks = tf.shape(latents)[1] // chunk_size + (1 if tf.shape(latents)[1] % chunk_size != 0 else 0)
        
        outputs = []
        for i in range(chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, tf.shape(latents)[1])
            chunk = latents[:, start_idx:end_idx]
            
            # Process chunk
            chunk_output = self._forward(chunk, timestep, input_ids, attention_mask, training)
            outputs.append(chunk_output)
            
        # Combine chunks
        return tf.concat(outputs, axis=1)
        
    def call(
        self,
        latents,
        timestep,
        input_ids=None,
        attention_mask=None,
        training=False,
    ):
        """Model forward pass."""
        # Handle list inputs
        if isinstance(latents, list):
            latents = tf.concat(latents, axis=0)
            
        # Use chunked forward if needed
        if tf.shape(latents)[1] > self.chunk_size:
            return self._chunked_forward(latents, timestep, input_ids, attention_mask, training)
        else:
            return self._forward(latents, timestep, input_ids, attention_mask, training)
            
    def decode(self, latents):
        """Decode latents to image."""
        # Add decoding logic here
        return latents  # Placeholder for now

    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'transformer_config': self.transformer_config.to_dict(),
            'patch_size': self.patch_size,
            'in_channels': self.in_channels,
            'pe_interpolation': self.pe_interpolation,
            'pos_embed_max_size': self.pos_embed_max_size,
            'chunk_size': self.chunk_size,
            'enable_checkpointing': self.enable_checkpointing,
        })
        return config
        
    @classmethod
    def from_config(cls, config):
        """Create model from configuration."""
        # Extract transformer config
        transformer_config = config.pop('transformer_config', None)
        if transformer_config is not None:
            transformer_config = Phi3Config(**transformer_config)
            
        # Create model
        return cls(transformer_config=transformer_config, **config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "OmniGen":
        """Load pretrained model."""
        # Get model path
        model_path = pretrained_model_name_or_path
        if not os.path.exists(model_path):
            cache_folder = os.getenv('HF_HUB_CACHE')
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
            
        # Update config with any provided kwargs
        transformer_config = kwargs.pop('transformer_config', {})
        config_dict.update(transformer_config)
        
        # Create config
        config = Phi3Config(**config_dict)
        
        # Create model
        model = cls(transformer_config=config, **kwargs)
        
        # Load weights
        weights_file = os.path.join(model_path, "model.safetensors")
        if os.path.exists(weights_file):
            model.load_weights_from_safetensors(weights_file)
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
