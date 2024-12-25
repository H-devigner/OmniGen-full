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
        
        # Initialize components
        self.x_embedder = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=transformer_config.hidden_size
        )
        
        self.time_token = TimeToken(
            embed_dim=transformer_config.hidden_size
        )
        
        self.final_layer = FinalLayer(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=transformer_config.hidden_size
        )
        
        # Memory optimization flags
        self.memory_efficient = False
        self.gradient_checkpointing = False
        
        # Create weights mapping
        self._create_weights_mapping()
        
    def enable_memory_efficient_inference(self, chunk_size=None):
        """Enable memory efficient inference."""
        if chunk_size is not None:
            self.chunk_size = chunk_size
        self.memory_efficient = True
        self.gradient_checkpointing = True
        self.transformer.enable_gradient_checkpointing()
        
    def disable_memory_efficient_inference(self):
        """Disable memory efficient inference mode."""
        self.memory_efficient = False
        self.gradient_checkpointing = False
        self.transformer.disable_gradient_checkpointing()
        
    def _create_weights_mapping(self):
        """Create mapping between PyTorch and TensorFlow weight names."""
        self.weights_map = {}
        
        # Transformer mappings
        self.weights_map.update({
            "transformer/wte/embeddings": "transformer.wte.weight",
            "transformer/norm/gamma": "transformer.norm.weight",
            "transformer/norm/beta": "transformer.norm.bias",
        })
        
        # Layer mappings
        for i in range(self.transformer_config.num_hidden_layers):
            layer_map = {
                f"transformer/layer_{i}/input_layernorm/gamma": f"transformer.layers.{i}.input_layernorm.weight",
                f"transformer/layer_{i}/input_layernorm/beta": f"transformer.layers.{i}.input_layernorm.bias",
                f"transformer/layer_{i}/self_attn/qkv_proj/kernel": f"transformer.layers.{i}.self_attn.qkv_proj.weight",
                f"transformer/layer_{i}/self_attn/qkv_proj/bias": f"transformer.layers.{i}.self_attn.qkv_proj.bias",
                f"transformer/layer_{i}/self_attn/o_proj/kernel": f"transformer.layers.{i}.self_attn.o_proj.weight",
                f"transformer/layer_{i}/self_attn/o_proj/bias": f"transformer.layers.{i}.self_attn.o_proj.bias",
                f"transformer/layer_{i}/post_attention_layernorm/gamma": f"transformer.layers.{i}.post_attention_layernorm.weight",
                f"transformer/layer_{i}/post_attention_layernorm/beta": f"transformer.layers.{i}.post_attention_layernorm.bias",
                f"transformer/layer_{i}/mlp/gate_up_proj/kernel": f"transformer.layers.{i}.mlp.gate_up_proj.weight",
                f"transformer/layer_{i}/mlp/gate_up_proj/bias": f"transformer.layers.{i}.mlp.gate_up_proj.bias",
                f"transformer/layer_{i}/mlp/down_proj/kernel": f"transformer.layers.{i}.mlp.down_proj.weight",
                f"transformer/layer_{i}/mlp/down_proj/bias": f"transformer.layers.{i}.mlp.down_proj.bias",
            }
            self.weights_map.update(layer_map)
            
        # Other components mappings
        self.weights_map.update({
            "x_embedder/proj/kernel": "x_embedder.proj.weight",
            "x_embedder/proj/bias": "x_embedder.proj.bias",
            "time_token/mlp_0/kernel": "time_token.mlp.0.weight",
            "time_token/mlp_0/bias": "time_token.mlp.0.bias",
            "time_token/mlp_2/kernel": "time_token.mlp.2.weight",
            "time_token/mlp_2/bias": "time_token.mlp.2.bias",
            "final_layer/proj/kernel": "final_layer.proj.weight",
            "final_layer/proj/bias": "final_layer.proj.bias",
        })

    def load_weights_from_safetensors(self, weights_file):
        """Load weights from safetensors file."""
        print("Loading safetensors weights...")
        from safetensors.torch import load_file
        
        # Load state dict
        state_dict = load_file(weights_file)
        
        # Convert weights to TensorFlow format
        tf_weights = {}
        for pt_name, param in state_dict.items():
            # Get corresponding TF name
            tf_name = self.weights_map.get(pt_name)
            if tf_name is not None:
                # Convert tensor to numpy array
                param_np = param.numpy()
                tf_weights[tf_name] = param_np
                
        # Load weights into model
        for w in self.trainable_weights:
            if w.name in tf_weights:
                w.assign(tf_weights[w.name])
                
        print("Weights loaded successfully!")

    def initialize_weights(self):
        """Initialize model weights to match PyTorch implementation."""
        # Initialize transformer layers
        def _basic_init(module):
            if isinstance(module, layers.Dense):
                # Xavier uniform initialization
                limit = tf.sqrt(6. / float(module.input_dim + module.units))
                module.kernel.assign(tf.random.uniform(
                    module.kernel.shape, -limit, limit
                ))
                if module.use_bias:
                    module.bias.assign(tf.zeros_like(module.bias))
                    
        for layer in self.layers:
            if hasattr(layer, 'kernel'):
                _basic_init(layer)
        
        # Initialize patch_embed like Dense (instead of Conv2D)
        for embedder in [self.x_embedder]:
            w = embedder.proj.kernel
            # Reshape to 2D for Xavier initialization
            w_flat = tf.reshape(w, [w.shape[0], -1])
            limit = tf.sqrt(6. / float(w_flat.shape[0] + w_flat.shape[1]))
            w_init = tf.random.uniform(w_flat.shape, -limit, limit)
            # Reshape back to 4D
            embedder.proj.kernel.assign(tf.reshape(w_init, w.shape))
            embedder.proj.bias.assign(tf.zeros_like(embedder.proj.bias))

        # Initialize timestep embedding MLP with normal distribution
        std = 0.02
        for embedder in [self.time_token]:
            for layer in embedder.mlp.layers:
                if isinstance(layer, layers.Dense):
                    layer.kernel.assign(tf.random.normal(
                        layer.kernel.shape, mean=0.0, stddev=std
                    ))

        # Zero-out output layers
        self.final_layer.proj.kernel.assign(
            tf.zeros_like(self.final_layer.proj.kernel)
        )
        self.final_layer.proj.bias.assign(
            tf.zeros_like(self.final_layer.proj.bias)
        )

    @tf.function(jit_compile=True)
    def unpatchify(self, x, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.in_channels
        batch_size = tf.shape(x)[0]
        
        # Reshape to match PyTorch's dimensions
        x = tf.reshape(x, [
            batch_size,
            h // self.patch_size,
            w // self.patch_size,
            self.patch_size,
            self.patch_size,
            c
        ])
        
        # Equivalent to PyTorch's einsum('nhwpqc->nchpwq')
        x = tf.transpose(x, [0, 5, 1, 3, 2, 4])
        
        # Final reshape to get output shape
        imgs = tf.reshape(x, [batch_size, c, h, w])
        return imgs

    def patch_multiple_resolutions(self, latents, padding_latent=None, is_input_images=False):
        """Process input latents with multiple resolutions."""
        if isinstance(latents, list):
            # Handle list of latents
            all_latents = []
            all_num_tokens = []
            all_shapes = []
            
            for x in latents:
                height = tf.shape(x)[1]
                width = tf.shape(x)[2]
                
                # Store original shape for position embeddings
                orig_h, orig_w = height, width
                
                # Apply embedding (keeping NHWC format)
                if is_input_images:
                    x = self.x_embedder(x)  # Returns [B, N, C]
                else:
                    x = self.x_embedder(x)  # Returns [B, N, C]
                
                # Calculate number of patches
                num_patches = (height // self.patch_size) * (width // self.patch_size)
                
                # Add position embeddings
                pos_embed = self.get_pos_embed(orig_h, orig_w)
                pos_embed = tf.reshape(pos_embed, [1, -1, self.transformer.config.hidden_size])
                pos_embed = pos_embed[:, :num_patches, :]  # Only use as many position embeddings as patches
                x = x + pos_embed
                
                all_latents.append(x)
                all_num_tokens.append(tf.shape(x)[1])
                all_shapes.append((orig_h, orig_w))
                
            # Pad and concatenate
            max_tokens = tf.reduce_max(all_num_tokens)
            padded_latents = []
            
            for x, num_tokens in zip(all_latents, all_num_tokens):
                if num_tokens < max_tokens:
                    padding = tf.zeros((tf.shape(x)[0], max_tokens - num_tokens, tf.shape(x)[-1]))
                    x = tf.concat([x, padding], axis=1)
                padded_latents.append(x)
                
            latents = tf.concat(padded_latents, axis=0)
            return latents, all_num_tokens, all_shapes
            
        else:
            # Handle single latent
            height = tf.shape(latents)[1]
            width = tf.shape(latents)[2]
            
            # Store original shape for position embeddings
            orig_h, orig_w = height, width
            
            # Apply embedding (keeping NHWC format)
            if is_input_images:
                latents = self.x_embedder(latents)  # Returns [B, N, C]
            else:
                latents = self.x_embedder(latents)  # Returns [B, N, C]
            
            # Calculate number of patches
            num_patches = (height // self.patch_size) * (width // self.patch_size)
            
            # Add position embeddings
            pos_embed = self.get_pos_embed(orig_h, orig_w)
            pos_embed = tf.reshape(pos_embed, [1, -1, self.transformer.config.hidden_size])
            pos_embed = pos_embed[:, :num_patches, :]  # Only use as many position embeddings as patches
            latents = latents + pos_embed
            
            num_tokens = tf.shape(latents)[1]
            return latents, num_tokens, [(orig_h, orig_w)]
            
    def get_pos_embed(self, height, width):
        """Get position embeddings."""
        # Convert to patches
        height = height // self.patch_size
        width = width // self.patch_size
        
        # Get base position embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.transformer.config.hidden_size,
            height,
            width
        )
        pos_embed = tf.convert_to_tensor(pos_embed, dtype=tf.float32)
        return pos_embed

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
        
    def _forward(self, latents, timestep, input_ids, attention_mask=None, training=False):
        """Forward pass without chunking."""
        batch_size = tf.shape(latents)[0]
        
        # Get input shape
        shapes = tf.shape(latents)
        
        # Process inputs
        x = self.x_embedder(latents)  # Shape: [batch_size, num_patches, hidden_size]
        
        # Get position embeddings for actual spatial dimensions
        h, w = tf.cast(shapes[1], tf.int32), tf.cast(shapes[2], tf.int32)
        pos_embed = self.get_pos_embed(h, w)
        
        # Add time embedding
        # Expand timestep to match batch size
        t = tf.fill([batch_size], timestep)
        time_embed = self.time_token(t)
        
        # Combine embeddings with time embedding
        x = x + tf.expand_dims(time_embed, axis=1)  # Add time embedding to each position
        
        # Get text embeddings from input_ids and expand to match batch size
        text_embeds = self.transformer.wte(input_ids)  # Shape: [1, seq_len, hidden_size]
        text_embeds = tf.repeat(text_embeds, batch_size, axis=0)  # Shape: [batch_size, seq_len, hidden_size]
        
        # Combine image and text embeddings
        hidden_states = tf.concat([text_embeds, x], axis=1)
        
        # Create combined attention mask if needed
        if attention_mask is not None:
            # Expand attention mask to match batch size
            attention_mask = tf.repeat(attention_mask, batch_size, axis=0)
            # Create attention mask for image tokens (all 1s)
            image_attention = tf.ones((batch_size, tf.shape(x)[1]), dtype=attention_mask.dtype)
            # Combine text and image attention masks
            combined_attention = tf.concat([attention_mask, image_attention], axis=1)
        else:
            combined_attention = None
        
        # Run through transformer
        output = self.transformer(
            hidden_states=hidden_states,
            attention_mask=combined_attention,
            training=training
        )
        
        if isinstance(output, tuple):
            output = output[0]
            
        return output

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
