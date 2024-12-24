"""OmniGen TensorFlow Model Implementation

This module contains the TensorFlow implementation of the OmniGen model, 
which is a diffusion model with a Transformer backbone. The implementation
closely follows the PyTorch version while utilizing TensorFlow-specific optimizations.
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from typing import Dict, Optional, List, Union
from safetensors import safe_open
from huggingface_hub import snapshot_download
from diffusers.loaders import PeftAdapterMixin

from omnigen_tf.transformer import Phi3Config, Phi3Transformer


@tf.function(jit_compile=True)
def modulate(x, shift, scale):
    """Apply adaptive layer normalization modulation.
    
    Args:
        x: Input tensor to modulate
        shift: Shift parameter for modulation
        scale: Scale parameter for modulation
    
    Returns:
        Modulated tensor with same shape as input
    """
    # Use float16 for GPU operations
    if tf.config.list_physical_devices('GPU'):
        x = tf.cast(x, tf.float16)
        shift = tf.cast(shift, tf.float16)
        scale = tf.cast(scale, tf.float16)
    else:
        x = tf.cast(x, tf.float32)
        shift = tf.cast(shift, tf.float32)
        scale = tf.cast(scale, tf.float32)
    return x * (1.0 + scale[:, tf.newaxis]) + shift[:, tf.newaxis]


class TimestepEmbedder(Model):
    """Embeds scalar timesteps into vector representations."""
    
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        # Use float16 for GPU operations
        dtype = tf.float16 if tf.config.list_physical_devices('GPU') else tf.float32
        
        self.mlp = tf.keras.Sequential([
            layers.Dense(hidden_size, use_bias=True, dtype=dtype),
            layers.Activation('silu', dtype=dtype),
            layers.Dense(hidden_size, use_bias=True, dtype=dtype)
        ])
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    @tf.function(jit_compile=True)
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings with XLA optimization."""
        half = dim // 2
        freqs = tf.exp(
            -tf.math.log(tf.cast(max_period, t.dtype)) * 
            tf.range(half, dtype=t.dtype) / tf.cast(half, t.dtype)
        )
        args = tf.cast(t[:, None], dtype=freqs.dtype) * freqs[None]
        embedding = tf.concat([tf.cos(args), tf.sin(args)], axis=-1)
        if dim % 2:
            embedding = tf.concat([embedding, tf.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    @tf.function(jit_compile=True)
    def call(self, t):
        # Use appropriate precision based on device
        dtype = tf.float16 if tf.config.list_physical_devices('GPU') else tf.float32
        t_freq = tf.cast(self.timestep_embedding(t, self.frequency_embedding_size), dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class FinalLayer(layers.Layer):
    """The final layer of the DiT model."""
    
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        # Use float16 for GPU operations
        dtype = tf.float16 if tf.config.list_physical_devices('GPU') else tf.float32
        
        self.norm_final = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=dtype)
        self.linear = layers.Dense(patch_size * patch_size * out_channels, 
                                 use_bias=True, dtype=dtype)
        self.adaLN_modulation = tf.keras.Sequential([
            layers.Activation('silu', dtype=dtype),
            layers.Dense(2 * hidden_size, use_bias=True, dtype=dtype)
        ])

    @tf.function(jit_compile=True)
    def call(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, axis=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class PatchEmbedMR(layers.Layer):
    """2D Image to Patch Embedding layer."""
    
    def __init__(self, patch_size=2, in_chans=4, embed_dim=768, bias=True):
        super().__init__()
        # Use float16 for GPU operations
        dtype = tf.float16 if tf.config.list_physical_devices('GPU') else tf.float32
        
        self.proj = layers.Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            use_bias=bias,
            name='patch_embed',
            dtype=dtype
        )

    @tf.function(jit_compile=True)
    def call(self, x):
        # Ensure input is in correct precision
        x = tf.cast(x, self.proj.dtype)
        return self.proj(x)


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
        
        # Enable mixed precision by default for GPU
        if tf.config.list_physical_devices('GPU'):
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Model configuration
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.pos_embed_max_size = pos_embed_max_size
        self.pe_interpolation = pe_interpolation
        
        hidden_size = transformer_config.hidden_size
        
        # Initialize embedders with float16 for GPU
        dtype = tf.float16 if tf.config.list_physical_devices('GPU') else tf.float32
        
        self.x_embedder = PatchEmbedMR(
            patch_size, in_channels, hidden_size, dtype=dtype
        )
        self.input_x_embedder = PatchEmbedMR(
            patch_size, in_channels, hidden_size, dtype=dtype
        )
        
        self.time_token = TimestepEmbedder(hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # Create positional embeddings with proper precision
        with tf.device('/CPU:0'):
            pos_embed = get_2d_sincos_pos_embed(
                hidden_size, pos_embed_max_size,
                interpolation_scale=self.pe_interpolation,
                base_size=64
            )
            self.pos_embed = tf.cast(
                tf.convert_to_tensor(pos_embed[None]), 
                dtype=dtype
            )
        
        self.final_layer = FinalLayer(
            hidden_size, patch_size, self.out_channels
        )
        
        self.transformer = Phi3Transformer(
            config=transformer_config
        )

    @tf.function(jit_compile=True)
    def call(self, x, timestep, input_ids=None, input_img_latents=None,
             input_image_sizes=None, attention_mask=None, position_ids=None,
             training=False):
        # Cast inputs to appropriate precision
        dtype = tf.float16 if tf.config.list_physical_devices('GPU') else tf.float32
        x = tf.cast(x, dtype)
        
        # Process inputs
        with tf.device('/CPU:0'):
            # Handle embeddings on CPU
            x_patches = self.x_embedder(x)
            t_emb = self.t_embedder(timestep)
            
            if input_img_latents is not None:
                input_img_patches = self.input_x_embedder(input_img_latents)
        
        # Move to GPU for main computation if available
        if tf.config.list_physical_devices('GPU'):
            with tf.device('/GPU:0'):
                return self._process_on_gpu(
                    x_patches, t_emb, input_ids, input_img_patches,
                    input_image_sizes, attention_mask, position_ids,
                    training
                )
        else:
            return self._process_on_cpu(
                x_patches, t_emb, input_ids, input_img_patches,
                input_image_sizes, attention_mask, position_ids,
                training
            )

    @tf.function(jit_compile=True)
    def _process_on_gpu(self, x_patches, t_emb, input_ids, input_img_patches,
                       input_image_sizes, attention_mask, position_ids, training):
        # Process with float16 on GPU
        hidden_states = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=x_patches,
            training=training
        )
        
        # Final processing
        output = self.final_layer(hidden_states, t_emb)
        
        # Clear GPU memory
        tf.keras.backend.clear_session()
        
        return output

    def _process_on_cpu(self, x_patches, t_emb, input_ids, input_img_patches,
                       input_image_sizes, attention_mask, position_ids, training):
        # Process with float32 on CPU
        hidden_states = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=x_patches,
            training=training
        )
        
        return self.final_layer(hidden_states, t_emb)

    def unpatchify(self, x, h, w):
        """Convert a sequence of patch embeddings back to an image.
        
        Args:
            x: Input tensor
            h: Output image height
            w: Output image width
            
        Returns:
            Tensor of shape [N, H, W, C] containing the reconstructed image
        """
        c = self.in_channels
        p = self.patch_size
        
        x = tf.reshape(x, [-1, h // p, w // p, p * p * c])
        x = tf.nn.depth_to_space(x, p)  # Equivalent to torch.pixel_shuffle
        return x

    def cropped_pos_embed(self, height, width):
        """Crop positional embeddings for the given image size.
        
        This ensures compatibility with different input sizes by either:
        1. Using the stored embeddings if sizes match
        2. Generating new embeddings for the specific size
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            Positional embeddings of appropriate size
        """
        h, w = height // self.patch_size, width // self.patch_size
        pos_embed = self.pos_embed
        
        if h * w != pos_embed.shape[1]:
            pos_embed = get_2d_sincos_pos_embed(
                self.transformer.config.hidden_size,
                (h, w),
                interpolation_scale=self.pe_interpolation,
                base_size=64
            )
            pos_embed = tf.cast(tf.convert_to_tensor(pos_embed[None]), self.pos_embed.dtype)
        
        return pos_embed

    def patch_multiple_resolutions(self, latents, padding_latent=None, is_input_images=False):
        """Process latents of multiple resolutions.
        
        This method handles input images or latents of different sizes by:
        1. Embedding each resolution separately
        2. Computing appropriate positional embeddings
        3. Optionally handling padding latents
        
        Args:
            latents: List of input tensors at different resolutions
            padding_latent: Optional padding latent to append
            is_input_images: Whether inputs are images (True) or latents (False)
            
        Returns:
            Tuple of (embedded_latents, positional_embeddings)
        """
        embedder = self.input_x_embedder if is_input_images else self.x_embedder
        batch_size = tf.shape(latents[0])[0]
        
        # Process each resolution
        x_list = []
        pos_embed_list = []
        num_tokens_list = []
        shapes_list = []
        
        for idx, x in enumerate(latents):
            h, w = tf.shape(x)[1], tf.shape(x)[2]
            x_embed = embedder(x)
            pos_embed = self.cropped_pos_embed(h, w)
            
            if padding_latent is not None and idx == len(latents) - 1:
                padding_embed = embedder(padding_latent)
                x_embed = tf.concat([x_embed, padding_embed], axis=1)
                pos_embed = tf.concat([pos_embed, pos_embed], axis=1)
            
            x_list.append(x_embed)
            pos_embed_list.append(pos_embed)
            num_tokens_list.append(tf.shape(x_embed)[1])
            shapes_list.append((h, w))
        
        return x_list, num_tokens_list, shapes_list

    @tf.function(jit_compile=True)
    def forward_with_cfg(self, x, timestep, input_ids=None, input_img_latents=None,
                        input_image_sizes=None, attention_mask=None, position_ids=None,
                        cfg_scale=7.5, use_img_cfg=False, img_cfg_scale=1.0):
        """Classifier-free guidance matching PyTorch implementation."""
        # Process with guidance
        model_out = self.call(
            x, timestep, input_ids, input_img_latents,
            input_image_sizes, attention_mask, position_ids
        )
        
        if use_img_cfg:
            # Split for image-based guidance
            cond, uncond, img_cond = tf.split(model_out, 3, axis=0)
            cond = uncond + img_cfg_scale * (img_cond - uncond) + cfg_scale * (cond - img_cond)
            model_out = [cond, cond, cond]
        else:
            # Standard guidance
            cond, uncond = tf.split(model_out, 2, axis=0)
            cond = uncond + cfg_scale * (cond - uncond)
            model_out = [cond, cond]
            
        return tf.concat(model_out, axis=0)

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        """Load model from pretrained weights.
        
        Args:
            model_name: Name or path of pretrained model
            **kwargs: Additional arguments to pass to model constructor
            
        Returns:
            Initialized model with pretrained weights
        """
        # Download model if needed
        if not os.path.exists(model_name):
            print(f"Downloading model from {model_name}...")
            model_name = snapshot_download(model_name)
            print(f"Downloaded model to {model_name}")
            
        # Load configuration
        config_path = os.path.join(model_name, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Config not found at {config_path}")
            
        config = Phi3Config.from_pretrained(model_name)
        
        # Create model instance
        model = cls(config, **kwargs)
        
        # Load weights
        weights_path = os.path.join(model_name, "model.safetensors")
        if os.path.exists(weights_path):
            print("Loading safetensors weights...")
            with safe_open(weights_path, framework="tf") as f:
                state_dict = {key: tf.convert_to_tensor(f.get_tensor(key)) for key in f.keys()}
        else:
            # Try PyTorch weights
            pt_path = os.path.join(model_name, "model.pt")
            if not os.path.exists(pt_path):
                raise ValueError(f"No weights found at {weights_path} or {pt_path}")
                
            print("Loading PyTorch weights...")
            import torch
            state_dict = torch.load(pt_path, map_location="cpu")
            state_dict = {k: tf.convert_to_tensor(v.numpy()) for k, v in state_dict.items()}
            
        # Set model weights
        model.set_weights([state_dict[key] for key in model.weights])
        print("Model loaded successfully!")
        
        return model


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
