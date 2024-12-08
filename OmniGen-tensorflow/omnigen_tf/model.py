"""OmniGen model implementation in TensorFlow."""
from __future__ import annotations

# Standard library imports
import os
import json
import math
from typing import Optional, Dict, Any, Union, List, Tuple, Callable, TypeVar, TYPE_CHECKING
import logging

# Third-party imports
import tensorflow as tf
from tensorflow.keras import layers, initializers
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
import onnx
from onnx_tf.backend import prepare

# Local imports
from .converter import WeightConverter
from .peft import PeftAdapterMixin
from .transformer import Phi3TransformerTF
from .export_model import download_and_convert

# Type definitions
TensorType = TypeVar('TensorType', bound=tf.Tensor)

def modulate(x, shift, scale):
    """Modulate layer norm output."""
    return x * (1 + tf.expand_dims(scale, 1)) + tf.expand_dims(shift, 1)

class TimestepEmbedder(tf.keras.layers.Layer):
    """Embeds scalar timesteps into vector representations."""
    
    def __init__(
        self,
        hidden_size,
        frequency_embedding_size=256,
        kernel_initializer=None,
        bias_initializer=None,
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.frequency_embedding_size = frequency_embedding_size
        
        # Initialize with proper initializers
        self.mlp = tf.keras.Sequential([
            layers.Dense(
                hidden_size,
                use_bias=True,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name=f'{name}_fc1' if name else 'fc1'
            ),
            layers.Activation('swish'),
            layers.Dense(
                hidden_size,
                use_bias=True,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name=f'{name}_fc2' if name else 'fc2'
            )
        ], name=f'{name}_mlp' if name else 'mlp')
        
    def call(self, t: tf.Tensor) -> tf.Tensor:
        """Forward pass."""
        # Create sinusoidal embeddings
        half_dim = self.frequency_embedding_size // 2
        emb = tf.math.log(10000.0) / (half_dim - 1)
        emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
        emb = tf.cast(t, dtype=tf.float32)[:, None] * emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=1)
        
        # Map to hidden_size
        return self.mlp(emb)

class FinalLayer(tf.keras.layers.Layer):
    """The final layer of OmniGen."""
    
    def __init__(
        self,
        hidden_size,
        patch_size,
        out_channels,
        kernel_initializer=None,
        bias_initializer=None,
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.norm_final = layers.LayerNormalization(
            epsilon=1e-6,
            center=False,
            scale=False,
            name=f'{name}_norm_final' if name else 'norm_final'
        )
        self.linear = layers.Dense(
            patch_size * patch_size * out_channels,
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name=f'{name}_linear' if name else 'linear'
        )
        self.adaLN_modulation = tf.keras.Sequential([
            layers.Activation('swish'),
            layers.Dense(
                2 * hidden_size,
                use_bias=True,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name=f'{name}_adaLN_modulation' if name else 'adaLN_modulation'
            )
        ], name=f'{name}_adaLN_seq' if name else 'adaLN_seq')
        
    def call(self, x: tf.Tensor, c: tf.Tensor) -> tf.Tensor:
        """Forward pass."""
        shift, scale = tf.split(self.adaLN_modulation(c), 2, axis=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class PatchEmbedMR(tf.keras.layers.Layer):
    """2D Image to Patch Embedding."""
    
    def __init__(
        self,
        patch_size=2,
        in_chans=4,
        embed_dim=768,
        bias=True,
        kernel_initializer=None,
        bias_initializer=None,
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.proj = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            use_bias=bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            data_format='channels_last',
            name=f'{name}_proj' if name else 'proj'
        )
        
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Forward pass."""
        B, H, W, C = x.shape
        x = self.proj(x)  # B, H//2, W//2, embed_dim
        return x

class OmniGenTF(tf.keras.Model, PeftAdapterMixin):
    """OmniGen model in TensorFlow."""
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__()
        
        # Model configuration
        self.hidden_size = config.get('hidden_size', 768)
        self.patch_size = 2
        self.in_channels = 4
        self.out_channels = 4
        self.pos_embed_max_size = 192
        self.pe_interpolation = 1.0
        
        # Create embedders with proper initialization
        kernel_init = initializers.TruncatedNormal(stddev=0.02)
        
        self.x_embedder = PatchEmbedMR(
            self.patch_size, 
            self.in_channels,
            self.hidden_size,
            kernel_initializer=kernel_init,
            bias_initializer='zeros',
            name='x_embedder'
        )
        
        self.input_x_embedder = PatchEmbedMR(
            self.patch_size,
            self.in_channels,
            self.hidden_size,
            kernel_initializer=kernel_init,
            bias_initializer='zeros',
            name='input_x_embedder'
        )
        
        # Time embeddings
        self.time_token = TimestepEmbedder(
            self.hidden_size,
            kernel_initializer=kernel_init,
            bias_initializer='zeros',
            name='time_token'
        )
        
        self.t_embedder = TimestepEmbedder(
            self.hidden_size,
            kernel_initializer=kernel_init,
            bias_initializer='zeros',
            name='t_embedder'
        )
        
        # Position embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            self.pos_embed_max_size,
            interpolation_scale=self.pe_interpolation,
            base_size=64
        )
        self.pos_embed = tf.Variable(
            initial_value=pos_embed[None],
            trainable=False,
            dtype=tf.float32,
            name='pos_embed'
        )
        
        # Final layer
        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            kernel_initializer=kernel_init,
            bias_initializer='zeros',
            name='final_layer'
        )
        
        # Initialize Phi3 transformer
        self.llm = Phi3TransformerTF(config)
        self.llm.use_cache = False
        
    def call(
        self,
        x: tf.Tensor,
        timestep: tf.Tensor,
        input_ids: Optional[tf.Tensor] = None,
        input_img_latents: Optional[tf.Tensor] = None,
        input_image_sizes: Optional[List[Tuple[int, int]]] = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        padding_latent: Optional[tf.Tensor] = None,
        past_key_values: Optional[List[Tuple[tf.Tensor]]] = None,
        return_past_key_values: bool = True,
        offload_model: bool = False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, List[Tuple[tf.Tensor]]]]:
        """Forward pass."""
        # Get batch size and device
        batch_size = tf.shape(x)[0]
        
        # Process input through patch embedding
        x = self.patch_multiple_resolutions(x, padding_latent)
        
        # Get time embeddings
        t = self.t_embedder(timestep)
        
        # Get position embeddings
        pos_embed = tf.repeat(self.pos_embed, batch_size, axis=0)
        
        # Combine embeddings
        h = x + pos_embed
        
        # Add time token
        time_token = self.time_token(timestep)
        h = tf.concat([time_token[:, None, :], h], axis=1)
        
        # Process through transformer
        if return_past_key_values:
            h, present_key_values = self.llm(
                h,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True
            )
        else:
            h = self.llm(
                h,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=False
            )
            present_key_values = None
            
        # Remove time token
        h = h[:, 1:]
        
        # Get output size
        H = W = int(math.sqrt(tf.shape(h)[1]))
        
        # Final layer
        h = self.final_layer(h, t)
        
        # Reshape output
        h = tf.reshape(h, [-1, H, W, self.patch_size, self.patch_size, self.out_channels])
        h = tf.transpose(h, [0, 1, 3, 2, 4, 5])
        h = tf.reshape(h, [-1, H * self.patch_size, W * self.patch_size, self.out_channels])
        
        if return_past_key_values:
            return h, present_key_values
        return h
        
    def forward_with_cfg(
        self,
        x: tf.Tensor,
        timestep: tf.Tensor,
        input_ids: Optional[tf.Tensor] = None,
        input_img_latents: Optional[tf.Tensor] = None,
        input_image_sizes: Optional[List[Tuple[int, int]]] = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        cfg_scale: float = 3.0,
        use_img_cfg: bool = True,
        img_cfg_scale: float = 1.6,
        past_key_values: Optional[List[Tuple[tf.Tensor]]] = None,
        use_kv_cache: bool = True,
        offload_model: bool = False
    ) -> tf.Tensor:
        """Forward with classifier-free guidance."""
        # Get batch size
        half = tf.shape(x)[0] // 2
        
        # Split input for classifier-free guidance
        x_cond, x_uncond = tf.split(x, 2)
        
        # Forward pass for conditional
        if use_kv_cache:
            out_cond, present_key_values = self.call(
                x_cond,
                timestep,
                input_ids=input_ids,
                input_img_latents=input_img_latents,
                input_image_sizes=input_image_sizes,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                return_past_key_values=True,
                offload_model=offload_model
            )
        else:
            out_cond = self.call(
                x_cond,
                timestep,
                input_ids=input_ids,
                input_img_latents=input_img_latents,
                input_image_sizes=input_image_sizes,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                return_past_key_values=False,
                offload_model=offload_model
            )
            present_key_values = None
            
        # Forward pass for unconditional
        out_uncond = self.call(
            x_uncond,
            timestep,
            input_ids=None,
            input_img_latents=None,
            input_image_sizes=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            return_past_key_values=False,
            offload_model=offload_model
        )
        
        # Apply classifier-free guidance
        out = out_uncond + cfg_scale * (out_cond - out_uncond)
        
        if use_kv_cache:
            return out, present_key_values
        return out
        
    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_path: str,
            use_mixed_precision: bool = False,
            **kwargs
        ):
        """Load pretrained model.
        
        Args:
            pretrained_model_path: Path to pretrained model or model identifier from huggingface.co
            use_mixed_precision: Whether to use mixed precision
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            OmniGenTF: Loaded model
        """
        # Try to load as ONNX model first
        try:
            # Download and convert ONNX model
            tf_model_path = download_and_convert(model_name=pretrained_model_path)
            
            # Load the converted TensorFlow model
            model = tf.saved_model.load(tf_model_path)
            return model
            
        except Exception as e:
            logging.warning(f"Failed to load ONNX model, falling back to weight conversion: {e}")
            
            # Fall back to original weight conversion method
            if os.path.isdir(pretrained_model_path):
                config_path = os.path.join(pretrained_model_path, "config.json")
            else:
                cache_dir = snapshot_download(pretrained_model_path)
                config_path = os.path.join(cache_dir, "config.json")

            with open(config_path) as f:
                config = json.load(f)

            model = cls(config, **kwargs)

            # Set mixed precision if requested
            if use_mixed_precision:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)

            # Load weights
            try:
                if os.path.isdir(pretrained_model_path):
                    safetensors_path = os.path.join(pretrained_model_path, "model.safetensors")
                    pytorch_path = os.path.join(pretrained_model_path, "pytorch_model.bin")
                else:
                    cache_dir = snapshot_download(pretrained_model_path)
                    safetensors_path = os.path.join(cache_dir, "model.safetensors")
                    pytorch_path = os.path.join(cache_dir, "pytorch_model.bin")

                if os.path.exists(safetensors_path):
                    state_dict = load_file(safetensors_path)
                else:
                    state_dict = torch.load(pytorch_path)

                converter = WeightConverter(state_dict)
                converter.convert_weights(model)

            except Exception as e:
                raise ValueError(f"Error loading weights: {e}")

            return model

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, interpolation_scale=1.0, base_size=1):
    """Generate 2D sinusoidal positional embeddings."""
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / interpolation_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Generate 2D positional embeddings from a grid."""
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Generate 1D positional embeddings."""
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb
