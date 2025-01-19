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
import torch
import torch.nn as nn

from omnigen_tf.transformer import Phi3Config, Phi3Transformer

# Conversion functions
def tf_to_torch(tensor):
    """Convert TensorFlow tensor to PyTorch tensor."""
    if isinstance(tensor, (list, tuple)):
        return [tf_to_torch(t) for t in tensor]
    if tensor is None:
        return None
    return torch.from_numpy(tensor.numpy())

def torch_to_tf(tensor):
    """Convert PyTorch tensor to TensorFlow tensor."""
    if isinstance(tensor, (list, tuple)):
        return [torch_to_tf(t) for t in tensor]
    if tensor is None:
        return None
    return tf.convert_to_tensor(tensor.detach().cpu().numpy())


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
        super().__init__()
        
        # Initialize transformer config
        if not isinstance(transformer_config, Phi3Config):
            transformer_config = Phi3Config(**transformer_config)
            
        # Save configuration
        self.transformer_config = transformer_config
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.pe_interpolation = pe_interpolation
        self.pos_embed_max_size = pos_embed_max_size
        self.chunk_size = chunk_size
        self.enable_checkpointing = enable_checkpointing
        
        # Initialize TensorFlow components
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
        
        # Initialize positional embeddings
        pos_embed = get_2d_sincos_pos_embed(
            transformer_config.hidden_size,
            pos_embed_max_size,
            interpolation_scale=1.0,
            base_size=64
        )
        self.pos_embed = tf.constant(
            tf.expand_dims(tf.cast(pos_embed, tf.float16), 0),
            dtype=tf.float16
        )
        
        # Initialize PyTorch transformer
        self.pytorch_transformer = Phi3Transformer(
            config=transformer_config
        )
        self.pytorch_transformer.config.use_cache = False
        
        # Enable GPU memory growth
        for device in tf.config.list_physical_devices('GPU'):
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except:
                pass
                
        # Enable mixed precision
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    def _convert_to_pytorch(self, tensor):
        """Convert TensorFlow tensor to PyTorch tensor."""
        if tensor is None:
            return None
        if isinstance(tensor, (list, tuple)):
            return [self._convert_to_pytorch(t) for t in tensor]
            
        # Detach from computation graph and move to CPU
        numpy_tensor = tensor.numpy()
        torch_tensor = torch.from_numpy(numpy_tensor)
        
        # Clear the numpy array
        del numpy_tensor
        
        return torch_tensor.contiguous()  # Ensure memory is contiguous

    def _convert_to_tensorflow(self, tensor):
        """Convert PyTorch tensor to TensorFlow tensor."""
        if tensor is None:
            return None
        if isinstance(tensor, (list, tuple)):
            return [self._convert_to_tensorflow(t) for t in tensor]
            
        # Detach from computation graph and move to CPU before conversion
        numpy_tensor = tensor.detach().cpu().numpy()
        tf_tensor = tf.convert_to_tensor(numpy_tensor)
        
        # Clear the numpy array
        del numpy_tensor
        
        return tf_tensor

    @tf.function(jit_compile=True)
    def _forward(self, latents, timestep, input_ids, input_img_latents=None, input_image_sizes=None, attention_mask=None, position_ids=None, padding_latent=None, past_key_values=None, return_past_key_values=True, training=False):
        """Memory-efficient forward pass using PyTorch transformer."""
        # Process latents
        input_is_list = isinstance(latents, list)
        x, num_tokens, shapes = self.patch_multiple_resolutions(latents, padding_latent)
        time_token = self.time_token(timestep, dtype=tf.float16)
        time_token = tf.expand_dims(time_token, 1)

        # Process input images if provided
        if input_img_latents is not None:
            input_latents, _, _ = self.patch_multiple_resolutions(input_img_latents, is_input_images=True)
            input_latents_torch = self._convert_to_pytorch(input_latents)
            del input_latents  # Free memory
        else:
            input_latents_torch = None

        # Convert to PyTorch tensors efficiently
        x_torch = self._convert_to_pytorch(x)
        del x  # Free memory
        time_token_torch = self._convert_to_pytorch(time_token)
        del time_token  # Free memory
        input_ids_torch = self._convert_to_pytorch(input_ids) if input_ids is not None else None
        attention_mask_torch = self._convert_to_pytorch(attention_mask) if attention_mask is not None else None
        position_ids_torch = self._convert_to_pytorch(position_ids) if position_ids is not None else None
        past_key_values_torch = self._convert_to_pytorch(past_key_values) if past_key_values is not None else None

        # Clear TensorFlow memory
        tf.keras.backend.clear_session()
        gc.collect()

        # Process through PyTorch transformer
        with torch.no_grad():
            # Handle conditional embeddings
            if input_ids_torch is not None:
                condition_embeds = self.pytorch_transformer.embed_tokens(input_ids_torch).clone()
                if input_latents_torch is not None:
                    input_img_inx = 0
                    for b_inx in input_image_sizes.keys():
                        for start_inx, end_inx in input_image_sizes[b_inx]:
                            condition_embeds[b_inx, start_inx:end_inx] = input_latents_torch[input_img_inx]
                            input_img_inx += 1
                    assert input_img_inx == len(input_latents_torch)
                    del input_latents_torch  # Free memory
                input_emb = torch.cat([condition_embeds, time_token_torch, x_torch], dim=1)
                del condition_embeds  # Free memory
            else:
                input_emb = torch.cat([time_token_torch, x_torch], dim=1)

            del time_token_torch, x_torch  # Free memory

            # Forward pass through transformer
            outputs = self.pytorch_transformer(
                inputs_embeds=input_emb,
                attention_mask=attention_mask_torch,
                position_ids=position_ids_torch,
                past_key_values=past_key_values_torch,
                use_cache=return_past_key_values,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                offload_model=False
            )

            del input_emb  # Free memory
            if attention_mask_torch is not None:
                del attention_mask_torch
            if position_ids_torch is not None:
                del position_ids_torch
            if past_key_values_torch is not None:
                del past_key_values_torch

        # Extract outputs and convert back to TensorFlow
        output = self._convert_to_tensorflow(outputs.last_hidden_state)
        past_key_values = self._convert_to_tensorflow(outputs.past_key_values) if return_past_key_values else None

        del outputs  # Free memory
        torch.cuda.empty_cache()

        # Process through final layer
        if input_is_list:
            image_embedding = output[:, -tf.reduce_max(num_tokens):]
            time_emb = self.t_embedder(timestep, dtype=tf.float16)
            x = self.final_layer(image_embedding, time_emb)
            del image_embedding, time_emb  # Free memory

            latents_out = []
            for i in range(tf.shape(x)[0]):
                latent = x[i:i+1, :num_tokens[i]]
                latent = self.unpatchify(latent, shapes[i][0], shapes[i][1])
                latents_out.append(latent)
            output = latents_out
            del x  # Free memory
        else:
            image_embedding = output[:, -num_tokens:]
            time_emb = self.t_embedder(timestep, dtype=tf.float16)
            x = self.final_layer(image_embedding, time_emb)
            del image_embedding, time_emb  # Free memory
            output = self.unpatchify(x, shapes[0], shapes[1])
            del x  # Free memory

        # Final cleanup
        tf.keras.backend.clear_session()
        gc.collect()

        if return_past_key_values:
            return output, past_key_values
        return output

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

    def call(self, latents, timestep, input_ids, input_img_latents=None, input_image_sizes=None, attention_mask=None, position_ids=None, padding_latent=None, past_key_values=None, return_past_key_values=True, training=False):
        """Main call method with memory optimization."""
        return self._forward(
            latents=latents,
            timestep=timestep,
            input_ids=input_ids,
            input_img_latents=input_img_latents,
            input_image_sizes=input_image_sizes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            padding_latent=padding_latent,
            past_key_values=past_key_values,
            return_past_key_values=return_past_key_values,
            training=training
        )
        
    @tf.function(jit_compile=True)
    def forward_with_cfg(self, x, timestep, input_ids, input_img_latents, input_image_sizes, attention_mask, position_ids, cfg_scale, use_img_cfg, img_cfg_scale, past_key_values, use_kv_cache, offload_model=False):      
        """Memory-efficient forward pass with classifier-free guidance."""
        # Set transformer cache mode
        self.pytorch_transformer.config.use_cache = use_kv_cache
        
        # Forward pass
        model_out, past_key_values = self._forward(
            x, timestep, input_ids, input_img_latents, input_image_sizes, 
            attention_mask, position_ids, past_key_values=past_key_values, 
            return_past_key_values=True, offload_model=offload_model
        )
        
        # Handle classifier-free guidance
        if use_img_cfg:
            # Split into conditional, unconditional, and image conditional
            cond, uncond, img_cond = tf.split(model_out, num_or_size_splits=3, axis=0)
            
            # Apply guidance scales
            cond = uncond + img_cfg_scale * (img_cond - uncond) + cfg_scale * (cond - img_cond)
            model_out = tf.concat([cond, cond, cond], axis=0)
            
            # Clean up intermediate tensors
            del cond, uncond, img_cond
        else:
            # Split into conditional and unconditional
            cond, uncond = tf.split(model_out, num_or_size_splits=2, axis=0)
            
            # Apply guidance scale
            cond = uncond + cfg_scale * (cond - uncond)
            model_out = tf.concat([cond, cond], axis=0)
            
            # Clean up intermediate tensors
            del cond, uncond
        
        # Clear caches
        tf.keras.backend.clear_session()
        gc.collect()
        
        return model_out, past_key_values

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
