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


class FinalLayer(layers.Layer):
    """The final layer of DiT."""
    
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, 
            center=False, 
            scale=False,
            name="norm_final"
        )
        self.linear = layers.Dense(
            patch_size * patch_size * out_channels,
            use_bias=True,
            name="linear"
        )
        self.adaLN_modulation = tf.keras.Sequential([
            layers.Activation('silu'),
            layers.Dense(2 * hidden_size, use_bias=True, name="adaLN_modulation_1")
        ])

    def call(self, x, c):
        shift, scale = tf.split(self.adaLN_modulation(c), 2, axis=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class PatchEmbedMR(layers.Layer):
    """2D Image to Patch Embedding."""
    
    def __init__(self, embed_dim=768, patch_size=16, in_chans=3, **kwargs):
        """Initialize patch embedding layer."""
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.in_chans = in_chans
        
        # Initialize projection layer
        self.proj = tf.keras.layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding='valid',
            data_format='channels_last',  # NHWC format
            name='proj'
        )
        
    def call(self, x):
        """Forward pass."""
        # Input should be in NHWC format
        x = self.proj(x)
        # Reshape to (batch, sequence_length, channels)
        x = tf.reshape(x, [tf.shape(x)[0], -1, tf.shape(x)[-1]])
        return x


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
        
        # Model configuration
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.pos_embed_max_size = pos_embed_max_size
        self.pe_interpolation = pe_interpolation
        
        hidden_size = transformer_config.hidden_size
        
        # Initialize embedders
        self.x_embedder = PatchEmbedMR(
            embed_dim=hidden_size,
            patch_size=patch_size,
            in_chans=in_channels
        )
        self.input_x_embedder = PatchEmbedMR(
            embed_dim=hidden_size,
            patch_size=patch_size,
            in_chans=in_channels
        )
        
        self.time_token = TimestepEmbedder(hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # Create positional embeddings
        pos_embed = get_2d_sincos_pos_embed(
            hidden_size, 
            pos_embed_max_size,
            interpolation_scale=self.pe_interpolation,
            base_size=64
        )
        self.pos_embed = tf.Variable(
            initial_value=pos_embed[None],
            trainable=False,
            name="pos_embed"
        )
        
        self.final_layer = FinalLayer(
            hidden_size=hidden_size,
            patch_size=patch_size,
            out_channels=self.out_channels
        )
        
        self.transformer = Phi3Transformer(
            config=transformer_config
        )
        # Disable caching to match PyTorch
        self.transformer.config.use_cache = False
        
        # Create weights mapping
        self._create_weights_mapping()
        
    def _create_weights_mapping(self):
        """Create mapping between PyTorch and TensorFlow weight names."""
        self.weights_map = {}
        
        # Transformer mappings
        for i in range(self.transformer.config.num_hidden_layers):
            # Attention layers
            self.weights_map.update({
                f"transformer/layer_{i}/self_attn/q_proj/kernel": f"transformer.h.{i}.attn.q_proj.weight",
                f"transformer/layer_{i}/self_attn/k_proj/kernel": f"transformer.h.{i}.attn.k_proj.weight",
                f"transformer/layer_{i}/self_attn/v_proj/kernel": f"transformer.h.{i}.attn.v_proj.weight",
                f"transformer/layer_{i}/self_attn/o_proj/kernel": f"transformer.h.{i}.attn.o_proj.weight",
                # Layer norms
                f"transformer/layer_{i}/input_layernorm/gamma": f"transformer.h.{i}.ln_1.weight",
                f"transformer/layer_{i}/input_layernorm/beta": f"transformer.h.{i}.ln_1.bias",
                f"transformer/layer_{i}/post_attention_layernorm/gamma": f"transformer.h.{i}.ln_2.weight",
                f"transformer/layer_{i}/post_attention_layernorm/beta": f"transformer.h.{i}.ln_2.bias",
                # MLP
                f"transformer/layer_{i}/mlp/gate_up_proj/kernel": f"transformer.h.{i}.mlp.gate_up_proj.weight",
                f"transformer/layer_{i}/mlp/down_proj/kernel": f"transformer.h.{i}.mlp.down_proj.weight",
            })
        
        # Final layer norm
        self.weights_map.update({
            "transformer/ln_f/gamma": "transformer.ln_f.weight",
            "transformer/ln_f/beta": "transformer.ln_f.bias",
        })
        
        # Embeddings
        self.weights_map.update({
            "transformer/wte/embeddings": "transformer.wte.weight",
        })
        
        # Timestep embedder mappings
        self.weights_map.update({
            "time_token/mlp_0/kernel": "time_token.mlp.0.weight",
            "time_token/mlp_0/bias": "time_token.mlp.0.bias",
            "time_token/mlp_2/kernel": "time_token.mlp.2.weight",
            "time_token/mlp_2/bias": "time_token.mlp.2.bias",
            "t_embedder/mlp_0/kernel": "t_embedder.mlp.0.weight",
            "t_embedder/mlp_0/bias": "t_embedder.mlp.0.bias",
            "t_embedder/mlp_2/kernel": "t_embedder.mlp.2.weight",
            "t_embedder/mlp_2/bias": "t_embedder.mlp.2.bias",
        })
        
        # Patch embedder mappings
        self.weights_map.update({
            "x_embedder/proj/kernel": "x_embedder.proj.weight",
            "x_embedder/proj/bias": "x_embedder.proj.bias",
            "input_x_embedder/proj/kernel": "input_x_embedder.proj.weight",
            "input_x_embedder/proj/bias": "input_x_embedder.proj.bias",
        })
        
        # Final layer mappings
        self.weights_map.update({
            "final_layer/norm_final/gamma": "final_layer.norm_final.weight",
            "final_layer/norm_final/beta": "final_layer.norm_final.bias",
            "final_layer/linear/kernel": "final_layer.linear.weight",
            "final_layer/linear/bias": "final_layer.linear.bias",
            "final_layer/adaLN_modulation_1/kernel": "final_layer.adaLN_modulation.1.weight",
            "final_layer/adaLN_modulation_1/bias": "final_layer.adaLN_modulation.1.bias",
        })
        
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
        for embedder in [self.x_embedder, self.input_x_embedder]:
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
        for embedder in [self.t_embedder, self.time_token]:
            for layer in embedder.mlp.layers:
                if isinstance(layer, layers.Dense):
                    layer.kernel.assign(tf.random.normal(
                        layer.kernel.shape, mean=0.0, stddev=std
                    ))

        # Zero-out output layers
        self.final_layer.adaLN_modulation.layers[-1].kernel.assign(
            tf.zeros_like(self.final_layer.adaLN_modulation.layers[-1].kernel)
        )
        self.final_layer.adaLN_modulation.layers[-1].bias.assign(
            tf.zeros_like(self.final_layer.adaLN_modulation.layers[-1].bias)
        )
        self.final_layer.linear.kernel.assign(
            tf.zeros_like(self.final_layer.linear.kernel)
        )
        self.final_layer.linear.bias.assign(
            tf.zeros_like(self.final_layer.linear.bias)
        )

    @tf.function(jit_compile=True)
    def unpatchify(self, x, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
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
                channels = tf.shape(x)[3]
                
                # Apply embedding (keeping NHWC format)
                if is_input_images:
                    x = self.input_x_embedder(x)
                else:
                    x = self.x_embedder(x)
                    
                # Now reshape to (batch, sequence_length, channels)
                x = tf.reshape(x, (-1, height * width, x.shape[-1]))
                
                # Add position embeddings
                pos_embed = self.get_pos_embed(height, width)
                x = x + pos_embed
                
                all_latents.append(x)
                all_num_tokens.append(tf.shape(x)[1])
                all_shapes.append((height, width))
                
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
            channels = tf.shape(latents)[3]
            
            # Apply embedding (keeping NHWC format)
            if is_input_images:
                latents = self.input_x_embedder(latents)
            else:
                latents = self.x_embedder(latents)
                
            # Now reshape to (batch, sequence_length, channels)
            latents = tf.reshape(latents, (-1, height * width, latents.shape[-1]))
            
            # Add position embeddings
            pos_embed = self.get_pos_embed(height, width)
            latents = latents + pos_embed
            
            num_tokens = tf.shape(latents)[1]
            return latents, num_tokens, [(height, width)]
            
    def get_pos_embed(self, height, width):
        """Get position embeddings for given dimensions."""
        # Convert to concrete values if tensors
        if isinstance(height, tf.Tensor):
            height = tf.cast(height, tf.int32)
        if isinstance(width, tf.Tensor):
            width = tf.cast(width, tf.int32)
            
        # Check dimensions
        if height > self.pos_embed_max_size or width > self.pos_embed_max_size:
            # Instead of raising error, resize to max size
            scale_factor = tf.minimum(
                self.pos_embed_max_size / height,
                self.pos_embed_max_size / width
            )
            height = tf.cast(tf.cast(height, tf.float32) * scale_factor, tf.int32)
            width = tf.cast(tf.cast(width, tf.float32) * scale_factor, tf.int32)
            
        # Get base position embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.transformer.config.hidden_size,
            int(self.pos_embed_max_size),
            cls_token=False,
            base_size=self.pos_embed_max_size
        )
        pos_embed = tf.convert_to_tensor(pos_embed, dtype=tf.float32)
        
        # Interpolate to target size
        if height != self.pos_embed_max_size or width != self.pos_embed_max_size:
            pos_embed = tf.reshape(pos_embed, (self.pos_embed_max_size, self.pos_embed_max_size, -1))
            pos_embed = tf.image.resize(
                pos_embed[tf.newaxis],
                (height, width),
                method='bilinear'
            )[0]
            
        pos_embed = tf.reshape(pos_embed, (height * width, -1))
        return pos_embed

    @tf.function(jit_compile=True)
    def call(self, x, timestep, input_ids=None, input_img_latents=None,
            input_image_sizes=None, attention_mask=None, position_ids=None,
            padding_latent=None, past_key_values=None, return_past_key_values=True,
            offload_model=False, training=False):
        """Forward pass of the model."""
        input_is_list = isinstance(x, list)
        x, num_tokens, shapes = self.patch_multiple_resolutions(x, padding_latent)
        
        # Get time token
        time_token = self.time_token(timestep)
        time_token = tf.expand_dims(time_token, 1)
        
        if input_img_latents is not None:
            input_latents, _, _ = self.patch_multiple_resolutions(
                input_img_latents, is_input_images=True
            )
            
        if input_ids is not None:
            condition_embeds = self.transformer.embed_tokens(input_ids)
            input_img_inx = 0
            
            for b_inx in input_image_sizes.keys():
                for start_inx, end_inx in input_image_sizes[b_inx]:
                    condition_embeds = tf.tensor_scatter_nd_update(
                        condition_embeds,
                        [[b_inx, i] for i in range(start_inx, end_inx)],
                        input_latents[input_img_inx]
                    )
                    input_img_inx += 1
                    
            if input_img_latents is not None:
                tf.debugging.assert_equal(
                    input_img_inx,
                    len(input_latents),
                    "Number of input images doesn't match"
                )
            
            input_emb = tf.concat([condition_embeds, time_token, x], axis=1)
        else:
            input_emb = tf.concat([time_token, x], axis=1)
            
        # Run through transformer
        output = self.transformer(
            inputs_embeds=input_emb,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            training=training
        )
        
        if return_past_key_values:
            output, past_key_values = output.last_hidden_state, output.past_key_values
        else:
            output = output.last_hidden_state
            
        if input_is_list:
            image_embedding = output[:, -tf.reduce_max(num_tokens):]
            time_emb = self.t_embedder(timestep)
            x = self.final_layer(image_embedding, time_emb)
            
            latents = []
            for i in range(tf.shape(x)[0]):
                latent = x[i:i+1, :num_tokens[i]]
                latent = self.unpatchify(latent, shapes[i][0], shapes[i][1])
                latents.append(latent)
        else:
            image_embedding = output[:, -num_tokens:]
            time_emb = self.t_embedder(timestep)
            x = self.final_layer(image_embedding, time_emb)
            latents = self.unpatchify(x, shapes[0], shapes[1])
            
        if return_past_key_values:
            return latents, past_key_values
        return latents

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs) -> "OmniGen":
        """Load pretrained model."""
        config = None
        model_path = pretrained_model_name_or_path
        
        # Download from hub if needed
        if not os.path.exists(pretrained_model_name_or_path):
            cache_folder = os.getenv('HF_HUB_CACHE')
            model_path = snapshot_download(
                repo_id=pretrained_model_name_or_path,
                cache_dir=cache_folder,
                ignore_patterns=['*.bin', '*.msgpack', '*.ot', '*.h5']
            )
            
        # Load config
        config_file = os.path.join(model_path, "config.json")
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config_dict = json.load(f)
                
                # Create config object with the loaded dictionary
                config = Phi3Config(**config_dict)
        else:
            # Use default config
            config = Phi3Config()
            
        # Create model
        model = cls(config, *model_args, **kwargs)
        
        # Load weights
        weights_file = os.path.join(model_path, "model.safetensors")
        if os.path.exists(weights_file):
            print("Loading safetensors weights...")
            from safetensors.torch import load_file
            
            # Load state dict
            state_dict = load_file(weights_file)
            
            # Convert weights to TensorFlow format
            tf_weights = {}
            for pt_name, param in state_dict.items():
                # Get corresponding TF name
                tf_name = model.weights_map.get(pt_name)
                if tf_name is not None:
                    # Convert tensor to numpy array
                    param_np = param.numpy()
                    tf_weights[tf_name] = param_np
            
            # Load weights into model
            model.set_weights([tf_weights[w.name] for w in model.trainable_weights if w.name in tf_weights])
            print("Weights loaded successfully!")
        else:
            print(f"No weights file found at {weights_file}")
            
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
