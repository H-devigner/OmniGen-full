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
from safetensors import safe_open
from huggingface_hub import snapshot_download
from diffusers.loaders import PeftAdapterMixin

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
            layers.Dense(hidden_size, use_bias=True, name="mlp.0"),
            layers.Activation('silu'),
            layers.Dense(hidden_size, use_bias=True, name="mlp.2")
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
            layers.Dense(2 * hidden_size, use_bias=True, name="adaLN_modulation.1")
        ])

    def call(self, x, c):
        shift, scale = tf.split(self.adaLN_modulation(c), 2, axis=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class PatchEmbedMR(layers.Layer):
    """2D Image to Patch Embedding."""
    
    def __init__(self, patch_size=2, in_chans=4, embed_dim=768, bias=True):
        super().__init__()
        self.proj = layers.Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            use_bias=bias,
            name="proj"
        )

    def call(self, x):
        x = self.proj(x)
        # NHWC -> NLC (equivalent to PyTorch's NCHW -> NLC)
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
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=hidden_size,
            bias=True
        )
        self.input_x_embedder = PatchEmbedMR(
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=hidden_size,
            bias=True
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

    def cropped_pos_embed(self, height, width):
        """Crops positional embeddings for SD3 compatibility."""
        if self.pos_embed_max_size is None:
            raise ValueError("`pos_embed_max_size` must be set for cropping.")

        height = height // self.patch_size
        width = width // self.patch_size
        
        if height > self.pos_embed_max_size:
            raise ValueError(
                f"Height ({height}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )
        if width > self.pos_embed_max_size:
            raise ValueError(
                f"Width ({width}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )

        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        
        # Reshape and crop
        spatial_pos_embed = tf.reshape(
            self.pos_embed,
            [1, self.pos_embed_max_size, self.pos_embed_max_size, -1]
        )
        spatial_pos_embed = spatial_pos_embed[:, top:top + height, left:left + width, :]
        spatial_pos_embed = tf.reshape(
            spatial_pos_embed,
            [1, -1, tf.shape(spatial_pos_embed)[-1]]
        )
        return spatial_pos_embed

    def patch_multiple_resolutions(self, latents, padding_latent=None, is_input_images=False):
        """Handle multiple resolution inputs."""
        if isinstance(latents, list):
            return_list = False
            if padding_latent is None:
                padding_latent = [None] * len(latents)
                return_list = True
                
            patched_latents, num_tokens, shapes = [], [], []
            for latent, padding in zip(latents, padding_latent):
                height, width = tf.shape(latent)[-2], tf.shape(latent)[-1]
                
                if is_input_images:
                    latent = self.input_x_embedder(latent)
                else:
                    latent = self.x_embedder(latent)
                    
                pos_embed = self.cropped_pos_embed(height, width)
                latent = latent + pos_embed
                
                if padding is not None:
                    latent = tf.concat([latent, padding], axis=-2)
                    
                patched_latents.append(latent)
                num_tokens.append(tf.shape(pos_embed)[1])
                shapes.append([height, width])
                
            if not return_list:
                latents = tf.concat(patched_latents, axis=0)
            else:
                latents = patched_latents
        else:
            height, width = tf.shape(latents)[-2], tf.shape(latents)[-1]
            
            if is_input_images:
                latents = self.input_x_embedder(latents)
            else:
                latents = self.x_embedder(latents)
                
            pos_embed = self.cropped_pos_embed(height, width)
            latents = latents + pos_embed
            num_tokens = tf.shape(latents)[1]
            shapes = [height, width]
            
        return latents, num_tokens, shapes

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
    def from_pretrained(cls, model_name, **kwargs):
        """Load model from pretrained weights."""
        if not os.path.exists(model_name):
            cache_folder = os.getenv('HF_HUB_CACHE')
            model_name = snapshot_download(
                repo_id=model_name,
                cache_dir=cache_folder,
                ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5']
            )
            
        # Load configuration
        config = Phi3Config.from_pretrained(model_name)
        
        # Create model instance
        model = cls(config, **kwargs)
        
        # Debug: Print model structure
        print("\nModel Structure:")
        for var in model.variables:
            print(f"  {var.name} - Shape: {var.shape}")
        
        # Load weights
        weights_path = os.path.join(model_name, "model.safetensors")
        if os.path.exists(weights_path):
            print("\nLoading safetensors weights...")
            
            # Create parameter mapping dictionary
            param_mapping = {
                # Embedders
                't_embedder.mlp.0.weight': 't_embedder/mlp_0/kernel',
                't_embedder.mlp.0.bias': 't_embedder/mlp_0/bias',
                't_embedder.mlp.2.weight': 't_embedder/mlp_2/kernel',
                't_embedder.mlp.2.bias': 't_embedder/mlp_2/bias',
                
                'time_token.mlp.0.weight': 'time_token/mlp_0/kernel',
                'time_token.mlp.0.bias': 'time_token/mlp_0/bias',
                'time_token.mlp.2.weight': 'time_token/mlp_2/kernel',
                'time_token.mlp.2.bias': 'time_token/mlp_2/bias',
                
                # Patch embedders
                'x_embedder.proj.weight': 'x_embedder/proj/kernel',
                'x_embedder.proj.bias': 'x_embedder/proj/bias',
                'input_x_embedder.proj.weight': 'input_x_embedder/proj/kernel',
                'input_x_embedder.proj.bias': 'input_x_embedder/proj/bias',
                
                # Final layer
                'final_layer.norm_final.weight': 'final_layer/norm_final/gamma',
                'final_layer.norm_final.bias': 'final_layer/norm_final/beta',
                'final_layer.linear.weight': 'final_layer/linear/kernel',
                'final_layer.linear.bias': 'final_layer/linear/bias',
                'final_layer.adaLN_modulation.1.weight': 'final_layer/adaLN_modulation_1/kernel',
                'final_layer.adaLN_modulation.1.bias': 'final_layer/adaLN_modulation_1/bias',
                
                # Transformer layers
                'llm.norm.weight': 'transformer/norm/gamma',
                'llm.norm.bias': 'transformer/norm/beta',
                'llm.embed_tokens.weight': 'transformer/embed_tokens/kernel',
            }
            
            # Add transformer layer mappings
            for i in range(32):  # Assuming 32 layers
                layer_prefix = f'llm.layers.{i}'
                tf_prefix = f'transformer/layers_{i}'
                layer_map = {
                    f'{layer_prefix}.input_layernorm.weight': f'{tf_prefix}/input_layernorm/gamma',
                    f'{layer_prefix}.post_attention_layernorm.weight': f'{tf_prefix}/post_attention_layernorm/gamma',
                    f'{layer_prefix}.self_attn.qkv_proj.weight': f'{tf_prefix}/self_attn/qkv_proj/kernel',
                    f'{layer_prefix}.self_attn.o_proj.weight': f'{tf_prefix}/self_attn/o_proj/kernel',
                    f'{layer_prefix}.mlp.gate_up_proj.weight': f'{tf_prefix}/mlp/gate_up_proj/kernel',
                    f'{layer_prefix}.mlp.down_proj.weight': f'{tf_prefix}/mlp/down_proj/kernel',
                }
                param_mapping.update(layer_map)
            
            # Get all model variables
            var_dict = {}
            for var in model.variables:
                name = var.name.split(':')[0]
                var_dict[name] = var
                
            with safe_open(weights_path, framework="tf") as f:
                # Load and assign weights
                for pt_name in f.keys():
                    # Get TensorFlow name
                    tf_name = param_mapping.get(pt_name)
                    if tf_name is None:
                        # Try direct conversion
                        tf_name = pt_name.replace('.', '/')
                        
                    if tf_name in var_dict:
                        tensor = f.get_tensor(pt_name)
                        
                        # Handle special cases
                        if 'kernel' in tf_name and len(tensor.shape) >= 2:
                            if 'conv' in tf_name.lower() or 'proj' in tf_name.lower():
                                # Conv2D weights need HWIO to OIHW conversion
                                tensor = np.transpose(tensor, (2, 3, 1, 0))
                            else:
                                # Dense layers need simple transpose
                                tensor = np.transpose(tensor)
                                
                        try:
                            var_dict[tf_name].assign(tensor)
                            print(f"Assigned {pt_name} -> {tf_name}")
                        except Exception as e:
                            print(f"Error assigning {pt_name} -> {tf_name}: {e}")
                            print(f"Shapes: PyTorch {tensor.shape} vs TF {var_dict[tf_name].shape}")
                    else:
                        print(f"Warning: Parameter {pt_name} not found in model")
                        
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
