"""OmniGen TensorFlow Model Implementation

This module contains the TensorFlow implementation of the OmniGen model, 
which is a diffusion model with a Transformer backbone. The implementation
closely follows the PyTorch version while utilizing TensorFlow-specific optimizations.
"""

import math
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import global_policy

from .transformer import Phi3Config, Phi3Transformer
from .layers import TimestepEmbedder, FinalLayer, PatchEmbedMR
import numpy as np
from typing import Dict, Optional, List, Union
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from diffusers.loaders import PeftAdapterMixin
import json
from dataclasses import fields
import gc


@tf.function(jit_compile=True)
def modulate(x, shift, scale):
    """Apply adaptive layer normalization modulation."""
    return x * (1 + tf.expand_dims(scale, 1)) + tf.expand_dims(shift, 1)


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


class OmniGen(Model):
    """OmniGen model implementation."""
    
    def __init__(
        self,
        transformer_config,
        patch_size=2,
        in_channels=4,
        pe_interpolation=1.0,
        pos_embed_max_size=192,
        **kwargs
    ):
        """Initialize OmniGen model."""
        super().__init__(**kwargs)
        
        self.transformer_config = transformer_config
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.pos_embed_max_size = pos_embed_max_size
        self.pe_interpolation = pe_interpolation
        
        hidden_size = transformer_config.hidden_size
        
        # Initialize components with mixed precision
        self.x_embedder = PatchEmbedMR(patch_size, in_channels, hidden_size, bias=True)
        self.input_x_embedder = PatchEmbedMR(patch_size, in_channels, hidden_size, bias=True)
        
        self.time_token = TimestepEmbedder(hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # Initialize position embeddings
        pos_embed = self._get_2d_sincos_pos_embed(
            hidden_size,
            pos_embed_max_size,
            interpolation_scale=pe_interpolation,
            base_size=64
        )
        self.pos_embed = tf.Variable(
            initial_value=pos_embed,
            trainable=False,
            name="pos_embed"
        )
        
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        
        # Initialize transformer
        self.transformer = Phi3Transformer(config=transformer_config)
        self.transformer.config.use_cache = False
        
        # Initialize weights
        self.initialize_weights()
        
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

    def call(
        self,
        x,
        timestep,
        input_ids,
        attention_mask=None,
        position_ids=None,
        training=False
    ):
        """Forward pass."""
        # Get batch size and sequence length
        batch_size = tf.shape(x)[0]
        
        # Embed timestep
        t_emb = self.time_token(timestep)  # [B, D]
        
        # Patch embed the input
        x = self.x_embedder(x)  # [B, H*W, D]
        
        # Get positional embeddings
        pos_embed = self.get_pos_embed(x.shape[1])  # [1, H*W, D]
        x = x + pos_embed
        
        # Prepare transformer inputs
        hidden_states = x
        
        # Add time token at the beginning
        time_tokens = tf.expand_dims(t_emb, axis=1)  # [B, 1, D]
        hidden_states = tf.concat([time_tokens, hidden_states], axis=1)
        
        # Create attention mask including time token
        if attention_mask is not None:
            # Add attention mask for time token
            time_token_mask = tf.ones((batch_size, 1), dtype=attention_mask.dtype)
            attention_mask = tf.concat([time_token_mask, attention_mask], axis=1)
        
        # Update position IDs to account for time token
        if position_ids is not None:
            time_token_pos = tf.zeros((batch_size, 1), dtype=position_ids.dtype)
            position_ids = tf.concat([time_token_pos, position_ids], axis=1)
        
        # Pass through transformer
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=hidden_states,
            training=training
        )
        
        hidden_states = transformer_outputs[0]
        
        # Remove time token
        hidden_states = hidden_states[:, 1:]
        
        # Final layer
        output = self.final_layer(hidden_states, t_emb)
        
        # Reshape to image
        height = width = int(tf.sqrt(tf.cast(hidden_states.shape[1], tf.float32)))
        output = tf.reshape(output, [batch_size, height, width, self.in_channels])
        
        return output
        
    def get_pos_embed(self, sequence_length):
        """Get position embeddings."""
        if not hasattr(self, "pos_embed"):
            # Initialize position embeddings
            pos_embed = self._get_2d_sincos_pos_embed(
                self.transformer_config.hidden_size,
                int(tf.sqrt(tf.cast(sequence_length, tf.float32)))
            )
            self.pos_embed = tf.Variable(
                initial_value=pos_embed,
                trainable=False,
                name="pos_embed"
            )
        return self.pos_embed[:, :sequence_length]
        
    def _get_2d_sincos_pos_embed(self, embed_dim, grid_size):
        """Generate 2D sinusoidal position embeddings."""
        grid_h = tf.range(grid_size, dtype=tf.float32)
        grid_w = tf.range(grid_size, dtype=tf.float32)
        grid = tf.stack(tf.meshgrid(grid_w, grid_h, indexing='xy'), axis=0)
        grid = tf.reshape(grid, [2, 1, grid_size, grid_size])
        
        emb_h = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        
        pos_embed = tf.concat([emb_h, emb_w], axis=-1)
        return tf.expand_dims(pos_embed, 0)
        
    def _get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        """Generate 1D sinusoidal position embeddings."""
        omega = tf.exp(
            -math.log(10000) * tf.range(embed_dim // 2, dtype=tf.float32) / embed_dim
        )
        
        pos = tf.cast(tf.reshape(pos, [-1]), tf.float32)
        out = tf.einsum('m,d->md', pos, omega)
        
        emb_sin = tf.sin(out)
        emb_cos = tf.cos(out)
        
        pos_embed = tf.concat([emb_sin, emb_cos], axis=-1)
        return pos_embed

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
                pos_embed = tf.reshape(pos_embed, [1, -1, self.transformer_config.hidden_size])
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
            pos_embed = tf.reshape(pos_embed, [1, -1, self.transformer_config.hidden_size])
            pos_embed = pos_embed[:, :num_patches, :]  # Only use as many position embeddings as patches
            latents = latents + pos_embed
            
            num_tokens = tf.shape(latents)[1]
            return latents, num_tokens, [(orig_h, orig_w)]
            
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
        output = self.transformer.transformer(
            hidden_states,
            attention_mask=combined_attention,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            training=training
        )
        
        if isinstance(output, tuple):
            output = output[0]
            
        return output

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
