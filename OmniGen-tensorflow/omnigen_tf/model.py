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


def modulate(x, shift, scale):
    """Apply adaptive layer normalization modulation.
    
    Args:
        x: Input tensor to modulate
        shift: Shift parameter for modulation
        scale: Scale parameter for modulation
    
    Returns:
        Modulated tensor with same shape as input
    """
    # Ensure all constants and tensors are the same dtype
    x = tf.cast(x, tf.float32)
    shift = tf.cast(shift, tf.float32)
    scale = tf.cast(scale, tf.float32)
    return x * (1.0 + scale[:, tf.newaxis]) + shift[:, tf.newaxis]


class TimestepEmbedder(Model):
    """Embeds scalar timesteps into vector representations.
    
    This module creates sinusoidal embeddings for the diffusion timesteps,
    similar to positional embeddings in transformers but for temporal positions.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=tf.float32):
        super().__init__()
        self.mlp = tf.keras.Sequential([
            layers.Dense(hidden_size, use_bias=True, dtype=dtype),
            layers.Activation('silu', dtype=dtype),  # Using silu to match PyTorch's SiLU
            layers.Dense(hidden_size, use_bias=True, dtype=dtype)
        ])
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        Args:
            t: a 1-D Tensor of N indices, one per batch element.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.
        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = tf.exp(
            -math.log(max_period) * tf.range(half, dtype=tf.float32) / half
        )
        args = tf.cast(t[:, None], dtype=tf.float32) * freqs[None]
        embedding = tf.concat([tf.cos(args), tf.sin(args)], axis=-1)
        if dim % 2:
            embedding = tf.concat([embedding, tf.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def call(self, t, dtype=tf.float32):
        t_freq = tf.cast(self.timestep_embedding(t, self.frequency_embedding_size), dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class FinalLayer(layers.Layer):
    """The final layer of the DiT model.
    
    This layer applies adaptive layer normalization followed by a linear projection
    to produce the output image. It uses modulation based on the timestep embedding
    to condition the normalization.
    
    Args:
        hidden_size: Size of the hidden dimension
        patch_size: Size of image patches
        out_channels: Number of output channels
    """
    def __init__(self, hidden_size, patch_size, out_channels, dtype=tf.float32):
        super().__init__()
        self.norm_final = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.linear = layers.Dense(patch_size * patch_size * out_channels, use_bias=True)
        self.adaLN_modulation = tf.keras.Sequential([
            layers.Activation('silu'),
            layers.Dense(2 * hidden_size, use_bias=True)
        ])

    def call(self, x, c):
        """Forward pass of the final layer.
        
        Args:
            x: Input tensor
            c: Conditioning tensor (timestep embedding)
            
        Returns:
            Output tensor after modulation and projection
        """
        modulation = self.adaLN_modulation(c)
        shift, scale = tf.split(modulation, 2, axis=1)
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


class PatchEmbedMR(layers.Layer):
    """2D Image to Patch Embedding layer.
    
    This layer converts a 2D image into a sequence of patch embeddings by:
    1. Splitting the image into fixed-size patches
    2. Projecting each patch to the embedding dimension using a convolutional layer
    3. Flattening the spatial dimensions to create a sequence
    
    Args:
        patch_size: Size of patches to extract (both height and width)
        in_chans: Number of input channels
        embed_dim: Dimension of the patch embeddings
        bias: Whether to include bias in the projection
    """
    def __init__(self, patch_size=2, in_chans=4, embed_dim=768, bias=True, dtype=tf.float32):
        super().__init__()
        self.proj = layers.Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            use_bias=bias,
            name='patch_embed',
            dtype=dtype
        )

    def call(self, x):
        """Forward pass of the patch embedding layer.
        
        Args:
            x: Input tensor of shape [B, H, W, C] or [B, C, H, W]
            
        Returns:
            Tensor of shape [B, N, D] where:
                B is batch size
                N is number of patches (H*W/patch_size^2)
                D is embedding dimension
        """
        # Handle both NHWC and NCHW formats
        if x.shape[1] == self.proj.input_shape[3]:  # NCHW format
            x = tf.transpose(x, [0, 2, 3, 1])  # Convert to NHWC
        
        x = self.proj(x)
        B, H, W, C = x.shape
        x = tf.reshape(x, [B, H * W, C])
        return x


class OmniGen(tf.keras.Model):
    """Diffusion model with a Transformer backbone.
    
    This is the main OmniGen model that combines:
    1. Patch embedding for processing images
    2. Timestep embedding for diffusion conditioning
    3. Transformer backbone for processing
    4. Adaptive layer normalization for conditioning
    5. LoRA support for efficient fine-tuning (via PeftAdapterMixin)
    
    The model can handle:
    - Multiple input resolutions
    - Classifier-free guidance
    - Key-value caching for efficient inference
    - Multi-modal inputs (text + images)
    - Parameter efficient fine-tuning with LoRA
    
    Args:
        transformer_config: Configuration for the Phi3 transformer
        patch_size: Size of image patches
        in_channels: Number of input channels
        pe_interpolation: Interpolation scale for positional embeddings (must be float)
        pos_embed_max_size: Maximum size for positional embeddings
    """
    def __init__(
        self,
        transformer_config: Phi3Config,
        patch_size=2,
        in_channels=4,
        pe_interpolation: float = 1.0,
        pos_embed_max_size: int = 192,
    ):
        super().__init__()
        
        # Memory management setup
        self._persistent_buffers = {}
        self._cached_weights = {}
        self._is_on_cpu = False
        
        # Model configuration
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.pos_embed_max_size = pos_embed_max_size
        self.pe_interpolation = pe_interpolation
        
        # Set compute dtype based on available hardware
        self._compute_dtype = tf.float32
        if tf.config.list_physical_devices('GPU'):
            self._compute_dtype = tf.float16
            
        # Initialize components with proper dtype
        hidden_size = transformer_config.hidden_size
        self.x_embedder = PatchEmbedMR(
            patch_size, in_channels, hidden_size,
            dtype=self._compute_dtype
        )
        self.input_x_embedder = PatchEmbedMR(
            patch_size, in_channels, hidden_size,
            dtype=self._compute_dtype
        )
        
        # Create embedders
        self.time_token = TimestepEmbedder(hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # Register positional embedding buffer
        pos_embed = get_2d_sincos_pos_embed(
            hidden_size, pos_embed_max_size,
            interpolation_scale=self.pe_interpolation,
            base_size=64
        )
        self._register_buffer(
            "pos_embed",
            tf.convert_to_tensor(pos_embed, dtype=self._compute_dtype)[None, ...],
            persistent=True
        )
        
        # Final layer and transformer
        self.final_layer = FinalLayer(
            hidden_size, patch_size, self.out_channels,
            dtype=self._compute_dtype
        )
        self.transformer = Phi3Transformer(
            config=transformer_config,
            dtype=self._compute_dtype
        )
        
        self.initialize_weights()
        
    def _register_buffer(self, name, tensor, persistent=False):
        """Register a persistent buffer similar to PyTorch."""
        if persistent:
            self._persistent_buffers[name] = tensor
        setattr(self, name, tensor)
        
    def _move_to_cpu(self):
        """Move model weights to CPU."""
        if not self._is_on_cpu:
            self._cached_weights = {}
            for var in self.trainable_variables:
                self._cached_weights[var.name] = var.numpy()
                with tf.device('/CPU:0'):
                    var.assign(tf.identity(var))
            self._is_on_cpu = True
            
    def _move_to_gpu(self):
        """Move model weights back to GPU."""
        if self._is_on_cpu and tf.config.list_physical_devices('GPU'):
            for var in self.trainable_variables:
                if var.name in self._cached_weights:
                    with tf.device('/GPU:0'):
                        var.assign(tf.identity(self._cached_weights[var.name]))
            self._cached_weights = {}
            self._is_on_cpu = False
            
    def _clear_memory(self):
        """Clear GPU memory."""
        if tf.config.list_physical_devices('GPU'):
            tf.keras.backend.clear_session()
            
    @tf.function(jit_compile=True)
    def call(self, x, timestep, input_ids=None, input_img_latents=None,
             input_image_sizes=None, attention_mask=None, position_ids=None,
             training=False):
        """Forward pass with memory optimizations."""
        try:
            # Process in smaller batches
            batch_size = x.shape[0]
            max_batch_size = 4  # Adjust based on memory
            
            outputs = []
            for i in range(0, batch_size, max_batch_size):
                end_idx = min(i + max_batch_size, batch_size)
                
                # Get batch slice
                x_batch = x[i:end_idx]
                timestep_batch = timestep[i:end_idx]
                
                # Process embeddings
                t_emb = self.t_embedder(timestep_batch)
                
                # Process latents
                latents = self.x_embedder(x_batch)
                height, width = x_batch.shape[2:4]
                pos_embed = self.cropped_pos_embed(height, width)
                latents = latents + pos_embed
                
                # Process input images
                if input_img_latents is not None:
                    img_latents = input_img_latents[i:end_idx]
                    input_latents = self.input_x_embedder(img_latents)
                    input_latents = input_latents + pos_embed
                else:
                    input_latents = None
                    
                # Run transformer
                hidden_states = self.transformer(
                    input_ids=input_ids[i:end_idx] if input_ids is not None else None,
                    attention_mask=attention_mask[i:end_idx] if attention_mask is not None else None,
                    position_ids=position_ids[i:end_idx] if position_ids is not None else None,
                    inputs_embeds=latents,
                    input_image_embeds=input_latents,
                    t_emb=t_emb,
                    training=training
                )
                
                # Process output
                output = self.final_layer(hidden_states, t_emb)
                output = self.unpatchify(output, height, width)
                outputs.append(output)
                
                # Clear memory after each batch
                if i + max_batch_size < batch_size:
                    self._clear_memory()
                    
            # Combine outputs
            return tf.concat(outputs, axis=0)
            
        finally:
            # Ensure memory is cleared
            self._clear_memory()
            
    def initialize_weights(self):
        """Initialize model weights."""
        # Helper function to initialize a single layer
        def _init_weights(layer):
            if isinstance(layer, tf.keras.layers.Dense):
                # Initialize weight matrices with truncated normal
                kernel_shape = layer.kernel.shape
                stddev = 1.0 / tf.sqrt(tf.cast(kernel_shape[-1], tf.float32))
                layer.kernel.assign(
                    tf.random.truncated_normal(kernel_shape, stddev=stddev)
                )
                
                # Initialize biases to zero if present
                if layer.use_bias:
                    layer.bias.assign(tf.zeros_like(layer.bias))
                    
            elif isinstance(layer, tf.keras.layers.LayerNormalization):
                # Initialize scale (gamma) to ones and offset (beta) to zeros
                # Only initialize if they are tf.Variables (not booleans)
                if hasattr(layer, 'scale') and isinstance(layer.scale, tf.Variable):
                    layer.scale.assign(tf.ones_like(layer.scale))
                if hasattr(layer, 'offset') and isinstance(layer.offset, tf.Variable):
                    layer.offset.assign(tf.zeros_like(layer.offset))
                    
            elif isinstance(layer, tf.keras.layers.Embedding):
                # Initialize embeddings with truncated normal
                kernel_shape = layer.embeddings.shape
                stddev = 1.0 / tf.sqrt(tf.cast(kernel_shape[-1], tf.float32))
                layer.embeddings.assign(
                    tf.random.truncated_normal(kernel_shape, stddev=stddev)
                )
        
        # Initialize all layers recursively
        for layer in self.layers:
            if hasattr(layer, 'layers'):  # For nested layers/models
                for sublayer in layer.layers:
                    _init_weights(sublayer)
            else:
                _init_weights(layer)
                
        # Initialize transformer separately since it's a custom model
        if hasattr(self, 'transformer') and hasattr(self.transformer, 'initialize_weights'):
            self.transformer.initialize_weights()
            
        print("Model weights initialized successfully")

    @classmethod
    def from_pretrained(cls, model_name, device=None, **kwargs):
        """Load model from pretrained weights.
        
        Args:
            model_name: Name or path of pretrained model
            device: Device to place model on (removed - handled by pipeline)
            **kwargs: Additional arguments to pass to model constructor
        """
        # Remove device from kwargs since it's handled by pipeline
        if 'device' in kwargs:
            del kwargs['device']
            
        # Download model if needed
        if not os.path.exists(model_name):
            from huggingface_hub import snapshot_download
            model_name = snapshot_download(model_name)
            print(f"Downloaded model to {model_name}")
            
        # Create model
        config = Phi3Config.from_pretrained(model_name)
        model = cls(config, **kwargs)
        
        # Load weights
        if os.path.exists(os.path.join(model_name, 'model.safetensors')):
            print("Loading safetensors weights...")
            from safetensors.torch import load_file
            state_dict = load_file(os.path.join(model_name, 'model.safetensors'))
            # Convert torch state dict to TF
            tf_state_dict = {}
            for k, v in state_dict.items():
                tf_state_dict[k] = tf.convert_to_tensor(v.numpy())
            model.load_weights(tf_state_dict)
        else:
            print("Loading PyTorch weights...")
            import torch
            state_dict = torch.load(os.path.join(model_name, 'model.pt'), map_location='cpu')
            # Convert torch state dict to TF
            tf_state_dict = {}
            for k, v in state_dict.items():
                tf_state_dict[k] = tf.convert_to_tensor(v.numpy())
            model.load_weights(tf_state_dict)
            
        return model

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
        
        if h * w != self.pos_embed.shape[1]:
            pos_embed = get_2d_sincos_pos_embed(
                self.transformer.config.hidden_size,
                (h, w),
                interpolation_scale=self.pe_interpolation,
                base_size=64
            )
            pos_embed = tf.convert_to_tensor(pos_embed[None])
        
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
