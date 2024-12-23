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


class OmniGen(tf.keras.Model, PeftAdapterMixin):
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
        device: Device to place model on ('CPU', 'GPU', or None for auto-detect)
        compute_dtype: Data type for computations (e.g. tf.float32, tf.bfloat16)
        mixed_precision: Whether to use mixed precision training
    """
    def __init__(
        self,
        transformer_config: Phi3Config,
        patch_size=2,
        in_channels=4,
        pe_interpolation: float = 1.0,
        pos_embed_max_size: int = 192,
        device=None,
        compute_dtype=tf.bfloat16,
        mixed_precision: bool = True,
    ):
        tf.keras.Model.__init__(self)
        PeftAdapterMixin.__init__(self)
        
        # Store configuration
        self._compute_dtype = compute_dtype
        self._device = None
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.pos_embed_max_size = pos_embed_max_size
        self.pe_interpolation = pe_interpolation
        
        # Set up device and dtype strategy
        if device is None:
            if len(tf.config.list_physical_devices('GPU')) > 0:
                device = 'GPU'
                print("Using GPU for inference")
                
                # Enable mixed precision if requested and on GPU
                if mixed_precision:
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    print("Enabled mixed precision training")
            else:
                device = 'CPU'
                print("No GPU found, using CPU for inference")
                
                # Disable mixed precision on CPU
                if mixed_precision:
                    print("Mixed precision disabled on CPU")
                    mixed_precision = False
                    
        self._device = device

        # Initialize memory optimization flags
        self.use_kv_cache = True
        self.offload_kv_cache = True
        self.model_cpu_offload = False
        self._kv_cache = None
        
        # Build model components with proper device placement
        with tf.device(f'/{self._device}:0'):
            # Create embedders
            self.x_embedder = PatchEmbedMR(
                patch_size, in_channels, transformer_config.hidden_size,
                dtype=self._compute_dtype
            )
            self.input_x_embedder = PatchEmbedMR(
                patch_size, in_channels, transformer_config.hidden_size,
                dtype=self._compute_dtype
            )
            
            # Create time embedders
            self.time_token = TimestepEmbedder(
                transformer_config.hidden_size,
                dtype=self._compute_dtype
            )
            self.t_embedder = TimestepEmbedder(
                transformer_config.hidden_size,
                dtype=self._compute_dtype
            )
            
            # Create positional embeddings
            pos_embed = get_2d_sincos_pos_embed(
                transformer_config.hidden_size,
                pos_embed_max_size,
                interpolation_scale=self.pe_interpolation,
                base_size=64
            )
            self.pos_embed = tf.constant(pos_embed, dtype=self._compute_dtype)
            
            # Create final layer
            self.final_layer = FinalLayer(
                transformer_config.hidden_size,
                patch_size,
                self.out_channels,
                dtype=self._compute_dtype
            )
            
            # Create transformer
            self.transformer = Phi3Transformer(
                transformer_config,
                dtype=self._compute_dtype
            )
            # Disable cache by default for memory efficiency
            self.transformer.config.use_cache = False
            
        self.initialize_weights()

    def enable_cpu_offload(self):
        """Enable CPU offloading for memory savings."""
        self.model_cpu_offload = True
        print("Moving model layers to CPU...")
        
        # Use tf.device context to ensure proper placement
        with tf.device('/CPU:0'):
            # Move transformer layers to CPU by forcing a copy
            for layer in self.transformer.layers:
                layer.set_weights([tf.identity(w).numpy() for w in layer.weights])
                
            # Move embedders to CPU
            self.x_embedder.set_weights([tf.identity(w).numpy() for w in self.x_embedder.weights])
            self.input_x_embedder.set_weights([tf.identity(w).numpy() for w in self.input_x_embedder.weights])
            
            # Move other components
            self.time_token.set_weights([tf.identity(w).numpy() for w in self.time_token.weights])
            self.t_embedder.set_weights([tf.identity(w).numpy() for w in self.t_embedder.weights])
            self.final_layer.set_weights([tf.identity(w).numpy() for w in self.final_layer.weights])
            
        # Clear GPU memory
        tf.keras.backend.clear_session()
        print("Model layers moved to CPU")

    def disable_cpu_offload(self):
        """Disable CPU offloading."""
        self.model_cpu_offload = False
        print("Moving model layers back to GPU...")
        
        # Use tf.device context to ensure proper placement
        with tf.device(f'/{self._device}:0'):
            # Move transformer layers back to GPU
            for layer in self.transformer.layers:
                layer.set_weights([tf.identity(w) for w in layer.weights])
                
            # Move embedders back to GPU
            self.x_embedder.set_weights([tf.identity(w) for w in self.x_embedder.weights])
            self.input_x_embedder.set_weights([tf.identity(w) for w in self.input_x_embedder.weights])
            
            # Move other components
            self.time_token.set_weights([tf.identity(w) for w in self.time_token.weights])
            self.t_embedder.set_weights([tf.identity(w) for w in self.t_embedder.weights])
            self.final_layer.set_weights([tf.identity(w) for w in self.final_layer.weights])
            
        print("Model layers moved back to GPU")

    def enable_kv_cache(self):
        """Enable key-value caching for faster inference."""
        self.use_kv_cache = True
        self.transformer.config.use_cache = True
        print("Key-value caching enabled")

    def disable_kv_cache(self):
        """Disable key-value caching."""
        self.use_kv_cache = False
        self.transformer.config.use_cache = False
        self._kv_cache = None
        print("Key-value caching disabled")

    def clear_kv_cache(self):
        """Clear the key-value cache."""
        self._kv_cache = None
        tf.keras.backend.clear_session()
        print("Key-value cache cleared")

    def offload_kv_cache_to_cpu(self):
        """Move key-value cache to CPU to save GPU memory."""
        if self._kv_cache is not None:
            print("Moving KV cache to CPU...")
            with tf.device('/CPU:0'):
                # Convert tensors to numpy and store on CPU
                self._kv_cache = [tf.identity(k).numpy() for k in self._kv_cache]
            print("KV cache moved to CPU")

    def load_kv_cache_to_gpu(self):
        """Move key-value cache back to GPU."""
        if self._kv_cache is not None:
            print("Moving KV cache to GPU...")
            with tf.device(f'/{self._device}:0'):
                # Convert numpy arrays back to tensors on GPU
                self._kv_cache = [tf.convert_to_tensor(k, dtype=self._compute_dtype) for k in self._kv_cache]
            print("KV cache moved to GPU")

    @tf.function(reduce_retracing=True)
    def call(self, *args, training=False, **kwargs):
        """Memory-efficient forward pass with KV caching."""
        # Handle CPU offloading
        if self.model_cpu_offload:
            self.enable_cpu_offload()
            
        # Clear session to free memory
        if not training and not self.use_kv_cache:
            tf.keras.backend.clear_session()
            
        try:
            # Move KV cache to GPU if needed
            if self.use_kv_cache and self._kv_cache is not None:
                self.load_kv_cache_to_gpu()
                
            # Forward pass
            outputs = super().call(*args, training=training, **kwargs)
            
            # Update and offload KV cache if needed
            if self.use_kv_cache and self.offload_kv_cache:
                if isinstance(outputs, tuple) and len(outputs) > 1:
                    self._kv_cache = outputs[1]
                    self.offload_kv_cache_to_cpu()
                    
            return outputs
            
        finally:
            # Always move back to GPU if needed
            if self.model_cpu_offload:
                self.disable_cpu_offload()
                
            # Clear any temporary tensors
            if not training:
                tf.keras.backend.clear_session()

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
    def from_pretrained(cls, model_name, device=None, mixed_precision=True, **kwargs):
        """Load model from pretrained weights.
        
        Args:
            model_name: Name of the model to load
            device: Device to place model on ('CPU', 'GPU', or None for auto-detect)
            mixed_precision: Whether to use mixed precision training
            **kwargs: Additional arguments passed to model initialization
            
        Returns:
            Initialized model with pretrained weights
        """
        if not os.path.exists(model_name):
            print(f"Downloading model from {model_name}")
            cache_folder = os.getenv('HF_HUB_CACHE')
            model_name = snapshot_download(
                repo_id=model_name,
                cache_dir=cache_folder,
                ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'pytorch_model.bin']
            )
            print(f"Downloaded model to {model_name}")

        config = Phi3Config.from_pretrained(model_name)
        model = cls(config, device=device, mixed_precision=mixed_precision, **kwargs)
        
        if os.path.exists(os.path.join(model_name, 'model.safetensors')):
            print("Loading safetensors weights...")
            try:
                # Load weights with CPU device
                with tf.device('/CPU:0'):
                    from safetensors.tensorflow import load_file
                    ckpt = load_file(os.path.join(model_name, 'model.safetensors'))
                    
                    # Convert weights to TensorFlow format
                    tf_weights = {}
                    for name, tensor in ckpt.items():
                        # Convert PyTorch weight names to TensorFlow format
                        tf_name = name.replace('.', '/')
                        # Ensure tensor is on CPU and convert to float32
                        tf_weights[tf_name] = tf.cast(tensor, tf.float32)
                    
                    # Load weights into model
                    model.load_weights(tf_weights)
                    print("Successfully loaded weights from safetensors")
                    
            except Exception as e:
                print(f"Error loading safetensors: {str(e)}")
                print("Attempting to load TensorFlow checkpoint...")
                try:
                    model = tf.saved_model.load(os.path.join(model_name, 'model'))
                except Exception as e2:
                    print(f"Error loading TensorFlow checkpoint: {str(e2)}")
                    raise ValueError("Failed to load model weights") from e2
        else:
            print("No safetensors found, loading TensorFlow checkpoint...")
            try:
                model = tf.saved_model.load(os.path.join(model_name, 'model'))
            except Exception as e:
                print(f"Error loading TensorFlow checkpoint: {str(e)}")
                raise ValueError("Failed to load model weights") from e
                
        print("Model loaded successfully")
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
        
        return x_list, pos_embed_list

    def call(self, x, timestep, input_ids, input_img_latents, input_image_sizes,
            attention_mask, position_ids, padding_latent=None, past_key_values=None,
            return_past_key_values=True, offload_model=False):
        """Forward pass of the model.
        
        This is the main method that processes inputs through the entire model pipeline:
        1. Patch embedding of input images/latents
        2. Time embedding for diffusion conditioning
        3. Processing through transformer
        4. Final layer for output generation
        
        Args:
            x: Input latents or list of latents
            timestep: Diffusion timesteps
            input_ids: Text input IDs
            input_img_latents: Optional input image latents
            input_image_sizes: Sizes of input images
            attention_mask: Attention mask for transformer
            position_ids: Position IDs for transformer
            padding_latent: Optional padding latent
            past_key_values: Optional cached key-values
            return_past_key_values: Whether to return key-values
            offload_model: Whether to offload model during processing
            
        Returns:
            Model output (image or list of images) and optionally past_key_values
        """
        # Process input latents
        x_list, pos_embed_list = self.patch_multiple_resolutions(
            [x] if not isinstance(x, (list, tuple)) else x,
            padding_latent,
            is_input_images=False
        )
        
        # Process input images if provided
        input_img_embed_list = []
        if input_img_latents is not None:
            input_img_embed_list, _ = self.patch_multiple_resolutions(
                input_img_latents,
                is_input_images=True
            )
        
        # Time embedding
        t_emb = self.t_embedder(timestep)
        time_tokens = self.time_token(timestep)[:, None]
        
        # Prepare inputs for transformer
        transformer_inputs = []
        for x_embed, pos_embed in zip(x_list, pos_embed_list):
            transformer_inputs.append(x_embed + pos_embed)
        
        hidden_states = tf.concat(transformer_inputs, axis=1)
        hidden_states = tf.concat([time_tokens, hidden_states], axis=1)
        
        # Add input image embeddings if available
        if input_img_embed_list:
            input_img_embeds = tf.concat(input_img_embed_list, axis=1)
            hidden_states = tf.concat([hidden_states, input_img_embeds], axis=1)
        
        # Transformer forward pass
        outputs = self.transformer(
            input_ids=input_ids,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=return_past_key_values,
            output_hidden_states=True
        )
        
        # Process outputs
        x = outputs.hidden_states[-1][:, 1:]  # Remove time token
        x = self.final_layer(x, t_emb)
        
        # Split output for different resolutions
        if isinstance(input_image_sizes, (list, tuple)):
            out_list = []
            current_idx = 0
            for h, w in input_image_sizes:
                h_patch, w_patch = h // self.patch_size, w // self.patch_size
                out = x[:, current_idx:current_idx + h_patch * w_patch]
                out = self.unpatchify(out, h, w)
                out_list.append(out)
                current_idx += h_patch * w_patch
            x = out_list[0] if len(out_list) == 1 else out_list
        else:
            x = self.unpatchify(x, input_image_sizes[0], input_image_sizes[1])
        
        if return_past_key_values:
            return x, outputs.past_key_values
        return x

    def forward_with_cfg(self, x, timestep, input_ids, input_img_latents, input_image_sizes,
                        attention_mask, position_ids, cfg_scale, use_img_cfg, img_cfg_scale,
                        past_key_values, use_kv_cache, offload_model):
        """Forward pass with classifier-free guidance.
        
        This method implements classifier-free guidance by:
        1. Running the model on both conditional and unconditional inputs
        2. Combining the outputs using the guidance scale
        
        Args:
            x: Input latents
            timestep: Diffusion timesteps
            input_ids: Text input IDs
            input_img_latents: Optional input image latents
            input_image_sizes: Sizes of input images
            attention_mask: Attention mask for transformer
            position_ids: Position IDs for transformer
            cfg_scale: Classifier-free guidance scale
            use_img_cfg: Whether to use image guidance
            img_cfg_scale: Image guidance scale
            past_key_values: Optional cached key-values
            use_kv_cache: Whether to use key-value cache
            offload_model: Whether to offload model during processing
            
        Returns:
            Guided model output and optionally past_key_values
        """
        # Double the inputs for cfg
        latents = [x] if not isinstance(x, (list, tuple)) else x
        timesteps = tf.concat([timestep] * 2, axis=0)
        input_ids_double = tf.concat([input_ids] * 2, axis=0)
        
        if attention_mask is not None:
            attention_mask = tf.concat([attention_mask] * 2, axis=0)
        if position_ids is not None:
            position_ids = tf.concat([position_ids] * 2, axis=0)
        
        # Handle input images
        if input_img_latents is not None and use_img_cfg:
            input_img_latents = [tf.concat([img] * 2, axis=0) for img in input_img_latents]
        
        # Forward pass
        if use_kv_cache and past_key_values is not None:
            model_output = self.call(
                latents,
                timesteps,
                input_ids_double,
                input_img_latents,
                input_image_sizes,
                attention_mask,
                position_ids,
                past_key_values=past_key_values,
                return_past_key_values=True,
                offload_model=offload_model
            )
            model_output, past_key_values = model_output
        else:
            model_output = self.call(
                latents,
                timesteps,
                input_ids_double,
                input_img_latents,
                input_image_sizes,
                attention_mask,
                position_ids,
                offload_model=offload_model
            )
        
        # Apply classifier-free guidance
        if isinstance(model_output, (list, tuple)):
            model_output = [
                self._apply_cfg(out, cfg_scale if i == 0 else img_cfg_scale)
                for i, out in enumerate(model_output)
            ]
        else:
            model_output = self._apply_cfg(model_output, cfg_scale)
        
        if use_kv_cache and past_key_values is not None:
            return model_output, past_key_values
        return model_output

    def _apply_cfg(self, model_output, scale):
        """Apply classifier-free guidance scaling.
        
        Args:
            model_output: Model output tensor
            scale: Guidance scale factor
            
        Returns:
            Scaled output tensor
        """
        batch_size = tf.shape(model_output)[0] // 2
        cond, uncond = tf.split(model_output, 2, axis=0)
        return uncond + scale * (cond - uncond)
