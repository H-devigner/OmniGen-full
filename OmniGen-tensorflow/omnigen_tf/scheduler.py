"""OmniGen TensorFlow Scheduler

This module provides the TensorFlow implementation of the OmniGen scheduler,
matching the PyTorch version's functionality while leveraging TensorFlow-specific optimizations.
"""

from typing import Optional, Dict, Any, Tuple, List
import gc
from tqdm import tqdm

import tensorflow as tf
import numpy as np


@tf.function(jit_compile=True)
def _update_cache_tensors(key_states: tf.Tensor, value_states: tf.Tensor, 
                         existing_key: Optional[tf.Tensor] = None,
                         existing_value: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]:
    """Update cache tensors with XLA optimization."""
    if existing_key is not None and existing_value is not None:
        key_states = tf.concat([existing_key, key_states], axis=-2)
        value_states = tf.concat([existing_value, value_states], axis=-2)
    return key_states, value_states


class OmniGenCache:
    """TensorFlow implementation of dynamic cache for OmniGen model."""

    def __init__(self, num_tokens_for_img: int, offload_kv_cache: bool = False) -> None:
        """Initialize cache.
        
        Args:
            num_tokens_for_img: Number of tokens for image
            offload_kv_cache: Whether to offload key-value cache to CPU
        """
        if not tf.config.list_physical_devices('GPU'):
            raise RuntimeError(
                "OffloadedCache can only be used with a GPU. If there is no GPU, "
                "you need to set use_kv_cache=False, which will result in longer inference time!"
            )
        
        self.key_cache = []
        self.value_cache = []
        self._seen_tokens = 0
        self.num_tokens_for_img = num_tokens_for_img
        self.offload_kv_cache = offload_kv_cache
        
        # Enable mixed precision for cache operations
        if tf.config.list_physical_devices('GPU'):
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Setup memory growth
        self._setup_memory_optimization()
    
    def _setup_memory_optimization(self):
        """Setup memory optimization configurations."""
        if tf.config.list_physical_devices('GPU'):
            # Enable memory growth and limit GPU memory
            for device in tf.config.list_physical_devices('GPU'):
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                    # Use 70% of available memory
                    memory_limit = int(tf.config.experimental.get_memory_info('GPU:0')['free'] * 0.7)
                    tf.config.set_logical_device_configuration(
                        device,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                    )
                except:
                    pass

    @tf.function(jit_compile=True)
    def update(
        self,
        key_states: tf.Tensor,
        value_states: tf.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Update cache with new key and value states.
        
        Args:
            key_states: New key states to cache
            value_states: New value states to cache
            layer_idx: Index of layer to cache states for
            cache_kwargs: Additional arguments for cache
            
        Returns:
            Tuple of updated key and value states
        """
        if len(self.key_cache) < layer_idx:
            raise ValueError("Cache does not support skipping layers. Use DynamicCache.")
        elif len(self.key_cache) == layer_idx:
            # Only cache states for condition tokens
            key_states = key_states[..., :-(self.num_tokens_for_img + 1), :]
            value_states = value_states[..., :-(self.num_tokens_for_img + 1), :]

            # Update seen tokens count
            if layer_idx == 0:
                self._seen_tokens += tf.shape(key_states)[-2]

            # Initialize cache for new layer with proper device placement
            device = "/CPU:0" if self.offload_kv_cache else "/GPU:0"
            with tf.device(device):
                # Use float16 for GPU tensors
                if device == "/GPU:0":
                    key_states = tf.cast(key_states, tf.float16)
                    value_states = tf.cast(value_states, tf.float16)
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
        else:
            # Update existing cache with XLA optimization
            key_states, value_states = _update_cache_tensors(
                key_states, value_states,
                self.key_cache[layer_idx], self.value_cache[layer_idx]
            )
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states

        return key_states, value_states

    def __getitem__(self, layer_idx: int) -> List[Tuple[tf.Tensor]]:
        """Get cache for layer, handling prefetch and eviction."""
        if layer_idx < len(self.key_cache):
            if self.offload_kv_cache:
                # Evict previous layer
                self.evict_previous_layer(layer_idx)
                
                # Load current layer to GPU with non-blocking transfer
                with tf.device("/GPU:0"):
                    key_tensor = tf.identity(self.key_cache[layer_idx])
                    value_tensor = tf.identity(self.value_cache[layer_idx])
                
                # Prefetch next layer asynchronously
                tf.function(self.prefetch_layer, jit_compile=True)((layer_idx + 1) % len(self.key_cache))
            else:
                key_tensor = self.key_cache[layer_idx]
                value_tensor = self.value_cache[layer_idx]
            return (key_tensor, value_tensor)
        else:
            raise KeyError(f"Cache only has {len(self.key_cache)} layers, attempted to access layer {layer_idx}")

    @tf.function(experimental_compile=True)
    def evict_previous_layer(self, layer_idx: int) -> None:
        """Move previous layer cache to CPU with XLA optimization."""
        if len(self.key_cache) > 2:
            prev_layer_idx = -1 if layer_idx == 0 else (layer_idx - 1) % len(self.key_cache)
            with tf.device("/CPU:0"):
                # Non-blocking transfer
                self.key_cache[prev_layer_idx] = tf.identity(self.key_cache[prev_layer_idx])
                self.value_cache[prev_layer_idx] = tf.identity(self.value_cache[prev_layer_idx])
                
                # Clear GPU memory
                tf.keras.backend.clear_session()

    @tf.function(experimental_compile=True)
    def prefetch_layer(self, layer_idx: int) -> None:
        """Prefetch next layer to GPU with XLA optimization."""
        if layer_idx < len(self.key_cache):
            with tf.device("/GPU:0"):
                # Non-blocking transfer with float16
                self.key_cache[layer_idx] = tf.cast(tf.identity(self.key_cache[layer_idx]), tf.float16)
                self.value_cache[layer_idx] = tf.cast(tf.identity(self.value_cache[layer_idx]), tf.float16)


class DDIMScheduler:
    """DDIM Scheduler for OmniGen model."""
    
    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=0,
        prediction_type="epsilon",
    ):
        """Initialize scheduler.
        
        Args:
            num_train_timesteps (int): Number of training timesteps
            beta_start (float): Starting beta value
            beta_end (float): Ending beta value
            beta_schedule (str): Beta schedule type
            clip_sample (bool): Whether to clip sample values
            set_alpha_to_one (bool): Whether to set alpha to 1 for final step
            steps_offset (int): Offset for number of steps
            prediction_type (str): Type of prediction to use
        """
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.clip_sample = clip_sample
        self.set_alpha_to_one = set_alpha_to_one
        self.steps_offset = steps_offset
        self.prediction_type = prediction_type
        
        # Initialize betas and alphas
        self.betas = self._get_betas()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = tf.math.cumprod(self.alphas)
        
        # Initialize timesteps
        self.timesteps = None
        
    def _get_betas(self):
        """Get beta schedule."""
        if self.beta_schedule == "linear":
            betas = np.linspace(self.beta_start, self.beta_end, self.num_train_timesteps)
        elif self.beta_schedule == "scaled_linear":
            # Glide paper betas
            betas = np.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.num_train_timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
        return tf.cast(betas, tf.float32)
        
    def set_timesteps(self, num_inference_steps):
        """Set timesteps for inference.
        
        Args:
            num_inference_steps (int): Number of inference steps
        """
        self.num_inference_steps = num_inference_steps
        
        # Create evenly spaced timesteps
        timesteps = np.linspace(0, self.num_train_timesteps - 1, num_inference_steps)
        timesteps = np.flip(timesteps)  # Reverse for denoising
        self.timesteps = tf.cast(timesteps, tf.int32)
        
    def step(self, model_output, timestep, sample):
        """Scheduler step for denoising.
        
        Args:
            model_output: Output from model
            timestep: Current timestep
            sample: Current sample
            
        Returns:
            Denoised sample
        """
        # Get alpha values for current and previous timestep
        t = timestep
        prev_t = t - 1 if t > 0 else t
        
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else tf.ones_like(alpha_prod_t)
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # Compute coefficients
        sqrt_alpha_prod = tf.sqrt(alpha_prod_t)
        sqrt_one_minus_alpha_prod = tf.sqrt(beta_prod_t)
        
        # Predict x0
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - sqrt_one_minus_alpha_prod * model_output) / sqrt_alpha_prod
        elif self.prediction_type == "v_prediction":
            pred_original_sample = sqrt_alpha_prod * sample - sqrt_one_minus_alpha_prod * model_output
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
            
        # Compute coefficients for denoising step
        sqrt_alpha_prod_prev = tf.sqrt(alpha_prod_t_prev)
        sqrt_one_minus_alpha_prod_prev = tf.sqrt(beta_prod_t_prev)
        
        # Denoise
        pred_sample_direction = sqrt_one_minus_alpha_prod_prev * model_output
        prev_sample = sqrt_alpha_prod_prev * pred_original_sample + pred_sample_direction
        
        if self.clip_sample:
            prev_sample = tf.clip_by_value(prev_sample, -1, 1)
            
        return prev_sample


class OmniGenScheduler:
    """Scheduler for OmniGen model."""

    def __init__(self, num_steps: int = 50, time_shifting_factor: int = 1):
        """Initialize scheduler.
        
        Args:
            num_steps: Number of diffusion steps
            time_shifting_factor: Factor for time shifting
        """
        self.num_steps = num_steps
        self.time_shift = time_shifting_factor

        # Precompute sigma values with better precision
        with tf.device("/CPU:0"):
            t = tf.cast(tf.linspace(0.0, 1.0, num_steps + 1), tf.float32)
            t = t / (t + time_shifting_factor - time_shifting_factor * t)
            self.sigma = t

    @tf.function(jit_compile=True)
    def crop_kv_cache(self, past_key_values, num_tokens_for_img):
        """Crop key-value cache with XLA optimization."""
        for i in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[i] = past_key_values.key_cache[i][..., :-(num_tokens_for_img + 1), :]
            past_key_values.value_cache[i] = past_key_values.value_cache[i][..., :-(num_tokens_for_img + 1), :]
        return past_key_values

    @tf.function(jit_compile=True)
    def crop_position_ids_for_cache(self, position_ids, num_tokens_for_img):
        """Crop position IDs for cache with XLA optimization."""
        return position_ids[:, -(num_tokens_for_img + 1):]

    @tf.function(jit_compile=True)
    def crop_attention_mask_for_cache(self, attention_mask, num_tokens_for_img):
        """Crop attention mask for cache with XLA optimization."""
        return attention_mask[..., -(num_tokens_for_img + 1):, :]

    def __call__(self, z, func, model_kwargs, use_kv_cache: bool = True, offload_kv_cache: bool = True):
        """Run diffusion process.
        
        Args:
            z: Input tensor
            func: Model function
            model_kwargs: Model arguments
            use_kv_cache: Whether to use key-value cache
            offload_kv_cache: Whether to offload cache to CPU
            
        Returns:
            Processed tensor
        """
        # Calculate tokens on CPU to avoid GPU memory usage
        with tf.device("/CPU:0"):
            num_tokens_for_img = tf.shape(z)[-1] * tf.shape(z)[-2] // 4
            
        # Initialize cache with optimized memory settings
        cache = OmniGenCache(num_tokens_for_img, offload_kv_cache) if use_kv_cache else None

        # Pre-allocate tensors for better memory efficiency
        batch_size = tf.shape(z)[0]
        timesteps = tf.zeros([batch_size], dtype=tf.float32)

        for i in tqdm(range(self.num_steps)):
            # Update timesteps efficiently
            timesteps = tf.fill([batch_size], self.sigma[i])
            
            # Run model step
            with tf.GradientTape() as tape:
                pred, cache = func(z, timesteps, cache=cache, **model_kwargs)
            
            # Update z with better precision
            sigma_next = self.sigma[i + 1]
            sigma = self.sigma[i]
            z = tf.add(z, (sigma_next - sigma) * pred)

            if i == 0 and use_kv_cache:
                # Update model kwargs for caching with XLA optimization
                model_kwargs["position_ids"] = self.crop_position_ids_for_cache(
                    model_kwargs["position_ids"], num_tokens_for_img
                )
                model_kwargs["attention_mask"] = self.crop_attention_mask_for_cache(
                    model_kwargs["attention_mask"], num_tokens_for_img
                )

            # Aggressive memory cleanup
            if use_kv_cache and i > 0:
                # Clear Python garbage
                gc.collect()
                
                # Clear GPU memory
                if tf.config.list_physical_devices('GPU'):
                    tf.keras.backend.clear_session()
                    for device in tf.config.list_physical_devices('GPU'):
                        tf.config.experimental.reset_memory_stats(device)

        return z


"""OmniGen Scheduler implementation."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

import tensorflow as tf


@tf.function(jit_compile=True)
def _get_timestep_embedding(timesteps, embedding_dim: int, dtype=None):
    """Create sinusoidal timestep embeddings.
    
    Args:
        timesteps: 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
        embedding_dim: Dimension of the embeddings to create.
        dtype: Data type of the embeddings.
        
    Returns:
        embedding: [N x embedding_dim] Tensor of positional embeddings.
    """
    timesteps = tf.cast(timesteps, dtype=tf.float32)
    
    half_dim = embedding_dim // 2
    emb = tf.math.log(10000.0) / (half_dim - 1)
    emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=1)
    
    if embedding_dim % 2 == 1:  # Zero pad odd dimensions
        emb = tf.pad(emb, [[0, 0], [0, 1]])
        
    if dtype is not None:
        emb = tf.cast(emb, dtype)
    return emb


class OmniGenScheduler:
    """Scheduler for OmniGen model."""
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        trained_betas: Optional[np.ndarray] = None,
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        **kwargs,
    ):
        """Initialize scheduler.
        
        Args:
            num_train_timesteps: Number of diffusion steps used to train the model.
            beta_start: Starting value for beta schedule.
            beta_end: Ending value for beta schedule.
            beta_schedule: Beta schedule, either "linear" or "scaled_linear".
            trained_betas: Optional pre-defined beta schedule.
            clip_sample: Whether to clip predicted sample between -1 and 1.
            prediction_type: Prediction type, either "epsilon" or "sample".
        """
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.trained_betas = trained_betas
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type
        
        # Initialize timesteps and betas
        self.timesteps = None
        if trained_betas is not None:
            self.betas = tf.constant(trained_betas, dtype=tf.float32)
        elif beta_schedule == "linear":
            self.betas = tf.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "scaled_linear":
            # Glide/DDPM schedule
            betas = tf.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps) ** 2
            self.betas = tf.cast(betas, tf.float32)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
            
        # Pre-compute values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = tf.math.cumprod(self.alphas)
        self.one = tf.constant(1.0, dtype=tf.float32)
        
        # For caching during inference
        self.num_inference_steps = None
        self.timestep_map = None
        
    def set_timesteps(self, num_inference_steps: int):
        """Set the timesteps used for the diffusion chain.
        
        Args:
            num_inference_steps: Number of diffusion steps to run.
        """
        if self.num_inference_steps == num_inference_steps:
            return
            
        self.num_inference_steps = num_inference_steps
        
        # Create evenly spaced timesteps
        timesteps = tf.linspace(0, self.num_train_timesteps - 1, num_inference_steps)
        self.timesteps = tf.cast(tf.flip(timesteps), tf.int32)  # Flip for denoising
        
        # Create timestep map for fast lookup
        self.timestep_map = tf.zeros(self.num_train_timesteps, dtype=tf.int32)
        indices = tf.cast(self.timesteps, tf.int32)
        updates = tf.range(len(self.timesteps))
        self.timestep_map = tf.tensor_scatter_nd_update(
            self.timestep_map,
            indices[:, None],
            updates
        )
        
    def _get_variance(self, timestep: int) -> tf.Tensor:
        """Get variance for given timestep."""
        prev_t = timestep - 1 if timestep > 0 else 0
        
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
        
        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        
        return variance
        
    def step(
        self,
        model_output: tf.Tensor,
        timestep: int,
        sample: tf.Tensor,
        return_dict: bool = True,
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """Predict the sample from the previous timestep by reversing the SDE.
        
        Args:
            model_output: Direct output from learned diffusion model.
            timestep: Current discrete timestep in the diffusion chain.
            sample: Current instance of sample being created by diffusion process.
            return_dict: Whether to return output as dict or tuple.
            
        Returns:
            pred_prev_sample: Predicted previous sample
        """
        # Get alphas for current and previous timestep
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[timestep - 1] if timestep > 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # For t > 0, compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        else:
            raise ValueError(f"prediction_type given as {self.prediction_type} must be one of `epsilon`, or `sample`")
            
        # Get previous sample based on (x_0, x_t)
        pred_prev_sample = (alpha_prod_t_prev ** (0.5) * pred_original_sample +
                          beta_prod_t_prev ** (0.5) * model_output)
        
        if self.clip_sample:
            pred_prev_sample = tf.clip_by_value(pred_prev_sample, -1, 1)
            
        return pred_prev_sample
