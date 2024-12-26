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
        # Ensure consistent precision (use float32 for computations)
        model_output = tf.cast(model_output, tf.float32)
        sample = tf.cast(sample, tf.float32)
        timestep = tf.cast(timestep, tf.int32)
        
        # Get alpha values for current and previous timestep
        t = timestep
        prev_t = t - 1 if t > 0 else t
        
        alpha_prod_t = tf.cast(self.alphas_cumprod[t], tf.float32)
        alpha_prod_t_prev = tf.cast(
            self.alphas_cumprod[prev_t] if prev_t >= 0 else 1.0, 
            tf.float32
        )
        beta_prod_t = 1 - alpha_prod_t
        
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
            
        # Cast back to original dtype of sample
        prev_sample = tf.cast(prev_sample, sample.dtype)
        
        return prev_sample


class OmniGenScheduler:
    """Scheduler for OmniGen model."""
    
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
        **kwargs
    ):
        """Initialize scheduler."""
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.clip_sample = clip_sample
        self.set_alpha_to_one = set_alpha_to_one
        self.steps_offset = steps_offset
        self.prediction_type = prediction_type
        
        # Initialize betas and alphas
        if beta_schedule == "linear":
            self.betas = tf.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "scaled_linear":
            # Scale the betas linearly
            self.betas = tf.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
            
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = tf.math.cumprod(self.alphas, axis=0)
        
        # Store for easy access
        self.final_alpha_cumprod = self.alphas_cumprod[-1]
        
        # For noise prediction
        self.init_noise_sigma = 1.0
        
    def set_timesteps(self, num_inference_steps):
        """Set timesteps for inference."""
        self.num_inference_steps = num_inference_steps
        
        # Create evenly spaced timesteps
        timesteps = tf.linspace(
            self.num_train_timesteps - 1,
            0,
            num_inference_steps
        )
        
        # Add offset and cast to int
        self.timesteps = tf.cast(timesteps + self.steps_offset, tf.int32)
        self.sigmas = tf.zeros_like(timesteps)  # For compatibility
        
    def _get_variance(self, timestep, prev_timestep):
        """Get variance for given timestep."""
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        
        return variance
        
    def scale_model_input(self, sample, timestep):
        """Scale input sample for model."""
        timestep = tf.cast(timestep, tf.int32)
        
        # Get step index
        step_index = tf.where(self.timesteps == timestep)[0][0]
        
        # No scaling needed for DDPM
        return sample
        
    def step(
        self,
        model_output,
        timestep,
        sample,
        return_dict=True,
        **kwargs
    ):
        """
        Predict the previous noisy sample x_t -> x_t-1.
        
        Args:
            model_output (Tensor): Predicted noise or sample from the model
            timestep (Tensor): Current timestep
            sample (Tensor): Current noisy sample
            return_dict (bool): Whether to return a dictionary or tuple
        
        Returns:
            Tensor or Dict: Denoised sample
        """
        # Debug print tensor shapes and dtypes
        def debug_tensor_info(tensor, name):
            print(f"{name} - Shape: {tensor.shape}, Dtype: {tensor.dtype}")
        
        debug_tensor_info(model_output, "model_output")
        debug_tensor_info(timestep, "timestep")
        debug_tensor_info(sample, "sample")
        
        # Ensure consistent precision and shape compatibility
        model_output = tf.cast(model_output, tf.float32)
        sample = tf.cast(sample, tf.float32)
        timestep = tf.cast(timestep, tf.int32)
        
        # Ensure timestep is a scalar
        timestep = tf.squeeze(timestep)
        
        # Compute noise schedule parameters
        try:
            alpha_prod_t = tf.cast(self.alphas_cumprod[timestep], tf.float32)
            alpha_prod_t_prev = tf.cast(
                self.alphas_cumprod[timestep - 1] if timestep > 0 else 1.0, 
                tf.float32
            )
        except Exception as e:
            print(f"Error accessing alphas_cumprod: {e}")
            print(f"Timestep value: {timestep}")
            print(f"Alphas_cumprod shape: {self.alphas_cumprod.shape}")
            raise
        
        # Compute beta product
        beta_prod_t = 1 - alpha_prod_t
        
        # Ensure scalar values are broadcast compatible
        alpha_prod_t = tf.broadcast_to(alpha_prod_t, sample.shape)
        alpha_prod_t_prev = tf.broadcast_to(alpha_prod_t_prev, sample.shape)
        beta_prod_t = tf.broadcast_to(beta_prod_t, sample.shape)
        
        # Compute predicted original sample
        if self.prediction_type == "epsilon":
            # Noise prediction
            # Ensure compatible shapes for subtraction and division
            sqrt_beta_prod_t = tf.sqrt(beta_prod_t)
            sqrt_alpha_prod_t = tf.sqrt(alpha_prod_t)
            
            # Ensure model_output is compatible with sample
            if model_output.shape != sample.shape:
                # Split into unconditional and conditional if batch dimension is different
                if model_output.shape[0] == 2 and len(model_output.shape) == 3:
                    model_output_uncond, model_output_text = tf.split(model_output, 2, axis=0)
                    model_output_uncond = tf.reduce_mean(model_output_uncond, axis=1)
                    model_output_text = tf.reduce_mean(model_output_text, axis=1)
                    
                    model_output_uncond = tf.reshape(
                        model_output_uncond, 
                        (1, sample.shape[1], sample.shape[2], -1)
                    )
                    model_output_text = tf.reshape(
                        model_output_text, 
                        (1, sample.shape[1], sample.shape[2], -1)
                    )
                    
                    # Resize if needed
                    if model_output_uncond.shape[-1] != sample.shape[-1]:
                        model_output_uncond = tf.image.resize(
                            model_output_uncond, 
                            (sample.shape[1], sample.shape[2]), 
                            method=tf.image.ResizeMethod.BILINEAR
                        )
                        model_output_text = tf.image.resize(
                            model_output_text, 
                            (sample.shape[1], sample.shape[2]), 
                            method=tf.image.ResizeMethod.BILINEAR
                        )
                    
                    # Recombine
                    model_output = tf.concat([model_output_uncond, model_output_text], axis=0)
                
                # Fallback resize if still not matching
                if model_output.shape != sample.shape:
                    model_output = tf.image.resize(
                        model_output, 
                        (sample.shape[1], sample.shape[2]), 
                        method=tf.image.ResizeMethod.BILINEAR
                    )
                
                # Ensure last dimension matches
                if model_output.shape[-1] != sample.shape[-1]:
                    model_output = model_output[..., :sample.shape[-1]]
            
            pred_original_sample = (
                sample - sqrt_beta_prod_t * model_output
            ) / sqrt_alpha_prod_t
        elif self.prediction_type == "sample":
            # Direct sample prediction
            pred_original_sample = model_output
        else:
            raise ValueError(f"Unsupported prediction type: {self.prediction_type}")
        
        # Compute variance
        variance = (
            (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * 
            (1 - alpha_prod_t / alpha_prod_t_prev)
        ) ** 0.5
        
        # Compute standard deviation
        std_dev = variance * model_output
        
        # Compute next sample
        sqrt_alpha_prod_t_prev = tf.sqrt(alpha_prod_t_prev)
        next_sample = (
            pred_original_sample * sqrt_alpha_prod_t_prev + 
            std_dev
        )
        
        # Clip sample if needed
        if self.clip_sample:
            next_sample = tf.clip_by_value(next_sample, -1, 1)
        
        # Cast back to original dtype of sample
        next_sample = tf.cast(next_sample, sample.dtype)
        pred_original_sample = tf.cast(pred_original_sample, sample.dtype)
        
        # Return results
        if return_dict:
            return {
                "prev_sample": next_sample,
                "pred_original_sample": pred_original_sample,
            }
        return next_sample

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

        for i in tqdm(range(self.num_inference_steps)):
            # Update timesteps efficiently
            timesteps = tf.fill([batch_size], self.timesteps[i])
            
            # Run model step
            with tf.GradientTape() as tape:
                pred, cache = func(z, timesteps, cache=cache, **model_kwargs)
            
            # Update z with better precision
            z = tf.add(z, (self.timesteps[i + 1] - self.timesteps[i]) * pred)

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
