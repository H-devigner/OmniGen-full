"""OmniGen TensorFlow Scheduler

This module provides the TensorFlow implementation of the OmniGen scheduler,
matching the PyTorch version's functionality while leveraging TensorFlow-specific optimizations.
"""

from typing import Optional, Dict, Any, Tuple, List
import gc
from tqdm import tqdm

import tensorflow as tf


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
        self.original_device = []
        self._seen_tokens = 0
        self.num_tokens_for_img = num_tokens_for_img
        self.offload_kv_cache = offload_kv_cache

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

            # Initialize cache for new layer
            with tf.device("/CPU:0" if self.offload_kv_cache else None):
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
                self.original_device.append(key_states.device)
        else:
            # Update existing cache
            self.key_cache[layer_idx] = tf.concat(
                [self.key_cache[layer_idx], key_states], axis=-2
            )
            self.value_cache[layer_idx] = tf.concat(
                [self.value_cache[layer_idx], value_states], axis=-2
            )

        return key_states, value_states

    def __getitem__(self, layer_idx: int) -> List[Tuple[tf.Tensor]]:
        """Get cache for layer, handling prefetch and eviction."""
        if layer_idx < len(self.key_cache):
            if self.offload_kv_cache:
                # Evict previous layer
                self.evict_previous_layer(layer_idx)
                
                # Load current layer
                key_tensor = tf.identity(self.key_cache[layer_idx])
                value_tensor = tf.identity(self.value_cache[layer_idx])
                
                # Prefetch next layer
                self.prefetch_layer((layer_idx + 1) % len(self.key_cache))
            else:
                key_tensor = self.key_cache[layer_idx]
                value_tensor = self.value_cache[layer_idx]
            return (key_tensor, value_tensor)
        else:
            raise KeyError(f"Cache only has {len(self.key_cache)} layers, attempted to access layer {layer_idx}")

    def evict_previous_layer(self, layer_idx: int) -> None:
        """Move previous layer cache to CPU."""
        if len(self.key_cache) > 2:
            prev_layer_idx = -1 if layer_idx == 0 else (layer_idx - 1) % len(self.key_cache)
            with tf.device("/CPU:0"):
                self.key_cache[prev_layer_idx] = tf.identity(self.key_cache[prev_layer_idx])
                self.value_cache[prev_layer_idx] = tf.identity(self.value_cache[prev_layer_idx])

    def prefetch_layer(self, layer_idx: int) -> None:
        """Prefetch next layer to GPU."""
        if layer_idx < len(self.key_cache):
            with tf.device("/GPU:0"):
                self.key_cache[layer_idx] = tf.identity(self.key_cache[layer_idx])
                self.value_cache[layer_idx] = tf.identity(self.value_cache[layer_idx])


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

        t = tf.linspace(0.0, 1.0, num_steps + 1)
        t = t / (t + time_shifting_factor - time_shifting_factor * t)
        self.sigma = t

    def crop_kv_cache(self, past_key_values, num_tokens_for_img):
        """Crop key-value cache."""
        for i in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[i] = past_key_values.key_cache[i][..., :-(num_tokens_for_img + 1), :]
            past_key_values.value_cache[i] = past_key_values.value_cache[i][..., :-(num_tokens_for_img + 1), :]
        return past_key_values

    def crop_position_ids_for_cache(self, position_ids, num_tokens_for_img):
        """Crop position IDs for cache."""
        return position_ids[:, -(num_tokens_for_img + 1):]

    def crop_attention_mask_for_cache(self, attention_mask, num_tokens_for_img):
        """Crop attention mask for cache."""
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
        num_tokens_for_img = tf.shape(z)[-1] * tf.shape(z)[-2] // 4
        cache = OmniGenCache(num_tokens_for_img, offload_kv_cache) if use_kv_cache else None

        for i in tqdm(range(self.num_steps)):
            timesteps = tf.fill([tf.shape(z)[0]], self.sigma[i])
            pred, cache = func(z, timesteps, cache=cache, **model_kwargs)
            sigma_next = self.sigma[i + 1]
            sigma = self.sigma[i]
            z += (sigma_next - sigma) * pred

            if i == 0 and use_kv_cache:
                # Update model kwargs for caching
                model_kwargs["position_ids"] = self.crop_position_ids_for_cache(
                    model_kwargs["position_ids"], num_tokens_for_img
                )
                model_kwargs["attention_mask"] = self.crop_attention_mask_for_cache(
                    model_kwargs["attention_mask"], num_tokens_for_img
                )

            # Clean up GPU memory
            if use_kv_cache and i > 0:
                gc.collect()
                if tf.config.list_physical_devices('GPU'):
                    for device in tf.config.list_physical_devices('GPU'):
                        tf.config.experimental.reset_memory_stats(device)

        return z
