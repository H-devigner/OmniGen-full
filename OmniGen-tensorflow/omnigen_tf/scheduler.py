"""OmniGen TensorFlow Scheduler with exact PyTorch equivalence."""

import tensorflow as tf
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
import gc
from tqdm import tqdm

class OmniGenCache:
    """TensorFlow implementation of dynamic cache with PyTorch equivalence."""
    
    def __init__(self, num_tokens_for_img: int, offload_kv_cache: bool = False):
        """Initialize cache.
        
        Args:
            num_tokens_for_img: Number of tokens for image
            offload_kv_cache: Whether to offload KV cache to CPU
        """
        if not tf.config.list_physical_devices('GPU'):
            raise RuntimeError(
                "OffloadedCache can only be used with a GPU. If there is no GPU, "
                "you need to set use_kv_cache=False, which will result in longer inference time!"
            )
            
        self.key_cache = []
        self.value_cache = []
        self.original_device = []
        self.num_tokens_for_img = num_tokens_for_img
        self.offload_kv_cache = offload_kv_cache
        
    def prefetch_layer(self, layer_idx: int):
        """Prefetch next layer cache to GPU."""
        if layer_idx < len(self.key_cache):
            # Use tf.function for async execution
            @tf.function(jit_compile=True)
            def _prefetch():
                device = self.original_device[layer_idx]
                with tf.device(device):
                    self.key_cache[layer_idx] = tf.identity(self.key_cache[layer_idx])
                    self.value_cache[layer_idx] = tf.identity(self.value_cache[layer_idx])
            
            _prefetch()
            
    def evict_previous_layer(self, layer_idx: int):
        """Move previous layer cache to CPU."""
        if len(self.key_cache) > 2:
            prev_layer_idx = -1 if layer_idx == 0 else (layer_idx - 1) % len(self.key_cache)
            
            @tf.function(jit_compile=True)
            def _evict():
                with tf.device("/CPU:0"):
                    self.key_cache[prev_layer_idx] = tf.identity(self.key_cache[prev_layer_idx])
                    self.value_cache[prev_layer_idx] = tf.identity(self.value_cache[prev_layer_idx])
                    
            _evict()
            
    def __getitem__(self, layer_idx: int) -> List[Tuple[tf.Tensor]]:
        """Get cache for layer with memory optimization."""
        if layer_idx < len(self.key_cache):
            if self.offload_kv_cache:
                # Ensure previous operations are complete
                tf.keras.backend.clear_session()
                self.evict_previous_layer(layer_idx)
                
                # Move current layer to GPU
                device = self.original_device[layer_idx]
                with tf.device(device):
                    self.key_cache[layer_idx] = tf.identity(self.key_cache[layer_idx])
                    self.value_cache[layer_idx] = tf.identity(self.value_cache[layer_idx])
                    
                # Prefetch next layer
                next_layer_idx = (layer_idx + 1) % len(self.key_cache)
                self.prefetch_layer(next_layer_idx)
                
            return [(self.key_cache[layer_idx], self.value_cache[layer_idx])]
            
        return []
        
    def update(self, key_states: tf.Tensor, value_states: tf.Tensor, layer_idx: int):
        """Update cache for layer with memory optimization."""
        if layer_idx >= len(self.key_cache):
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.original_device.append(key_states.device)
        else:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
            
    def __len__(self) -> int:
        return len(self.key_cache)


class OmniGenScheduler:
    """TensorFlow scheduler with exact PyTorch equivalence."""
    
    def __init__(self, num_steps: int = 50, time_shifting_factor: int = 1):
        """Initialize scheduler.
        
        Args:
            num_steps: Number of inference steps
            time_shifting_factor: Time shifting factor for sigma calculation
        """
        self.num_steps = num_steps
        self.time_shift = time_shifting_factor
        
        # Match PyTorch initialization exactly
        t = tf.linspace(0.0, 1.0, num_steps + 1)
        t = t / (t + time_shifting_factor - time_shifting_factor * t)
        self.sigma = t
        
    @tf.function(jit_compile=True)
    def crop_kv_cache(self, past_key_values, num_tokens_for_img):
        """Crop KV cache with XLA optimization."""
        if past_key_values is None:
            return None
            
        cropped = []
        for layer_past in past_key_values:
            if isinstance(layer_past, tuple):
                cropped.append((
                    layer_past[0][:, :num_tokens_for_img, :],
                    layer_past[1][:, :num_tokens_for_img, :]
                ))
            else:
                cropped.append(layer_past)
                
        return cropped
        
    @tf.function(jit_compile=True)
    def crop_position_ids_for_cache(self, position_ids, num_tokens_for_img):
        """Crop position IDs with XLA optimization."""
        if position_ids is None:
            return None
        return position_ids[:, :num_tokens_for_img]
        
    @tf.function(jit_compile=True)
    def crop_attention_mask_for_cache(self, attention_mask, num_tokens_for_img):
        """Crop attention mask with XLA optimization."""
        if attention_mask is None:
            return None
        return attention_mask[:, :, :num_tokens_for_img]
        
    @tf.function(jit_compile=True)
    def crop_cache(self, cache, num_tokens_for_img):
        """Crop cache with XLA optimization."""
        if not isinstance(cache, OmniGenCache):
            return cache
            
        for i in range(len(cache)):
            cache.key_cache[i] = cache.key_cache[i][:, :num_tokens_for_img, :]
            cache.value_cache[i] = cache.value_cache[i][:, :num_tokens_for_img, :]
            
        return cache
        
    def __call__(self, z, func, model_kwargs, use_kv_cache: bool = True, offload_kv_cache: bool = True):
        """Run diffusion process with memory optimization."""
        batch_size = tf.shape(z)[0]
        
        # Initialize progress bar
        pbar = tqdm(range(self.num_steps))
        pbar.set_description("Denoising")
        
        # Initialize cache if needed
        if use_kv_cache:
            num_tokens_for_img = tf.shape(model_kwargs["attention_mask"])[1]
            cache = OmniGenCache(num_tokens_for_img, offload_kv_cache)
            model_kwargs["past_key_values"] = cache
            
        # Run diffusion steps
        for i in pbar:
            t = tf.ones([batch_size]) * i
            
            # Get model output
            with tf.GradientTape(persistent=False) as tape:
                model_output = func(z, t, **model_kwargs)
                
            # Update z
            z = z - model_output * (self.sigma[i + 1] - self.sigma[i])
            
            # Memory cleanup
            if i < self.num_steps - 1:
                tf.keras.backend.clear_session()
                gc.collect()
                
        return z
