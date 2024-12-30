"""OmniGen TensorFlow Scheduler with exact PyTorch equivalence."""

import tensorflow as tf
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
import gc
from tqdm import tqdm

class OmniGenCache:
    """TensorFlow implementation of dynamic cache with PyTorch equivalence."""
    
    def __init__(self, num_tokens_for_img: int, offload_kv_cache: bool = False):
        """Initialize cache with PyTorch equivalence."""
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
        self._seen_tokens = 0
        
    def prefetch_layer(self, layer_idx: int):
        """Prefetch next layer cache to GPU exactly like PyTorch."""
        if layer_idx < len(self.key_cache):
            device = self.original_device[layer_idx]
            with tf.device(device):
                self.key_cache[layer_idx] = tf.identity(self.key_cache[layer_idx])
                self.value_cache[layer_idx] = tf.identity(self.value_cache[layer_idx])
            
    def evict_previous_layer(self, layer_idx: int):
        """Move previous layer cache to CPU exactly like PyTorch."""
        if len(self.key_cache) > 2:
            prev_layer_idx = -1 if layer_idx == 0 else (layer_idx - 1) % len(self.key_cache)
            with tf.device("/CPU:0"):
                self.key_cache[prev_layer_idx] = tf.identity(self.key_cache[prev_layer_idx])
                self.value_cache[prev_layer_idx] = tf.identity(self.value_cache[prev_layer_idx])
            
    def __getitem__(self, layer_idx: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Get cache for layer exactly like PyTorch."""
        if layer_idx < len(self.key_cache):
            if self.offload_kv_cache:
                # Ensure previous operations are complete
                tf.keras.backend.clear_session()
                
                # Evict previous layer
                self.evict_previous_layer(layer_idx)
                
                # Move current layer to original device
                device = self.original_device[layer_idx]
                with tf.device(device):
                    key_tensor = tf.identity(self.key_cache[layer_idx])
                    value_tensor = tf.identity(self.value_cache[layer_idx])
                    
                # Prefetch next layer
                self.prefetch_layer((layer_idx + 1) % len(self.key_cache))
                
                return (key_tensor, value_tensor)
            else:
                return (self.key_cache[layer_idx], self.value_cache[layer_idx])
            
        raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")
        
    def update(self, key_states: tf.Tensor, value_states: tf.Tensor, layer_idx: int,
              cache_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """Update cache exactly like PyTorch."""
        if len(self.key_cache) < layer_idx:
            raise ValueError("Cache does not support skipping layers. Use DynamicCache.")
            
        elif len(self.key_cache) == layer_idx:
            # Only cache states for condition tokens
            key_states = key_states[..., :-(self.num_tokens_for_img + 1), :]
            value_states = value_states[..., :-(self.num_tokens_for_img + 1), :]
            
            # Update seen tokens
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[-2]
                
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.original_device.append(key_states.device)
            
            if self.offload_kv_cache:
                self.evict_previous_layer(layer_idx)
                
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
            
        else:
            # Only cache states for condition tokens
            key_tensor, value_tensor = self[layer_idx]
            k = tf.concat([key_tensor, key_states], axis=-2)
            v = tf.concat([value_tensor, value_states], axis=-2)
            return k, v
            
    def __len__(self) -> int:
        return len(self.key_cache)


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
        timestep_spacing="leading",
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
        self.timestep_spacing = timestep_spacing
        
        # Initialize betas and alphas
        if beta_schedule == "linear":
            self.betas = tf.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "scaled_linear":
            # This schedule is used in the original DDPM paper
            self.betas = tf.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
            
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = tf.math.cumprod(self.alphas)
        
        # Calculate sigmas
        self.init_noise_sigma = tf.cast(tf.math.sqrt(1.0 / self.alphas_cumprod[0] - 1.0), tf.float32)
        
        # Set timesteps
        self.set_timesteps(num_train_timesteps)
        
    def set_timesteps(self, num_inference_steps):
        """Set timesteps for inference."""
        if self.timestep_spacing == "leading":
            timesteps = tf.range(0, self.num_train_timesteps, self.num_train_timesteps // num_inference_steps)
        else:
            timesteps = tf.linspace(0, self.num_train_timesteps - 1, num_inference_steps)
            
        self.timesteps = tf.cast(timesteps, tf.int32)
        
    def scale_model_input(self, sample, timestep):
        """Scale the model input according to the timestep."""
        step_index = tf.where(self.timesteps == timestep)[0][0]
        sigma = tf.math.sqrt(1.0 / self.alphas_cumprod[step_index] - 1.0)
        
        # Scale input
        sample = sample / ((sigma * sigma + 1) ** 0.5)
        return sample
        
    def step(self, model_output, timestep, sample):
        """Predict the sample at the previous timestep."""
        # Get step index
        step_index = tf.where(self.timesteps == timestep)[0][0]
        prev_timestep = self.timesteps[step_index + 1] if step_index < len(self.timesteps) - 1 else 0
        
        # Get alpha values
        alpha = self.alphas_cumprod[timestep]
        alpha_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else tf.ones_like(alpha)
        
        # Get beta value
        beta = 1 - alpha / alpha_prev
        
        # Compute predicted original sample
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta ** (0.5) * model_output) / alpha ** (0.5)
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        else:
            raise ValueError(f"Unknown prediction type {self.prediction_type}")
            
        # Clip predicted sample if needed
        if self.clip_sample:
            pred_original_sample = tf.clip_by_value(pred_original_sample, -1, 1)
            
        # Compute coefficients
        sigma = beta ** (0.5)
        
        # Compute predicted previous sample
        pred_prev_sample = alpha_prev ** (0.5) * pred_original_sample + (1 - alpha_prev) ** (0.5) * model_output
        
        return pred_prev_sample

    def __call__(self, z, func, model_kwargs, use_kv_cache: bool = True, offload_kv_cache: bool = True):
        """Run scheduler with PyTorch equivalence."""
        num_tokens_for_img = z.shape[-1] * z.shape[-2] // 4
        
        # Initialize cache
        if isinstance(model_kwargs['input_ids'], list):
            cache = [OmniGenCache(num_tokens_for_img, offload_kv_cache) 
                    for _ in range(len(model_kwargs['input_ids']))] if use_kv_cache else None
        else:
            cache = OmniGenCache(num_tokens_for_img, offload_kv_cache) if use_kv_cache else None
            
        # Run diffusion steps
        for i in tqdm(range(self.num_train_timesteps)):
            # Match PyTorch's timestep handling
            timesteps = tf.zeros((z.shape[0],), dtype=z.dtype) + self.timesteps[i]
            
            # Run model step
            pred, cache = func(z, timesteps, past_key_values=cache, **model_kwargs)
            
            # Update latents exactly like PyTorch
            z = self.step(pred, timesteps, z)
            
            # Update cache on first step
            if i == 0 and use_kv_cache:
                if isinstance(cache, list):
                    model_kwargs['input_ids'] = [None] * len(cache)
                else:
                    model_kwargs['input_ids'] = None
                    
                model_kwargs['position_ids'] = self.crop_position_ids_for_cache(
                    model_kwargs['position_ids'], num_tokens_for_img)
                model_kwargs['attention_mask'] = self.crop_attention_mask_for_cache(
                    model_kwargs['attention_mask'], num_tokens_for_img)
                    
        # Clean up memory
        del cache
        tf.keras.backend.clear_session()
        gc.collect()
            
        return z

    def crop_kv_cache(self, past_key_values, num_tokens_for_img):
        """Crop KV cache exactly like PyTorch."""
        if past_key_values is None:
            return None
            
        cropped = []
        for layer in past_key_values:
            k, v = layer[0]  # Unpack tuple
            k = k[..., :-(num_tokens_for_img + 1), :]
            v = v[..., :-(num_tokens_for_img + 1), :]
            cropped.append((k, v))
            
        return cropped
        
    def crop_position_ids_for_cache(self, position_ids, num_tokens_for_img):
        """Crop position IDs exactly like PyTorch."""
        if isinstance(position_ids, list):
            return [p[..., -(num_tokens_for_img + 1):] for p in position_ids]
        return position_ids[..., -(num_tokens_for_img + 1):]
        
    def crop_attention_mask_for_cache(self, attention_mask, num_tokens_for_img):
        """Crop attention mask exactly like PyTorch."""
        if isinstance(attention_mask, list):
            return [m[..., -(num_tokens_for_img + 1):, :] for m in attention_mask]
        return attention_mask[..., -(num_tokens_for_img + 1):, :]
        
    def crop_cache(self, cache, num_tokens_for_img):
        """Crop cache exactly like PyTorch."""
        if cache is None:
            return None
            
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i][..., :-(num_tokens_for_img + 1), :]
            cache.value_cache[i] = cache.value_cache[i][..., :-(num_tokens_for_img + 1), :]
            
        return cache
