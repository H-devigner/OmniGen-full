import tensorflow as tf
import numpy as np
from tqdm import tqdm


class TensorFlowCache:
    def __init__(self, num_tokens_for_img, offload_to_cpu=False):
        self.key_cache = []
        self.value_cache = []
        self.original_device = []
        self.num_tokens_for_img = num_tokens_for_img
        self.offload_to_cpu = offload_to_cpu

    def update(self, key_states, value_states, layer_idx):
        if len(self.key_cache) <= layer_idx:
            # Initialize cache for the new layer
            self.key_cache.append(key_states[..., :-(self.num_tokens_for_img + 1), :])
            self.value_cache.append(value_states[..., :-(self.num_tokens_for_img + 1), :])
            self.original_device.append(key_states.device)
        else:
            # Append to existing cache
            self.key_cache[layer_idx] = tf.concat(
                [self.key_cache[layer_idx], key_states], axis=-2
            )
            self.value_cache[layer_idx] = tf.concat(
                [self.value_cache[layer_idx], value_states], axis=-2
            )

    def get_cache(self, layer_idx):
        if layer_idx >= len(self.key_cache):
            raise ValueError("Invalid layer index for cache retrieval.")
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def evict_previous_layer(self, layer_idx):
        if self.offload_to_cpu and layer_idx > 0:
            prev_idx = layer_idx - 1
            self.key_cache[prev_idx] = tf.identity(self.key_cache[prev_idx].numpy())
            self.value_cache[prev_idx] = tf.identity(self.value_cache[prev_idx].numpy())

    def prefetch_layer(self, layer_idx):
        if layer_idx < len(self.key_cache):
            self.key_cache[layer_idx] = tf.convert_to_tensor(self.key_cache[layer_idx])
            self.value_cache[layer_idx] = tf.convert_to_tensor(self.value_cache[layer_idx])


class OmniGenScheduler:
    def __init__(self, num_steps=50, time_shifting_factor=1):
        self.num_steps = num_steps
        t = np.linspace(0, 1, num_steps + 1)
        t = t / (t + time_shifting_factor - time_shifting_factor * t)
        self.sigma = tf.constant(t, dtype=tf.float32)

    def crop_cache(self, cache, num_tokens_for_img):
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i][..., :-(num_tokens_for_img + 1), :]
            cache.value_cache[i] = cache.value_cache[i][..., :-(num_tokens_for_img + 1), :]
        return cache

    def crop_position_ids_for_cache(self, position_ids, num_tokens_for_img):
        return position_ids[:, -(num_tokens_for_img + 1):]

    def crop_attention_mask_for_cache(self, attention_mask, num_tokens_for_img):
        return attention_mask[..., -(num_tokens_for_img + 1):, :]

    def __call__(self, z, func, model_kwargs, use_kv_cache=True, offload_kv_cache=True):
        num_tokens_for_img = z.shape[-1] * z.shape[-2] // 4
        cache = TensorFlowCache(num_tokens_for_img, offload_to_cpu=offload_kv_cache) if use_kv_cache else None

        for i in tqdm(range(self.num_steps)):
            timesteps = tf.fill([tf.shape(z)[0]], self.sigma[i])
            pred, cache = func(z, timesteps, cache=cache, **model_kwargs)
            sigma_next = self.sigma[i + 1]
            sigma = self.sigma[i]
            z += (sigma_next - sigma) * pred

            if i == 0 and use_kv_cache:
                model_kwargs["position_ids"] = self.crop_position_ids_for_cache(
                    model_kwargs["position_ids"], num_tokens_for_img
                )
                model_kwargs["attention_mask"] = self.crop_attention_mask_for_cache(
                    model_kwargs["attention_mask"], num_tokens_for_img
                )

        return z
