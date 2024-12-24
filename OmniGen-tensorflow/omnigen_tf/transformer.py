import math
import warnings
from typing import List, Optional, Tuple, Union

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from transformers import TFPhi3Model, Phi3Config
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging

logger = logging.get_logger(__name__)

class Phi3Transformer(TFPhi3Model):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers.
    We only modified the attention mask and added memory optimization.
    Args:
        config: Phi3Config
    """
    def __init__(self, config: Phi3Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._is_gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        
        # Setup memory optimization
        if self._is_gpu_available:
            for device in tf.config.list_physical_devices('GPU'):
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                    # Set memory limit to 90% of available memory
                    tf.config.set_logical_device_configuration(
                        device,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=int(tf.config.experimental.get_memory_info('GPU:0')['free'] * 0.9))]
                    )
                except:
                    pass
            
            # Use mixed precision for better memory efficiency
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Initialize prefetch stream
        self.prefetch_stream = None
        if self._is_gpu_available:
            tf.config.experimental.set_synchronous_execution(False)
    
    def prefetch_layer(self, layer_idx: int, device: str = '/GPU:0'):
        """Starts prefetching the next layer cache"""
        if not self._is_gpu_available:
            return
            
        layer = self.layers[layer_idx]
        with tf.device(device):
            for weight in layer.trainable_weights:
                tf.identity(weight, name=f"prefetch_{layer_idx}")

    def evict_previous_layer(self, layer_idx: int):
        """Moves the previous layer cache to the CPU"""
        if not self._is_gpu_available or layer_idx <= 0:
            return
            
        prev_layer = self.layers[layer_idx - 1]
        with tf.device('/CPU:0'):
            for weight in prev_layer.trainable_weights:
                tf.identity(weight, name=f"evict_{layer_idx-1}")
        
        # Clear GPU memory
        tf.keras.backend.clear_session()
            
    def get_offload_layer(self, layer_idx: int, device: str = '/GPU:0'):
        """Manages layer offloading for memory efficiency"""
        # Make sure the current layer is ready
        tf.keras.backend.clear_session()
        self.evict_previous_layer(layer_idx)
        
        # Prefetch next layer
        next_layer_idx = (layer_idx + 1) % len(self.layers)
        self.prefetch_layer(next_layer_idx, device)
        
    def _get_chunk_size(self, total_size: int) -> int:
        """Calculate optimal chunk size based on available memory"""
        if not self._is_gpu_available:
            return 16
            
        try:
            gpu_mem = tf.config.experimental.get_memory_info('GPU:0')
            available_mem = gpu_mem['free'] * 0.9  # Use 90% of free memory
            
            # Calculate based on tensor size and dtype
            bytes_per_element = 2  # Using mixed precision (float16)
            size_per_item = total_size * bytes_per_element
            
            # Set chunk size with headroom
            chunk_size = min(32, int(available_mem / (size_per_item * 2)))
            return max(1, chunk_size)
        except:
            return 16
    
    def call(
        self,
        input_ids: Optional[tf.Tensor] = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_values: Optional[List[tf.Tensor]] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[tf.Tensor] = None,
        offload_model: Optional[bool] = False,
        training: bool = False,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # Get defaults from config
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Handle caching
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "Legacy cache format detected. Please use Cache class from transformers.cache_utils"
                )

        # Process inputs
        if attention_mask is not None and len(tf.shape(attention_mask)) == 3:
            attention_mask = (1 - attention_mask) * tf.float32.min
            attention_mask = tf.expand_dims(attention_mask, axis=1)
        
        # Process in chunks for memory efficiency
        batch_size = tf.shape(inputs_embeds)[0]
        chunk_size = self._get_chunk_size(tf.size(inputs_embeds))
        
        # Initialize outputs
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        
        # Process in chunks
        hidden_states_chunks = []
        
        for i in range(0, batch_size, chunk_size):
            end_idx = tf.minimum(i + chunk_size, batch_size)
            
            # Get chunk inputs
            chunk_inputs = tf.identity(inputs_embeds[i:end_idx])
            chunk_mask = tf.identity(attention_mask[i:end_idx]) if attention_mask is not None else None
            chunk_positions = tf.identity(position_ids[i:end_idx]) if position_ids is not None else None
            
            # Process chunk
            hidden_states = chunk_inputs
            
            # Layer-wise processing
            for idx, layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # Memory management for offloading
                if offload_model and not training:
                    self.get_offload_layer(idx, device=tf.device.current_device_name())
                
                # Get layer cache
                past_key_value = past_key_values[idx] if past_key_values is not None else None
                
                # Process layer
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=chunk_mask,
                    position_ids=chunk_positions,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    training=training
                )
                
                hidden_states = layer_outputs[0]
                
                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]
                    
                if output_attentions:
                    all_self_attns = all_self_attns + (layer_outputs[1],)
                
                # Clear intermediates
                if self._is_gpu_available:
                    tf.keras.backend.clear_session()
            
            # Final layer norm
            hidden_states = self.final_layernorm(hidden_states)
            hidden_states_chunks.append(hidden_states)
        
        # Combine chunks
        hidden_states = tf.concat(hidden_states_chunks, axis=0)
        
        # Add final hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Handle cache
        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache and next_cache is not None:
            next_cache = next_cache.to_legacy_cache()
        
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
