import math
import warnings
from typing import List, Optional, Tuple, Union, Dict

import tensorflow as tf
from tensorflow.keras import layers, Model
from transformers import Phi3Config
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Cache:
    """Base cache class for key-value caching in attention layers."""
    def __init__(self):
        self.cache = {}
        self._seen_tokens = 0

    def get_seq_length(self) -> int:
        """Get the sequence length of the cached keys."""
        if not self.cache:
            return 0
        return next(iter(self.cache.values()))[0].shape[1]

    def update(self, key: str, key_value: Tuple[tf.Tensor, tf.Tensor], layer_idx: int):
        """Update the cache with new key-value pairs."""
        if layer_idx not in self.cache:
            self.cache[layer_idx] = {}
        self.cache[layer_idx][key] = key_value


class DynamicCache(Cache):
    """Dynamic cache that grows with each forward pass."""
    
    def __init__(self, num_tokens_for_img: int = None, offload_kv_cache: bool = False):
        super().__init__()
        self.key_cache = []
        self.value_cache = []
        self.original_device = []
        self.num_tokens_for_img = num_tokens_for_img
        self.offload_kv_cache = offload_kv_cache
        self._active_stream = None
        
    def update(self, key_states: tf.Tensor, value_states: tf.Tensor, layer_idx: int):
        """Concatenate new key-value pairs with existing ones."""
        if layer_idx >= len(self.key_cache):
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.original_device.append(key_states.device)
        else:
            with tf.device(key_states.device):
                self.key_cache[layer_idx] = tf.concat([self.key_cache[layer_idx], key_states], axis=-2)
                self.value_cache[layer_idx] = tf.concat([self.value_cache[layer_idx], value_states], axis=-2)
                
    def __getitem__(self, layer_idx: int):
        """Gets the cache for this layer, handling device placement."""
        if layer_idx >= len(self.key_cache):
            raise IndexError(f"Cache index {layer_idx} out of range")
            
        # Prefetch next layer
        if layer_idx + 1 < len(self.key_cache):
            self.prefetch_layer(layer_idx + 1)
            
        # Evict previous layer
        if layer_idx > 0:
            self.evict_previous_layer(layer_idx - 1)
            
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
        
    def prefetch_layer(self, layer_idx: int):
        """Prefetch next layer to device."""
        if not self.offload_kv_cache or layer_idx >= len(self.key_cache):
            return
            
        if tf.config.list_physical_devices('GPU'):
            # Start async prefetch
            self._active_stream = tf.device('/GPU:0')
            with self._active_stream:
                self.prefetch_async(self.key_cache[layer_idx])
                self.prefetch_async(self.value_cache[layer_idx])
                
    def prefetch_async(self, tensor):
        """Async prefetch helper."""
        if tensor.device != '/GPU:0':
            with tf.device('/GPU:0'):
                tf.identity(tensor)  # Force async copy
                
    def evict_previous_layer(self, layer_idx: int):
        """Move previous layer to CPU."""
        if not self.offload_kv_cache or layer_idx < 0:
            return
            
        # Wait for any active prefetch
        if self._active_stream is not None:
            with self._active_stream:
                pass  # Synchronize
            self._active_stream = None
            
        # Move to CPU asynchronously
        if tf.config.list_physical_devices('GPU'):
            with tf.device('/CPU:0'):
                self.evict_async(self.key_cache[layer_idx])
                self.evict_async(self.value_cache[layer_idx])
                
    def evict_async(self, tensor):
        """Async eviction helper."""
        if tensor.device != '/CPU:0':
            with tf.device('/CPU:0'):
                tf.identity(tensor)  # Force async copy


class StaticCache(Cache):
    """Static cache with fixed size."""
    def update(self, key: str, key_value: Tuple[tf.Tensor, tf.Tensor], layer_idx: int):
        """Update cache without growing."""
        if layer_idx not in self.cache:
            self.cache[layer_idx] = {}
        self.cache[layer_idx][key] = key_value


class Phi3Transformer(tf.keras.Model):
    """Transformer decoder with memory optimizations."""
    
    def __init__(self, config, dtype=None):
        super().__init__()
        self.config = config
        self._dtype = dtype or tf.float32
        self.num_hidden_layers = config.num_hidden_layers
        
        # Memory optimization flags
        self._active_layer = None
        self._prefetch_queue = []
        self._evict_queue = []
        
        # Initialize layers
        self.decoder_layers = []
        for i in range(config.num_hidden_layers):
            layer = Phi3DecoderLayer(config, dtype=self._dtype)
            self.decoder_layers.append(layer)
            
        self.norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-5,
            dtype=self._dtype
        )
        
    def prefetch_layer(self, layer_idx: int):
        """Prefetch layer weights to GPU."""
        if layer_idx >= self.num_hidden_layers:
            return
            
        if tf.config.list_physical_devices('GPU'):
            layer = self.decoder_layers[layer_idx]
            self._prefetch_queue.append(layer)
            
            # Start async prefetch
            with tf.device('/GPU:0'):
                for weight in layer.weights:
                    tf.identity(weight)
                    
    def evict_layer(self, layer_idx: int):
        """Move layer weights to CPU."""
        if layer_idx < 0:
            return
            
        if tf.config.list_physical_devices('GPU'):
            layer = self.decoder_layers[layer_idx]
            self._evict_queue.append(layer)
            
            # Start async eviction
            with tf.device('/CPU:0'):
                for weight in layer.weights:
                    tf.identity(weight)
                    
    def _update_causal_mask(
        self, 
        attention_mask: Optional[tf.Tensor],
        input_shape: tf.TensorShape,
        dtype: tf.dtypes.DType,
        past_key_values_length: int = 0
    ) -> tf.Tensor:
        """
        Update the causal mask for auto-regressive decoding.
        
        Args:
            attention_mask: Optional attention mask
            input_shape: Shape of input
            dtype: Data type of mask
            past_key_values_length: Length of past key-values for caching
            
        Returns:
            Updated causal attention mask
        """
        batch_size, seq_length = input_shape[0], input_shape[1]
        
        if attention_mask is None:
            attention_mask = tf.ones((batch_size, seq_length))
            
        # Create causal mask
        # tf.experimental.numpy.triu is similar to torch.triu
        causal_mask = 1 - tf.experimental.numpy.triu(
            tf.ones((seq_length, seq_length)), k=1
        )
        causal_mask = tf.cast(causal_mask[None, None, :, :], dtype)
        
        if past_key_values_length > 0:
            causal_mask = tf.pad(
                causal_mask,
                [[0, 0], [0, 0], [0, 0], [past_key_values_length, 0]]
            )
            
        if attention_mask is not None:
            # Extend attention mask for sequence length
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = tf.cast(extended_attention_mask, dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * tf.dtypes.as_dtype(dtype).min
            causal_mask = causal_mask + extended_attention_mask
            
        return causal_mask

    def call(self, inputs_embeds, attention_mask=None, position_ids=None,
             past_key_values=None, use_cache=None, output_attentions=None,
             output_hidden_states=None, return_dict=None, training=False):
        """Forward pass with memory optimizations."""
        
        # Handle defaults
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = (
            output_attentions if output_attentions is not None 
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None 
            else self.config.use_return_dict
        )
        
        # Initialize caches
        if past_key_values is None:
            past_key_values = tuple([None] * self.num_hidden_layers)
            
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        hidden_states = inputs_embeds
        
        # Process through layers with memory optimization
        for idx, decoder_layer in enumerate(self.decoder_layers):
            # Memory management
            self._active_layer = idx
            if idx + 1 < self.num_hidden_layers:
                self.prefetch_layer(idx + 1)
            if idx > 0:
                self.evict_layer(idx - 1)
                
            # Layer processing
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[idx] if past_key_values is not None else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
                training=training,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
                
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Clear active layer and queues
        self._active_layer = None
        self._prefetch_queue.clear()
        self._evict_queue.clear()
        
        # Return results
        if not return_dict:
            return tuple(v for v in [
                hidden_states,
                next_decoder_cache,
                all_hidden_states,
                all_self_attns,
            ] if v is not None)
            
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Phi3DecoderLayer(tf.keras.Model):
    """Decoder layer for Phi3 model with self-attention and feed-forward networks."""
    
    def __init__(self, config: Phi3Config, dtype=None, **kwargs):
        super().__init__(**kwargs)
        print(f"Initializing Phi3DecoderLayer with config..")
        
        self.config = config
        
        # Get configuration values with defaults
        layer_norm_eps = getattr(config, 'layer_norm_eps', 1e-5)
        attention_dropout = getattr(config, 'attention_dropout', 0.0)
        hidden_dropout = getattr(config, 'hidden_dropout_prob', 0.1)
        
        # Initialize attention layer
        print("Creating self-attention layer...")
        self.self_attn = layers.MultiHeadAttention(
            num_heads=config.num_attention_heads,
            key_dim=config.hidden_size // config.num_attention_heads,
            dropout=attention_dropout,
            name="self_attn"
        )
        
        # Initialize MLP
        print("Creating MLP layers...")
        self.mlp = tf.keras.Sequential([
            layers.Dense(config.intermediate_size, activation="gelu", name="fc1"),
            layers.Dense(config.hidden_size, name="fc2"),
            layers.Dropout(hidden_dropout),
        ], name="mlp")
        
        # Initialize layer norms
        print("Creating layer normalizations...")
        self.input_layernorm = layers.LayerNormalization(
            epsilon=layer_norm_eps, 
            name="input_layernorm"
        )
        self.post_attention_layernorm = layers.LayerNormalization(
            epsilon=layer_norm_eps, 
            name="post_attention_layernorm"
        )
        
        print("Phi3DecoderLayer initialization completed")
        print(f"Attention heads: {config.num_attention_heads}")
        print(f"Hidden size: {config.hidden_size}")
        print(f"Intermediate size: {config.intermediate_size}")
        print(f"Layer norm epsilon: {layer_norm_eps}")
        print(f"Attention dropout: {attention_dropout}")
        print(f"Hidden dropout: {hidden_dropout}")

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_value: Optional[Tuple[tf.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor, ...]:
        """
        Forward pass of decoder layer.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_value: Past key-value states
            output_attentions: Whether to output attention weights
            use_cache: Whether to use caching
            training: Whether in training mode
            
        Returns:
            Tuple of output tensor and optional cache
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs = self.self_attn(
            query=hidden_states,
            value=hidden_states,
            key=hidden_states,
            attention_mask=attention_mask,
            training=training,
        )
        hidden_states = residual + attn_outputs

        # Feed Forward
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, training=training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_outputs,)
        if use_cache:
            outputs += (past_key_value,)

        return outputs
