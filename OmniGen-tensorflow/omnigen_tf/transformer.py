import math
import warnings
from typing import List, Optional, Tuple, Union, Dict

import tensorflow as tf
from tensorflow.keras import layers, Model
from transformers import Phi3Config
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

logger = logging.get_logger(__name__)


class OptimizedMultiHeadAttention(tf.keras.layers.Layer):
    """Optimized multi-head attention implementation."""
    
    def __init__(self, config, dtype=tf.float32):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.hidden_size = config.hidden_size
        self.dropout = getattr(config, 'attention_dropout', 0.0)
        self._dtype = dtype
        
        # Initialize QKV projections
        self.q_proj = layers.Dense(self.hidden_size, use_bias=False, dtype=self._dtype)
        self.k_proj = layers.Dense(self.hidden_size, use_bias=False, dtype=self._dtype)
        self.v_proj = layers.Dense(self.hidden_size, use_bias=False, dtype=self._dtype)
        self.out_proj = layers.Dense(self.hidden_size, use_bias=True, dtype=self._dtype)
        
    @tf.function(jit_compile=True)
    def _split_heads(self, tensor):
        """Split heads for parallel computation."""
        batch_size = tf.shape(tensor)[0]
        tensor = tf.reshape(tensor, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(tensor, perm=[0, 2, 1, 3])
        
    @tf.function(jit_compile=True)
    def call(self, hidden_states, attention_mask=None, past_key_value=None, 
             use_cache=False, training=False):
        """Optimized attention computation."""
        batch_size = tf.shape(hidden_states)[0]
        seq_length = tf.shape(hidden_states)[1]
        
        # Compute Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Handle cached key/values
        if past_key_value is not None:
            key_states = tf.concat([past_key_value[0], key_states], axis=1)
            value_states = tf.concat([past_key_value[1], value_states], axis=1)
            
        if use_cache:
            present = (key_states, value_states)
        else:
            present = None
            
        # Split heads
        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)
        
        # Compute attention scores efficiently
        scale = tf.cast(1.0 / tf.math.sqrt(tf.cast(self.head_dim, self._dtype)), self._dtype)
        attention_scores = tf.matmul(query_states, key_states, transpose_b=True) * scale
        
        # Apply attention mask
        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, self._dtype)
            attention_scores = tf.where(attention_mask, attention_scores, 
                                      tf.fill(tf.shape(attention_scores), tf.cast(-1e9, self._dtype)))
            
        # Compute attention probabilities
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        attention_probs = tf.keras.layers.Dropout(self.dropout)(attention_probs, training=training)
        
        # Compute context
        context = tf.matmul(attention_probs, value_states)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, seq_length, self.hidden_size))
        
        # Output projection
        outputs = self.out_proj(context)
        
        return outputs, present


class OptimizedDynamicCache:
    """Optimized dynamic cache for transformer layers."""
    
    def __init__(self):
        self.cache = {}
        self._prefetch_stream = None
        self._current_device = None
        self._is_gpu_available = tf.config.list_physical_devices('GPU')
        
    def update(self, layer_idx: int, key_value: Tuple[tf.Tensor, tf.Tensor]):
        """Update cache with new key-value pair."""
        if layer_idx not in self.cache:
            self.cache[layer_idx] = []
        self.cache[layer_idx].append(key_value)
        
    def get(self, layer_idx: int) -> Optional[Tuple[tf.Tensor, tf.Tensor]]:
        """Get cached values for layer."""
        return self.cache.get(layer_idx)
        
    @tf.function(jit_compile=True)
    def prefetch_layer(self, layer_idx: int):
        """Efficient layer prefetching."""
        if not self._is_gpu_available or layer_idx not in self.cache:
            return
            
        if self._prefetch_stream is None:
            self._prefetch_stream = tf.device('/GPU:0')
            
        with self._prefetch_stream:
            # Batch transfer weights
            for key_value in self.cache[layer_idx]:
                # Non-blocking transfer
                tf.identity(key_value[0])
                tf.identity(key_value[1])
                
    @tf.function(jit_compile=True)
    def evict_layer(self, layer_idx: int):
        """Efficient layer eviction."""
        if layer_idx not in self.cache:
            return
            
        with tf.device('/CPU:0'):
            for i, key_value in enumerate(self.cache[layer_idx]):
                # Move to CPU efficiently
                self.cache[layer_idx][i] = (
                    tf.identity(key_value[0]),
                    tf.identity(key_value[1])
                )


class Phi3DecoderLayer(tf.keras.layers.Layer):
    """Optimized decoder layer implementation."""
    
    def __init__(self, config: Phi3Config, dtype=None):
        super().__init__()
        self.config = config
        self._dtype = dtype or tf.float32
        
        # Initialize attention
        self.self_attn = OptimizedMultiHeadAttention(config, dtype=self._dtype)
        
        # Initialize MLP with optimal chunk size
        self.mlp = tf.keras.Sequential([
            layers.Dense(
                config.intermediate_size,
                activation=self._gelu,
                dtype=self._dtype
            ),
            layers.Dense(config.hidden_size, dtype=self._dtype),
            layers.Dropout(getattr(config, 'hidden_dropout', 0.1))
        ])
        
        # Layer norms
        self.input_layernorm = layers.LayerNormalization(
            epsilon=getattr(config, 'layer_norm_eps', 1e-5),
            dtype=self._dtype
        )
        self.post_attention_layernorm = layers.LayerNormalization(
            epsilon=getattr(config, 'layer_norm_eps', 1e-5),
            dtype=self._dtype
        )
        
    @staticmethod
    @tf.function(jit_compile=True)
    def _gelu(x):
        """Optimized GELU implementation."""
        return 0.5 * x * (1.0 + tf.math.tanh(
            tf.cast(0.7978845608028654, x.dtype) * 
            (x + tf.cast(0.044715, x.dtype) * tf.pow(x, 3))
        ))
        
    @tf.function(jit_compile=True)
    def call(self, hidden_states, attention_mask=None, past_key_value=None,
             use_cache=False, training=False):
        """Optimized forward pass."""
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        attention_outputs, present = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            training=training
        )
        
        hidden_states = attention_outputs + residual
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, training=training)
        hidden_states = hidden_states + residual
        
        outputs = (hidden_states,)
        if use_cache:
            outputs += (present,)
            
        return outputs


class Phi3Transformer(tf.keras.Model):
    """Optimized transformer implementation."""
    
    def __init__(self, config: Phi3Config, dtype=tf.float32):
        super().__init__()
        self.config = config
        self._dtype = dtype
        
        # Initialize layers
        self.decoder_layers = [
            Phi3DecoderLayer(config, dtype=self._dtype)
            for _ in range(config.num_hidden_layers)
        ]
        
        self.norm = layers.LayerNormalization(
            epsilon=getattr(config, 'layer_norm_eps', 1e-5),
            dtype=self._dtype
        )
        
        # Cache management
        self.cache = OptimizedDynamicCache()
        
    def _get_optimal_chunk_size(self, total_size):
        """Calculate optimal chunk size based on available memory."""
        if not tf.config.list_physical_devices('GPU'):
            return min(32, total_size)
            
        # Estimate based on model size and available GPU memory
        return min(64, total_size)
        
    @tf.function(jit_compile=True)
    def call(self, inputs_embeds, attention_mask=None, position_ids=None,
             past_key_values=None, use_cache=False, training=False):
        """Optimized forward pass with memory management."""
        
        # Process in optimal chunks
        batch_size = tf.shape(inputs_embeds)[0]
        chunk_size = self._get_optimal_chunk_size(batch_size)
        
        all_hidden_states = []
        all_presents = [] if use_cache else None
        
        for i in range(0, batch_size, chunk_size):
            end_idx = tf.minimum(i + chunk_size, batch_size)
            chunk_inputs = inputs_embeds[i:end_idx]
            
            if attention_mask is not None:
                chunk_mask = attention_mask[i:end_idx]
            else:
                chunk_mask = None
                
            hidden_states = chunk_inputs
            presents = [] if use_cache else None
            
            # Process through layers
            for idx, layer in enumerate(self.decoder_layers):
                # Prefetch next layer
                if idx < len(self.decoder_layers) - 1:
                    self.cache.prefetch_layer(idx + 1)
                    
                past_key_value = past_key_values[idx] if past_key_values is not None else None
                
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=chunk_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    training=training
                )
                
                hidden_states = layer_outputs[0]
                
                if use_cache:
                    presents.append(layer_outputs[1])
                    
                # Evict previous layer
                if idx > 0:
                    self.cache.evict_layer(idx - 1)
                    
            # Final layer norm
            hidden_states = self.norm(hidden_states)
            
            all_hidden_states.append(hidden_states)
            if use_cache:
                all_presents.append(presents)
                
        # Combine results
        hidden_states = tf.concat(all_hidden_states, axis=0)
        
        if use_cache:
            presents = tuple(zip(*all_presents))
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=presents
            )
            
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)
