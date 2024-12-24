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
    """Optimized dynamic cache for transformer layers with prefetching."""
    
    def __init__(self):
        self.cache = {}
        self._prefetch_stream = None
        self._current_device = None
        self._is_gpu_available = tf.config.list_physical_devices('GPU')
        
        if self._is_gpu_available:
            # Create separate stream for prefetching
            self._prefetch_stream = tf.distribute.get_strategy().experimental_run_v2
        
    def update(self, layer_idx: int, key_value: Tuple[tf.Tensor, tf.Tensor]):
        """Update cache with new key-value pair."""
        if layer_idx not in self.cache:
            self.cache[layer_idx] = []
        self.cache[layer_idx].append(key_value)
        
    def get(self, layer_idx: int) -> Optional[Tuple[tf.Tensor, tf.Tensor]]:
        """Get cached values for layer."""
        return self.cache.get(layer_idx)
        
    def prefetch_layer(self, layer_idx: int, device: str = '/GPU:0'):
        """Prefetch layer data to device."""
        if not self._is_gpu_available or layer_idx not in self.cache:
            return
            
        if self._prefetch_stream is not None:
            def prefetch_fn():
                for key_value in self.cache[layer_idx]:
                    with tf.device(device):
                        # Non-blocking copy
                        tf.identity(key_value[0])
                        tf.identity(key_value[1])
            
            self._prefetch_stream(prefetch_fn)
            
    def evict_layer(self, layer_idx: int):
        """Move layer cache to CPU."""
        if layer_idx not in self.cache:
            return
            
        with tf.device('/CPU:0'):
            for key_value in self.cache[layer_idx]:
                # Non-blocking move to CPU
                key_value = (
                    tf.identity(key_value[0]),
                    tf.identity(key_value[1])
                )
                
        # Clear GPU memory
        if self._is_gpu_available:
            tf.keras.backend.clear_session()


class Phi3DecoderLayer(tf.keras.layers.Layer):
    """Optimized decoder layer with memory management."""
    
    def __init__(self, config: Phi3Config, dtype=None):
        super().__init__()
        self.config = config
        self._dtype = dtype or tf.float32
        
        # Initialize attention with memory optimization
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
        
        # Layer norms with memory-efficient epsilon
        eps = getattr(config, 'layer_norm_eps', 1e-5)
        self.input_layernorm = layers.LayerNormalization(
            epsilon=eps,
            dtype=self._dtype
        )
        self.post_attention_layernorm = layers.LayerNormalization(
            epsilon=eps,
            dtype=self._dtype
        )
        
    @tf.function(jit_compile=True)
    def _gelu(self, x):
        """Memory-efficient GELU."""
        return 0.5 * x * (1.0 + tf.tanh(tf.sqrt(2.0 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))
        
    def call(self, hidden_states, attention_mask=None, past_key_value=None,
             use_cache=False, training=False):
        """Memory-optimized forward pass."""
        
        # Layer norm first (saves memory)
        normed_hidden_states = self.input_layernorm(hidden_states)
        
        # Self attention
        attn_output, present = self.self_attn(
            normed_hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            training=training
        )
        
        # First residual connection
        hidden_states = attn_output + hidden_states
        
        # Layer norm second
        normed_hidden_states = self.post_attention_layernorm(hidden_states)
        
        # MLP
        mlp_output = self.mlp(normed_hidden_states, training=training)
        
        # Second residual connection with memory cleanup
        hidden_states = mlp_output + hidden_states
        del mlp_output, attn_output, normed_hidden_states
        
        if use_cache:
            return hidden_states, present
        return hidden_states, None


class Phi3Transformer(tf.keras.Model):
    """Optimized transformer with advanced memory management."""
    
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
        
        # Enhanced cache management
        self.cache = OptimizedDynamicCache()
        
        # Memory optimization settings
        self._chunk_size = None
        self._max_memory = 0.9  # Use 90% of available memory
        
    def _get_optimal_chunk_size(self, total_size):
        """Calculate optimal chunk size based on available memory."""
        if self._chunk_size is not None:
            return self._chunk_size
            
        if not tf.config.list_physical_devices('GPU'):
            self._chunk_size = 16  # Default CPU chunk size
            return self._chunk_size
            
        # Get GPU memory info
        gpu = tf.config.list_physical_devices('GPU')[0]
        gpu_mem = tf.config.experimental.get_memory_info('GPU:0')
        available_mem = gpu_mem['free'] * self._max_memory
        
        # Calculate size per item
        size_per_item = total_size * 4  # Assuming float32
        
        # Set chunk size
        self._chunk_size = min(32, int(available_mem / size_per_item))
        return self._chunk_size
        
    def call(self, inputs_embeds, attention_mask=None, position_ids=None,
             past_key_values=None, use_cache=False, training=False):
        """Memory-optimized forward pass."""
        
        # Get batch size for chunking
        batch_size = tf.shape(inputs_embeds)[0]
        chunk_size = self._get_optimal_chunk_size(tf.size(inputs_embeds))
        
        # Process in chunks
        all_hidden_states = []
        all_presents = [] if use_cache else None
        
        for i in range(0, batch_size, chunk_size):
            end_idx = tf.minimum(i + chunk_size, batch_size)
            chunk_inputs = inputs_embeds[i:end_idx]
            
            # Get chunk attention mask
            chunk_attention_mask = None
            if attention_mask is not None:
                chunk_attention_mask = attention_mask[i:end_idx]
                
            # Get chunk position ids
            chunk_position_ids = None
            if position_ids is not None:
                chunk_position_ids = position_ids[i:end_idx]
                
            # Process chunk
            hidden_states = chunk_inputs
            presents = [] if use_cache else None
            
            for idx, decoder_layer in enumerate(self.decoder_layers):
                past_key_value = past_key_values[idx] if past_key_values is not None else None
                
                # Layer memory management
                if idx > 0:
                    self.cache.evict_layer(idx - 1)
                if idx < len(self.decoder_layers) - 1:
                    self.cache.prefetch_layer(idx + 1)
                    
                # Run layer
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=chunk_attention_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    training=training
                )
                hidden_states = layer_outputs[0]
                
                if use_cache:
                    presents.append(layer_outputs[1])
                    
            # Final layer norm
            hidden_states = self.norm(hidden_states)
            
            # Collect results
            all_hidden_states.append(hidden_states)
            if use_cache:
                all_presents.append(presents)
                
        # Combine results
        hidden_states = tf.concat(all_hidden_states, axis=0)
        
        if use_cache:
            presents = [
                tuple(tf.concat([p[i][j] for p in all_presents], axis=0) 
                      for j in range(2))
                for i in range(len(self.decoder_layers))
            ]
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=presents
            )
            
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None
        )
