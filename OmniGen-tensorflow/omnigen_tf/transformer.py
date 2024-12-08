"""Phi3 Transformer implementation in TensorFlow."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, initializers
import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple

class MultiHeadAttention(layers.Layer):
    """Multi-head attention layer."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.num_attention_heads = config.get('num_attention_heads', 32)
        self.hidden_size = config.get('hidden_size', 2048)
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.scale = 1.0 / tf.math.sqrt(tf.cast(self.attention_head_size, tf.float32))
        
        # Initialize with truncated normal
        kernel_init = initializers.TruncatedNormal(stddev=0.02)
        
        self.query = layers.Dense(
            self.hidden_size,
            use_bias=True,
            kernel_initializer=kernel_init,
            bias_initializer='zeros',
            name='q_proj'
        )
        self.key = layers.Dense(
            self.hidden_size,
            use_bias=True,
            kernel_initializer=kernel_init,
            bias_initializer='zeros',
            name='k_proj'
        )
        self.value = layers.Dense(
            self.hidden_size,
            use_bias=True,
            kernel_initializer=kernel_init,
            bias_initializer='zeros',
            name='v_proj'
        )
        self.out = layers.Dense(
            self.hidden_size,
            use_bias=True,
            kernel_initializer=kernel_init,
            bias_initializer='zeros',
            name='out_proj'
        )
        
    def transpose_for_scores(self, x: tf.Tensor) -> tf.Tensor:
        """Transpose and reshape tensor for attention computation."""
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]
        
        # [batch_size, seq_length, hidden_size] -> [batch_size, seq_length, num_heads, head_size]
        x = tf.reshape(x, (batch_size, seq_length, self.num_attention_heads, self.attention_head_size))
        
        # [batch_size, seq_length, num_heads, head_size] -> [batch_size, num_heads, seq_length, head_size]
        return tf.transpose(x, [0, 2, 1, 3])
        
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        past_key_value: Optional[Tuple[tf.Tensor, tf.Tensor]] = None,
        use_cache: bool = False
    ) -> Union[Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]], tf.Tensor]:
        """Forward pass."""
        batch_size = tf.shape(hidden_states)[0]
        
        # Project queries, keys, and values
        query_states = self.transpose_for_scores(self.query(hidden_states))
        key_states = self.transpose_for_scores(self.key(hidden_states))
        value_states = self.transpose_for_scores(self.value(hidden_states))
        
        # Use cached key/value states if provided
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = tf.concat([past_key, key_states], axis=2)
            value_states = tf.concat([past_value, value_states], axis=2)
            
        # Cache current key/value states if requested
        if use_cache:
            current_key_value = (key_states, value_states)
            
        # Compute attention scores
        attention_scores = tf.matmul(query_states, key_states, transpose_b=True)
        attention_scores = attention_scores * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        # Normalize attention scores
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        
        # Compute context layer
        context_layer = tf.matmul(attention_probs, value_states)
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
        context_layer = tf.reshape(context_layer, (batch_size, -1, self.hidden_size))
        
        # Project output
        attention_output = self.out(context_layer)
        
        if use_cache:
            return attention_output, current_key_value
        return attention_output

class Phi3Block(layers.Layer):
    """Transformer block for Phi3."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        
        # Initialize MLP with proper initializers
        kernel_init = initializers.TruncatedNormal(stddev=0.02)
        self.mlp = tf.keras.Sequential([
            layers.Dense(
                config.get('intermediate_size', 8192),
                activation='gelu',
                kernel_initializer=kernel_init,
                bias_initializer='zeros',
                name='fc1'
            ),
            layers.Dense(
                config.get('hidden_size', 2048),
                kernel_initializer=kernel_init,
                bias_initializer='zeros',
                name='fc2'
            )
        ])
        
        # Layer norms
        self.input_layernorm = layers.LayerNormalization(epsilon=1e-5, name='input_layernorm')
        self.post_attention_layernorm = layers.LayerNormalization(epsilon=1e-5, name='post_attention_layernorm')
        
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        past_key_value: Optional[Tuple[tf.Tensor]] = None,
        use_cache: bool = False
    ) -> Union[Tuple[tf.Tensor, Tuple[tf.Tensor]], tf.Tensor]:
        """Forward pass."""
        # Pre-attention norm
        norm_states = self.input_layernorm(hidden_states)
        
        # Self-attention
        attention_output = self.attention(
            norm_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        
        # Handle cached key/values
        if use_cache:
            attention_output, present_key_value = attention_output
            
        # First residual connection
        hidden_states = attention_output + hidden_states
        
        # MLP
        mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        
        # Second residual connection
        hidden_states = mlp_output + hidden_states
        
        if use_cache:
            return hidden_states, present_key_value
        return hidden_states

class Phi3TransformerTF(layers.Layer):
    """Phi3 Transformer model."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.hidden_size = config.get('hidden_size', 2048)
        self.num_hidden_layers = config.get('num_hidden_layers', 32)
        
        # Transformer blocks
        self.blocks = [Phi3Block(config) for _ in range(self.num_hidden_layers)]
        
        # Final layer norm
        self.final_layernorm = layers.LayerNormalization(epsilon=1e-5, name='final_layernorm')
        
        # Cache for key/value states
        self.use_cache = False
        
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        past_key_values: Optional[List[Tuple[tf.Tensor]]] = None,
        use_cache: Optional[bool] = None
    ) -> Union[Tuple[tf.Tensor, List[Tuple[tf.Tensor]]], tf.Tensor]:
        """Forward pass."""
        use_cache = use_cache if use_cache is not None else self.use_cache
        
        # Initialize present key/values list if caching
        presents = [] if use_cache else None
        
        # Process through transformer blocks
        for i, block in enumerate(self.blocks):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            if use_cache:
                hidden_states, present_key_value = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache
                )
                presents.append(present_key_value)
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache
                )
                
        # Final layer norm
        hidden_states = self.final_layernorm(hidden_states)
        
        if use_cache:
            return hidden_states, presents
        return hidden_states
