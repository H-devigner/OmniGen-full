"""OmniGen Transformer implementation."""

import math
from typing import Optional, Tuple, Union, Dict, List, Any
import warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from transformers import PretrainedConfig
from transformers.modeling_tf_utils import TFPreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Phi3Config(PretrainedConfig):
    """Configuration class for Phi-2 model."""
    model_type = "phi"
    
    def __init__(
        self,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=24,
        num_attention_heads=32,
        max_position_embeddings=2048,
        layer_norm_eps=1e-5,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        use_cache=True,
        vocab_size=51200,
        tie_word_embeddings=False,
        output_attentions=False,
        output_hidden_states=False,
        use_return_dict=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        sep_token_id=None,
        cls_token_id=None,
        mask_token_id=None,
        unk_token_id=None,
        **kwargs
    ):
        """Initialize config."""
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_epsilon = layer_norm_eps
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        
        # Compute head dimensions
        self.head_dim = self.hidden_size // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads (got hidden_size={self.hidden_size} "
                f"and num_attention_heads={self.num_attention_heads})"
            )
            
        # Initialize parent class
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            sep_token_id=sep_token_id,
            cls_token_id=cls_token_id,
            mask_token_id=mask_token_id,
            unk_token_id=unk_token_id,
            **kwargs
        )
        
    def to_dict(self):
        """Convert config to dictionary."""
        output = {
            'hidden_size': self.hidden_size,
            'intermediate_size': self.intermediate_size,
            'num_hidden_layers': self.num_hidden_layers,
            'num_attention_heads': self.num_attention_heads,
            'max_position_embeddings': self.max_position_embeddings,
            'layer_norm_eps': self.layer_norm_epsilon,
            'hidden_dropout': self.hidden_dropout,
            'attention_dropout': self.attention_dropout,
            'initializer_range': self.initializer_range,
            'use_cache': self.use_cache,
            'vocab_size': self.vocab_size,
            'tie_word_embeddings': self.tie_word_embeddings,
            'output_attentions': self.output_attentions,
            'output_hidden_states': self.output_hidden_states,
            'pad_token_id': self.pad_token_id,
            'bos_token_id': self.bos_token_id,
            'eos_token_id': self.eos_token_id,
            'sep_token_id': self.sep_token_id,
            'cls_token_id': self.cls_token_id,
            'mask_token_id': self.mask_token_id,
            'unk_token_id': self.unk_token_id,
        }
        return output


class Phi3Transformer(layers.Layer):
    """Phi-3 transformer model."""
    
    def __init__(self, config, **kwargs):
        """Initialize transformer."""
        super().__init__(**kwargs)
        self.config = config
        
        # Create embedding layers
        self.wte = layers.Embedding(
            config.vocab_size,
            config.hidden_size,
            name="wte"
        )
        
        # Create transformer blocks
        self.blocks = []
        for i in range(config.num_hidden_layers):
            self.blocks.append(
                TransformerLayer(
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    max_position_embeddings=config.max_position_embeddings,
                    layer_norm_epsilon=config.layer_norm_epsilon,
                    hidden_dropout_prob=config.hidden_dropout,
                    attention_probs_dropout_prob=config.attention_dropout,
                    name=f"block_{i}"
                )
            )
            
        # Create final layer norm
        self.ln_f = layers.LayerNormalization(
            epsilon=config.layer_norm_epsilon,
            name="ln_f"
        )
        
    def build(self, input_shape):
        """Build the model."""
        # Add blocks to the layer
        for block in self.blocks:
            self._tracker.track(block)
        super().build(input_shape)
        
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        offload_model=False,
        training=False,
    ):
        """Forward pass."""
        # Get input embeddings
        if input_ids is not None:
            input_shape = tf.shape(input_ids)
            inputs_embeds = self.wte(input_ids)
        elif inputs_embeds is not None:
            input_shape = tf.shape(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
            
        # Initialize states
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        
        # Initialize hidden states and attention outputs
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # Process through blocks
        for idx, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            # Get past key value
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            # Process through block
            layer_outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                training=training
            )
            
            # Update hidden states
            hidden_states = layer_outputs[0]
            
            # Update attention outputs
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Add final hidden states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        # Return outputs
        return TransformerOutputs(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions
        )


class TransformerLayer(layers.Layer):
    """Transformer layer."""
    
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        max_position_embeddings,
        layer_norm_epsilon,
        hidden_dropout_prob,
        attention_probs_dropout_prob,
        **kwargs
    ):
        """Initialize layer."""
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_epsilon = layer_norm_epsilon
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        
        # Initialize components
        self.attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout=attention_probs_dropout_prob,
            name="attention"
        )
        self.attention_norm = layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            name="attention_norm"
        )
        self.attention_dropout = layers.Dropout(hidden_dropout_prob)
        
        self.mlp_in = layers.Dense(
            4 * hidden_size,
            activation="gelu",
            name="mlp_in"
        )
        self.mlp_out = layers.Dense(
            hidden_size,
            name="mlp_out"
        )
        self.mlp_norm = layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            name="mlp_norm"
        )
        self.mlp_dropout = layers.Dropout(hidden_dropout_prob)
        
    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        training=False,
    ):
        """Forward pass."""
        # Self attention
        attn_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            training=training
        )
        
        if isinstance(attn_outputs, tuple):
            attn_output = attn_outputs[0]
            attention_weights = attn_outputs[1] if len(attn_outputs) > 1 else None
            present_key_value = attn_outputs[2] if len(attn_outputs) > 2 else None
        else:
            attn_output = attn_outputs
            attention_weights = None
            present_key_value = None
            
        # Apply attention dropout and residual connection
        attn_output = self.attention_dropout(attn_output, training=training)
        hidden_states = self.attention_norm(attn_output + hidden_states)
        
        # MLP
        mlp_output = self.mlp_in(hidden_states)
        mlp_output = self.mlp_out(mlp_output)
        mlp_output = self.mlp_dropout(mlp_output, training=training)
        hidden_states = self.mlp_norm(mlp_output + hidden_states)
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attention_weights,)
        if use_cache:
            outputs += (present_key_value,)
            
        return outputs[0] if len(outputs) == 1 else outputs


class MultiHeadAttention(layers.Layer):
    """Multi-head attention layer."""
    
    def __init__(self, hidden_size, num_attention_heads, dropout, **kwargs):
        """Initialize layer."""
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.head_size = hidden_size // num_attention_heads
        
        # Initialize query, key, value projections
        self.q_proj = layers.Dense(hidden_size, use_bias=False, name="q_proj")
        self.k_proj = layers.Dense(hidden_size, use_bias=False, name="k_proj")
        self.v_proj = layers.Dense(hidden_size, use_bias=False, name="v_proj")
        
        # Initialize output projection
        self.o_proj = layers.Dense(hidden_size, use_bias=False, name="o_proj")
        
        # Initialize dropout
        self.dropout = layers.Dropout(dropout)
        
    def _split_heads(self, x):
        """Split heads."""
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]
        
        # Reshape to [batch_size, seq_length, num_heads, head_size]
        x = tf.reshape(x, [batch_size, seq_length, self.num_attention_heads, self.head_size])
        
        # Transpose to [batch_size, num_heads, seq_length, head_size]
        return tf.transpose(x, [0, 2, 1, 3])
        
    def _merge_heads(self, x):
        """Merge heads."""
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[2]
        
        # Transpose back to [batch_size, seq_length, num_heads, head_size]
        x = tf.transpose(x, [0, 2, 1, 3])
        
        # Reshape to [batch_size, seq_length, hidden_size]
        return tf.reshape(x, [batch_size, seq_length, self.hidden_size])
        
    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        training=False,
    ):
        """Forward pass."""
        batch_size = tf.shape(hidden_states)[0]
        seq_length = tf.shape(hidden_states)[1]
        
        # Project query, key, value
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Split heads
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        
        # Use past key value if provided
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = tf.concat([past_key, key], axis=2)
            value = tf.concat([past_value, value], axis=2)
            
        # Save current key, value if using cache
        if use_cache:
            present_key_value = (key, value)
        else:
            present_key_value = None
            
        # Compute attention scores
        scale = tf.cast(tf.math.sqrt(tf.cast(self.head_size, tf.float32)), hidden_states.dtype)
        attention_scores = tf.matmul(query, tf.transpose(key, [0, 1, 3, 2])) / scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        # Apply softmax
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        
        # Apply dropout
        attention_probs = self.dropout(attention_probs, training=training)
        
        # Apply attention to values
        context = tf.matmul(attention_probs, value)
        
        # Merge heads
        context = self._merge_heads(context)
        
        # Apply output projection
        output = self.o_proj(context)
        
        outputs = (output,)
        if output_attentions:
            outputs += (attention_probs,)
        if use_cache:
            outputs += (present_key_value,)
            
        return outputs[0] if len(outputs) == 1 else outputs


class OmniGenTransformer(layers.Layer):
    """Transformer model for OmniGen."""
    
    def __init__(self, config: Phi3Config, **kwargs):
        """Initialize transformer."""
        super().__init__(**kwargs)
        
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.chunk_size = 32  # Default chunk size for processing
        
        # Initialize components with mixed precision
        self.h = [TransformerLayer(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            layer_norm_epsilon=config.layer_norm_epsilon,
            hidden_dropout_prob=config.hidden_dropout,
            attention_probs_dropout_prob=config.attention_dropout,
            name=f"h.{i}"
        ) for i in range(self.num_hidden_layers)]
        self.norm = layers.LayerNormalization(epsilon=1e-5, dtype=tf.float16, name="norm")
        
    def _process_chunk(self, chunk, layer_module, attention_mask, position_ids, past_key_value, output_attentions, use_cache, training):
        """Process a single chunk through a transformer layer."""
        # Process chunk with layer
        layer_outputs = layer_module(
            chunk,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            training=training
        )
        return layer_outputs
        
    def _process_in_chunks(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, training=False):
        """Process hidden states in chunks to save memory."""
        # Get dimensions
        batch_size = tf.shape(hidden_states)[0]
        seq_length = tf.shape(hidden_states)[1]
        
        # Calculate number of chunks
        num_chunks = tf.cast(tf.math.ceil(seq_length / self.chunk_size), tf.int32)
        
        # Process each chunk
        chunk_outputs = []
        for i in range(num_chunks):
            # Get chunk indices
            start_idx = i * self.chunk_size
            end_idx = tf.minimum(start_idx + self.chunk_size, seq_length)
            
            # Extract chunk
            chunk = hidden_states[:, start_idx:end_idx, :]
            
            # Process chunk through layers
            for layer_module in self.h:
                chunk = self._process_chunk(
                    chunk,
                    layer_module,
                    attention_mask[:, start_idx:end_idx] if attention_mask is not None else None,
                    position_ids[:, start_idx:end_idx] if position_ids is not None else None,
                    past_key_value,
                    output_attentions,
                    use_cache,
                    training
                )[0]  # Get hidden states from layer outputs
            
            chunk_outputs.append(chunk)
        
        # Concatenate chunks
        hidden_states = tf.concat(chunk_outputs, axis=1)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        return hidden_states
        
    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        training=False,
    ):
        """Forward pass with memory optimization."""
        if hidden_states is None:
            return hidden_states
            
        # Cast inputs to float16
        hidden_states = tf.cast(hidden_states, tf.float16)
        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, tf.int32)
            
        # Process in chunks if sequence length is large
        if tf.shape(hidden_states)[1] > self.chunk_size:
            return self._process_in_chunks(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                training
            )
        
        # Regular processing for small sequences
        all_hidden_states = () if output_attentions else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        # Process through layers
        for i, layer_module in enumerate(self.h):
            past_key_value_layer = past_key_value[i] if past_key_value is not None else None
            
            # Process layer
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value_layer,
                output_attentions=output_attentions,
                use_cache=use_cache,
                training=training
            )
                
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
                
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Prepare outputs
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (all_self_attns,)
        if use_cache:
            outputs += (next_decoder_cache,)
            
        return outputs[0] if len(outputs) == 1 else outputs


class OmniGenLayer(layers.Layer):
    """Transformer layer for OmniGen."""
    
    def __init__(self, config: Phi3Config, **kwargs):
        """Initialize layer."""
        super().__init__(**kwargs)
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        # Initialize components
        self.input_layernorm = layers.LayerNormalization(epsilon=1e-5, name="input_layernorm")
        self.self_attn = OmniGenAttention(config, name="self_attn")
        self.post_attention_layernorm = layers.LayerNormalization(epsilon=1e-5, name="post_attention_layernorm")
        self.mlp = OmniGenMLP(config.hidden_size, config.intermediate_size, name="mlp")
        
    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        training=False,
    ):
        """Forward pass of transformer layer with memory optimization."""
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            training=training,
        )
        
        if isinstance(attn_outputs, tuple):
            attn_output = attn_outputs[0]
            attn_weights = attn_outputs[1] if len(attn_outputs) > 1 else None
            present_key_value = attn_outputs[2] if len(attn_outputs) > 2 else None
        else:
            attn_output = attn_outputs
            attn_weights = None
            present_key_value = None
            
        hidden_states = attn_output + residual
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, training=training)
        hidden_states = hidden_states + residual
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
            
        return outputs


class OmniGenAttention(layers.Layer):
    """Multi-head attention layer for OmniGen."""
    
    def __init__(self, config: Phi3Config, **kwargs):
        """Initialize attention layer."""
        super().__init__(**kwargs)
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        
        if self.head_dim * self.num_attention_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads (got hidden_size={self.hidden_size} "
                f"and num_attention_heads={self.num_attention_heads})."
            )
            
        # Initialize components
        self.qkv_proj = layers.Dense(3 * self.hidden_size, use_bias=False, name="qkv_proj")
        self.o_proj = layers.Dense(self.hidden_size, use_bias=False, name="o_proj")
        
        self.attention_dropout = layers.Dropout(config.attention_dropout)
        self.resid_dropout = layers.Dropout(config.hidden_dropout)
        
    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        """Reshape tensor for attention computation."""
        return tf.transpose(
            tf.reshape(tensor, (bsz, seq_len, self.num_attention_heads, self.head_dim)),
            (0, 2, 1, 3)
        )
        
    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        training=False,
    ):
        """Forward pass."""
        batch_size, seq_length = tf.shape(hidden_states)[0], tf.shape(hidden_states)[1]
        
        # Project input to query, key, value
        qkv = self.qkv_proj(hidden_states)
        qkv = tf.reshape(qkv, (batch_size, seq_length, 3, self.num_attention_heads, self.head_dim))
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = tf.unstack(qkv, axis=0)
        
        # Reuse past key and value if provided
        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = tf.concat([past_key, k], axis=2)
            v = tf.concat([past_value, v], axis=2)
            
        # Save current key and value if needed
        present = (k, v) if use_cache else None
            
        # Compute attention scores
        scale = tf.cast(1.0 / tf.math.sqrt(tf.cast(self.head_dim, tf.float32)), hidden_states.dtype)
        attn_weights = tf.matmul(q, k, transpose_b=True) * scale  # [batch_size, num_heads, seq_length, seq_length]
        
        # Add attention mask if provided
        if attention_mask is not None:
            # Expand attention_mask: [batch_size, seq_length] -> [batch_size, 1, 1, seq_length]
            attention_mask = tf.expand_dims(tf.expand_dims(attention_mask, axis=1), axis=1)
            attention_mask = tf.cast(attention_mask, attn_weights.dtype)
            
            # Convert mask of 0s and 1s to mask of -inf and 0s
            attention_mask = (1.0 - attention_mask) * tf.cast(-10000.0, attention_mask.dtype)
            attn_weights = attn_weights + attention_mask
            
        # Normalize attention weights
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attn_weights = self.attention_dropout(attn_weights, training=training)
        
        # Compute attention output
        attn_output = tf.matmul(attn_weights, v)
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, (batch_size, seq_length, self.hidden_size))
        
        # Project output
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output, training=training)
        
        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present,)
            
        return outputs


class OmniGenMLP(layers.Layer):
    """MLP layer with memory optimization."""

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Initialize components with mixed precision
        self.gate_up_proj = layers.Dense(
            2 * intermediate_size,
            use_bias=True,
            dtype=tf.float16,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)
        )
        self.down_proj = layers.Dense(
            hidden_size,
            use_bias=True,
            dtype=tf.float16,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)
        )
        
    @tf.function(jit_compile=True)
    def _process_chunk(self, chunk, training=False):
        """Process a single chunk of data with XLA optimization."""
        # Project to higher dimension
        gate_up = self.gate_up_proj(chunk)
        
        # Split activation
        gate, up = tf.split(gate_up, 2, axis=-1)
        
        # Apply activation and multiply
        gate = tf.keras.activations.swish(gate)
        hidden_states = gate * up
        
        # Project back to original dimension
        return self.down_proj(hidden_states)
        
    def call(self, x, training=False):
        """Forward pass with memory optimization."""
        # Cast input to float16
        x = tf.cast(x, tf.float16)
        
        # Project to higher dimension
        gate_up = self.gate_up_proj(x)
        
        # Split activation
        gate, up = tf.split(gate_up, 2, axis=-1)
        
        # Apply activation and multiply
        gate = tf.keras.activations.swish(gate)
        hidden_states = gate * up
        
        # Project back to original dimension
        return self.down_proj(hidden_states)


class TransformerOutputs:
    def __init__(self, last_hidden_state, past_key_values=None, hidden_states=None, attentions=None):
        self.last_hidden_state = last_hidden_state
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions
