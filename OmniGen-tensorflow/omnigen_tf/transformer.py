import math
import warnings
from typing import List, Optional, Tuple, Union

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from transformers.modeling_tf_utils import TFPreTrainedModel, unpack_inputs
from transformers import PretrainedConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging

logger = logging.get_logger(__name__)

class Phi3Config(PretrainedConfig):
    """Configuration class for Phi3 model."""
    model_type = "phi3"
    
    def __init__(
        self,
        vocab_size=51200,
        hidden_size=2560,
        num_hidden_layers=32,
        num_attention_heads=32,
        intermediate_size=10240,
        hidden_act="gelu",
        hidden_dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=2048,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        **kwargs,
    ):
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache

class Phi3Transformer(TFPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers.
    Matches PyTorch's implementation while optimizing for TensorFlow.
    
    Args:
        config: Phi3Config
    """
    config_class = Phi3Config
    base_model_prefix = "transformer"
    
    def __init__(self, config: Phi3Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        
        # Initialize layers
        self.wte = layers.Embedding(config.vocab_size, self.embed_dim)
        self.drop = layers.Dropout(config.hidden_dropout)
        self.h = [PhiDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.ln_f = layers.LayerNormalization(epsilon=config.layer_norm_eps)
        
        # Setup memory optimization
        self._setup_memory_optimization()
    
    def _setup_memory_optimization(self):
        """Setup memory optimization configurations."""
        self._is_gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        
        if self._is_gpu_available:
            # Enable memory growth
            for device in tf.config.list_physical_devices('GPU'):
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                    tf.config.set_logical_device_configuration(
                        device,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=int(tf.config.experimental.get_memory_info('GPU:0')['free'] * 0.9))]
                    )
                except:
                    pass
            
            # Use mixed precision
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    def get_input_embeddings(self):
        return self.wte
    
    def set_input_embeddings(self, value):
        self.wte = value
    
    def _get_chunk_size(self, total_size: int) -> int:
        """Calculate optimal chunk size based on available memory."""
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
    
    @unpack_inputs
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
        training: bool = False,
    ) -> Union[Tuple, dict]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = tf.shape(input_ids)[0], tf.shape(input_ids)[1]
            inputs_embeds = self.wte(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_length = tf.shape(inputs_embeds)[0], tf.shape(inputs_embeds)[1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if position_ids is None:
            position_ids = tf.range(seq_length, dtype=tf.int32)[tf.newaxis, :]

        # Handle cache conversion
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

        # Process attention mask
        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, dtype=inputs_embeds.dtype)
            attention_mask = (1.0 - attention_mask) * tf.float32.min

        # Initialize outputs
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # Process in chunks for memory efficiency
        hidden_states = self.drop(inputs_embeds, training=training)
        chunk_size = self._get_chunk_size(tf.size(hidden_states))
        hidden_states_chunks = []

        for i in range(0, batch_size, chunk_size):
            end_idx = tf.minimum(i + chunk_size, batch_size)
            chunk_states = hidden_states[i:end_idx]
            
            # Process through layers
            for idx, layer in enumerate(self.h):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (chunk_states,)

                past_key_value = past_key_values[idx] if past_key_values is not None else None

                layer_outputs = layer(
                    chunk_states,
                    attention_mask=attention_mask[i:end_idx] if attention_mask is not None else None,
                    position_ids=position_ids[i:end_idx],
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    training=training,
                )

                chunk_states = layer_outputs[0]
                
                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns = all_self_attns + (layer_outputs[1],)

            # Final layer norm
            chunk_states = self.ln_f(chunk_states)
            hidden_states_chunks.append(chunk_states)

        # Combine chunks
        hidden_states = tf.concat(hidden_states_chunks, axis=0)

        # Add final hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Handle cache
        if use_cache:
            if return_legacy_cache:
                next_decoder_cache = next_decoder_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_decoder_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
        }

class PhiDecoderLayer(layers.Layer):
    """Phi model decoder layer."""
    
    def __init__(self, config: Phi3Config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        self.self_attn = PhiAttention(config)
        self.input_layernorm = layers.LayerNormalization(epsilon=config.layer_norm_eps)
        self.mlp = PhiMLP(config)
        self.post_attention_layernorm = layers.LayerNormalization(epsilon=config.layer_norm_eps)
        
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_value: Optional[Tuple[tf.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], Optional[Tuple[tf.Tensor, tf.Tensor]]]:
        """
        Args:
            hidden_states: (batch, seq_len, embed_dim)
            attention_mask: (batch, 1, seq_len, seq_len)
            position_ids: (batch, seq_len)
            past_key_value: tuple(2 tensors) of shape (batch, num_heads, seq_len - 1, head_dim)
            output_attentions: bool
            use_cache: bool
            training: bool
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            training=training,
        )
        
        hidden_states = attn_outputs[0]
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (attn_outputs[1],)
        if use_cache:
            outputs += (attn_outputs[-1],)
            
        return outputs

class PhiAttention(layers.Layer):
    """Multi-head attention implementation for Phi model."""
    
    def __init__(self, config: Phi3Config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        self.k_proj = layers.Dense(self.embed_dim, use_bias=False)
        self.v_proj = layers.Dense(self.embed_dim, use_bias=False)
        self.q_proj = layers.Dense(self.embed_dim, use_bias=False)
        self.out_proj = layers.Dense(self.embed_dim, use_bias=True)
        self.dropout = layers.Dropout(config.attention_dropout)
        
    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        return tf.transpose(
            tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), 
            (0, 2, 1, 3)
        )
        
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_value: Optional[Tuple[tf.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor]]:
        """Input shape: Batch x Time x Channel"""
        
        bsz, q_len, _ = tf.shape(hidden_states)[0], tf.shape(hidden_states)[1], tf.shape(hidden_states)[2]
        
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = self._shape(query_states, q_len, bsz)
        key_states = self._shape(key_states, q_len, bsz)
        value_states = self._shape(value_states, q_len, bsz)
        
        if past_key_value is not None:
            key_states = tf.concat([past_key_value[0], key_states], axis=2)
            value_states = tf.concat([past_key_value[1], value_states], axis=2)
            
        if use_cache:
            present = (key_states, value_states)
            
        # Compute attention
        attn_weights = tf.matmul(query_states, tf.transpose(key_states, (0, 1, 3, 2)))
        
        if attention_mask is not None:
            attn_weights = tf.where(attention_mask, attn_weights, tf.float32.min)
            
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attn_weights = self.dropout(attn_weights, training=training)
        
        attn_output = tf.matmul(attn_weights, value_states)
        attn_output = tf.transpose(attn_output, (0, 2, 1, 3))
        attn_output = tf.reshape(attn_output, (bsz, q_len, self.embed_dim))
        
        attn_output = self.out_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
            
        outputs = (attn_output, attn_weights)
        if use_cache:
            outputs += (present,)
            
        return outputs

class PhiMLP(layers.Layer):
    """MLP implementation for Phi model."""
    
    def __init__(self, config: Phi3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.fc1 = layers.Dense(self.intermediate_size, use_bias=False)
        self.fc2 = layers.Dense(self.hidden_size, use_bias=True)
        self.dropout = layers.Dropout(config.hidden_dropout)
        
    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = tf.nn.gelu(hidden_states, approximate=False)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        return hidden_states
