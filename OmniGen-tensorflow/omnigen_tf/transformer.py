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


class Phi3Transformer(TFPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers.
    Matches PyTorch's implementation while optimizing for TensorFlow.
    """
    config_class = Phi3Config
    base_model_prefix = "transformer"
    
    def __init__(self, config, **kwargs):
        """Initialize transformer."""
        # Initialize parent class with config
        super().__init__(config, **kwargs)
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        # Enable memory optimization flags
        self.gradient_checkpointing_enabled = False
        self.use_mixed_precision = True
        self.chunk_size = None  # For chunked processing
        
        # Initialize embeddings
        self.wte = layers.Embedding(config.vocab_size, self.hidden_size, dtype='float16')
        self.drop = layers.Dropout(config.hidden_dropout)
        
        # Initialize transformer layers
        self.transformer = OmniGenTransformer(config)
        self.ln_f = layers.LayerNormalization(epsilon=config.layer_norm_epsilon, dtype='float16')
        
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
                    # Use 70% of available memory
                    memory_limit = int(tf.config.experimental.get_memory_info('GPU:0')['free'] * 0.7)
                    tf.config.set_logical_device_configuration(
                        device,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                    )
                except:
                    pass
            
            # Use mixed precision
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    def prefetch_layer(self, layer_idx: int):
        """Starts prefetching the next layer cache."""
        if not self._is_gpu_available:
            return
            
        with tf.device('/GPU:0'):
            layer = self.transformer.layers[layer_idx]
            # Non-blocking transfer
            for weight in layer.trainable_weights:
                tf.identity(weight)
    
    def evict_previous_layer(self, layer_idx: int):
        """Moves the previous layer cache to the CPU."""
        if not self._is_gpu_available or layer_idx <= 0:
            return
            
        with tf.device('/CPU:0'):
            prev_layer = self.transformer.layers[layer_idx - 1]
            for weight in prev_layer.trainable_weights:
                tf.identity(weight)
        
        # Clear GPU memory
        tf.keras.backend.clear_session()
    
    def get_offload_layer(self, layer_idx: int):
        """Manages layer offloading for memory efficiency."""
        if not self._is_gpu_available:
            return
            
        # Make sure current layer is ready
        tf.keras.backend.clear_session()
        
        # Move previous layer to CPU
        self.evict_previous_layer(layer_idx)
        
        # Prefetch next layer
        next_layer_idx = (layer_idx + 1) % len(self.transformer.layers)
        self.prefetch_layer(next_layer_idx)
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing_enabled = True
        
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        
    def set_chunk_size(self, chunk_size: Optional[int]):
        """Set chunk size for processing large inputs."""
        self.chunk_size = chunk_size
        
    def process_in_chunks(self, hidden_states):
        """Process hidden states in chunks to save memory."""
        if not self.chunk_size or hidden_states.shape[1] <= self.chunk_size:
            return hidden_states
            
        chunks = tf.split(
            hidden_states,
            num_or_size_splits=[self.chunk_size] * (hidden_states.shape[1] // self.chunk_size) + [hidden_states.shape[1] % self.chunk_size],
            axis=1
        )
        
        # Remove empty chunks
        chunks = [chunk for chunk in chunks if chunk.shape[1] > 0]
        
        # Process each chunk
        processed_chunks = []
        for chunk in chunks:
            processed_chunk = self._process_chunk(chunk)
            processed_chunks.append(processed_chunk)
            
        # Concatenate chunks
        return tf.concat(processed_chunks, axis=1)
        
    def _process_chunk(self, hidden_states):
        """Process a single chunk through the transformer layers."""
        # Apply transformer layers
        for layer in self.transformer.layers:
            if self.gradient_checkpointing_enabled:
                hidden_states = tf.stop_gradient(hidden_states)
            hidden_states = layer(hidden_states)
            
        # Apply final normalization
        hidden_states = self.ln_f(hidden_states)
        return hidden_states
        
    def call(
        self,
        input_ids: Optional[tf.Tensor] = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[tf.Tensor, tf.Tensor], ...]] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        offload_model: Optional[bool] = False,
        training: bool = False,
    ) -> Union[Tuple[Any, ...], Dict[str, Any]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
            
        if input_ids is not None:
            # Move embedding to CPU and cast to float16 for GPU operations
            with tf.device('/CPU:0'):
                inputs_embeds = tf.cast(self.wte(input_ids), tf.float16)
            batch_size, seq_length = tf.shape(input_ids)[0], tf.shape(input_ids)[1]
        elif inputs_embeds is not None:
            # Cast inputs to float16 for GPU operations
            inputs_embeds = tf.cast(inputs_embeds, tf.float16)
            batch_size, seq_length = tf.shape(inputs_embeds)[0], tf.shape(inputs_embeds)[1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # Generate position IDs on CPU
        if position_ids is None:
            with tf.device('/CPU:0'):
                position_ids = tf.range(seq_length, dtype=tf.int32)[tf.newaxis, :]

        # Handle cache conversion
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        # Process attention mask with float16
        if attention_mask is not None:
            with tf.device('/CPU:0'):
                attention_mask = tf.cast(attention_mask, tf.float16)
                attention_mask = (1.0 - attention_mask) * tf.float16.min

        # Initialize outputs
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # Process on CPU first
        with tf.device('/CPU:0'):
            hidden_states = tf.cast(self.drop(inputs_embeds, training=training), tf.float16)

        # Process through layers
        for idx, layer in enumerate(self.transformer.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Memory management
            if offload_model and not training:
                self.get_offload_layer(idx)

            # Get layer cache
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            # Process layer with float16
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                training=training,
            )

            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
                self_attn = layer_outputs[1] if len(layer_outputs) > 1 else None
                present_key_value = layer_outputs[2] if len(layer_outputs) > 2 else None
            else:
                hidden_states = layer_outputs
                self_attn = None
                present_key_value = None
                
            if use_cache:
                next_decoder_cache = next_decoder_cache + (present_key_value,) if next_decoder_cache is not None else (present_key_value,)

            if output_attentions:
                all_self_attns = all_self_attns + (self_attn,)

            # Clear intermediate tensors
            if tf.config.list_physical_devices('GPU'):
                tf.keras.backend.clear_session()

        # Final layer norm with float16
        hidden_states = tf.cast(self.ln_f(hidden_states), tf.float16)

        # Add final hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Handle cache
        if use_cache and return_legacy_cache and next_decoder_cache is not None:
            next_decoder_cache = next_decoder_cache.to_legacy_cache()

        # Clear any remaining GPU memory
        if tf.config.list_physical_devices('GPU'):
            tf.keras.backend.clear_session()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_decoder_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
        }
        
    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'config': self.config.to_dict(),
            'gradient_checkpointing_enabled': self.gradient_checkpointing_enabled,
            'use_mixed_precision': self.use_mixed_precision,
            'chunk_size': self.chunk_size,
        })
        return config
        
    @classmethod
    def from_config(cls, config):
        """Create model from configuration."""
        # Extract transformer config
        transformer_config = config.pop('config', None)
        if transformer_config is not None:
            transformer_config = Phi3Config(**transformer_config)
            
        # Create model
        model = cls(config=transformer_config)
        
        # Set optimization flags
        model.gradient_checkpointing_enabled = config.get('gradient_checkpointing_enabled', False)
        model.use_mixed_precision = config.get('use_mixed_precision', False)
        model.chunk_size = config.get('chunk_size', None)
        
        return model


class OmniGenTransformer(layers.Layer):
    """Transformer model for OmniGen."""
    
    def __init__(self, config: Phi3Config, **kwargs):
        """Initialize transformer."""
        super().__init__(**kwargs)
        
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.chunk_size = 32  # Default chunk size for processing
        
        # Initialize components with mixed precision
        self.h = [OmniGenLayer(config, name=f"h.{i}") for i in range(self.num_hidden_layers)]
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
        
    def _merge_heads(self, x):
        """Merge heads to [batch_size, seq_length, hidden_size]."""
        # x is [batch_size, num_heads, seq_length, head_dim]
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[2]

        # Transpose to [batch_size, seq_length, num_heads, head_dim]
        x = tf.transpose(x, [0, 2, 1, 3])

        # Reshape to [batch_size, seq_length, hidden_size]
        x = tf.reshape(x, [batch_size, seq_length, self.hidden_size])
        return x
        
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
        qkv = tf.reshape(qkv, [batch_size, seq_length, 3, self.num_attention_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])  # [3, batch_size, num_heads, seq_length, head_dim]
        query, key, value = tf.unstack(qkv, axis=0)  # Each has shape [batch_size, num_heads, seq_length, head_dim]

        # Compute attention scores
        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_scores = attention_scores / tf.math.sqrt(float(self.head_dim))

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, attention_scores.dtype)
            attention_scores = attention_scores + attention_mask

        # Compute attention probabilities
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        attention_probs = self.attention_dropout(attention_probs, training=training)

        # Compute context
        context = tf.matmul(attention_probs, value)  # [batch_size, num_heads, seq_length, head_dim]
        context = OmniGenLayer._merge_heads(context)  # [batch_size, seq_length, hidden_size]
        # context = tf.transpose(context, [0, 2, 1, 3])  # [batch_size, seq_length, num_heads, head_dim]
        # context = tf.reshape(context, [batch_size, seq_length, self.hidden_size])

        # Project output
        output = self.o_proj(context)
        output = self.resid_dropout(output, training=training)

        if output_attentions:
            return output, attention_probs
        return output


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
