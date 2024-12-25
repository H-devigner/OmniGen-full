"""OmniGen Transformer implementation."""

import math
from typing import Optional, Tuple, Union, Dict, List, Any
import warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from transformers import PretrainedConfig
from transformers.modeling_tf_utils import TFPreTrainedModel, unpack_inputs
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Phi3Config(PretrainedConfig):
    """Configuration class for Phi3 model."""
    model_type = "phi3"
    attribute_map = {
        "num_attention_heads": "num_attention_heads",
        "hidden_size": "hidden_size",
        "num_hidden_layers": "num_hidden_layers"
    }
    
    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        max_position_embeddings: int = 2048,
        layer_norm_eps: float = 1e-5,
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        vocab_size: int = 32000,
        tie_word_embeddings: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_return_dict: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        sep_token_id: Optional[int] = None,
        cls_token_id: Optional[int] = None,
        mask_token_id: Optional[int] = None,
        unk_token_id: int = 3,
        **kwargs
    ):
        """Initialize config."""
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        
        # Compute derived attributes
        self.head_dim = self.hidden_size // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads (got hidden_size={self.hidden_size} "
                f"and num_attention_heads={self.num_attention_heads})"
            )
            
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            sep_token_id=sep_token_id,
            cls_token_id=cls_token_id,
            mask_token_id=mask_token_id,
            unk_token_id=unk_token_id,
            return_dict=use_return_dict,
            **kwargs
        )


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
        self.gradient_checkpointing = False
        self.use_mixed_precision = True
        self.chunk_size = None  # For chunked processing
        
        # Initialize embeddings
        self.wte = layers.Embedding(config.vocab_size, self.hidden_size, dtype='float16')
        self.drop = layers.Dropout(config.hidden_dropout)
        
        # Initialize transformer layers
        self.transformer = OmniGenTransformer(config)
        self.ln_f = layers.LayerNormalization(epsilon=config.layer_norm_eps, dtype='float16')
        
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
    
    @unpack_inputs
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

            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache = layer_outputs[-1]

            if output_attentions:
                all_self_attns = all_self_attns + (layer_outputs[1],)

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


class OmniGenTransformer(layers.Layer):
    """Transformer model for OmniGen."""
    
    def __init__(self, config, **kwargs):
        """Initialize transformer."""
        super().__init__(**kwargs)
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        # Enable memory optimization flags
        self.gradient_checkpointing = False
        self.use_mixed_precision = True
        self.chunk_size = None  # For chunked processing
        
        # Initialize layers
        self.layers = [OmniGenLayer(config, name=f"layer_{i}") for i in range(self.num_hidden_layers)]
        self.norm = layers.LayerNormalization(epsilon=1e-5, name="norm")
        
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True
        
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
        
    def set_chunk_size(self, chunk_size: Optional[int]):
        """Set chunk size for processing large inputs."""
        self.chunk_size = chunk_size
        
    def process_in_chunks(self, fn, inputs, chunk_size):
        """Process input in chunks to save memory."""
        if chunk_size is None or inputs.shape[1] <= chunk_size:
            return fn(inputs)
            
        chunks = tf.split(inputs, 
                         num_or_size_splits=math.ceil(inputs.shape[1]/chunk_size),
                         axis=1)
        output_chunks = [fn(chunk) for chunk in chunks]
        return tf.concat(output_chunks, axis=1)
        
    @unpack_inputs
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
        """Forward pass of transformer with memory optimization."""
        
        # Enable mixed precision if flag is set
        if self.use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            hidden_states = tf.cast(hidden_states, policy.compute_dtype)
            if attention_mask is not None:
                attention_mask = tf.cast(attention_mask, policy.compute_dtype)
        
        # Process in chunks if chunk size is set
        if self.chunk_size is not None:
            def chunk_forward(chunk):
                outputs = self._forward_pass(
                    chunk, attention_mask, position_ids,
                    past_key_value, output_attentions,
                    use_cache, training
                )
                return outputs[0] if isinstance(outputs, tuple) else outputs
                
            hidden_states = self.process_in_chunks(
                chunk_forward, hidden_states, self.chunk_size
            )
            return hidden_states
            
        return self._forward_pass(
            hidden_states, attention_mask, position_ids,
            past_key_value, output_attentions,
            use_cache, training
        )
        
    def _forward_pass(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions,
        use_cache,
        training,
    ):
        """Core forward pass implementation."""
        
        all_hidden_states = () if output_attentions else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        # Use gradient checkpointing if enabled
        layer_fn = self._checkpointed_layer if self.gradient_checkpointing else self._regular_layer
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_attentions:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            past_key_value_layer = past_key_value[idx] if past_key_value is not None else None
            
            hidden_states, layer_self_attn, present_key_value = layer_fn(
                decoder_layer,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value_layer,
                output_attentions,
                use_cache,
                training,
            )
            
            if use_cache:
                next_decoder_cache = next_decoder_cache + (present_key_value,)
                
            if output_attentions:
                all_self_attns = all_self_attns + (layer_self_attn,)
                
        hidden_states = self.norm(hidden_states)
        
        # Convert back to float32 for output
        if self.use_mixed_precision:
            hidden_states = tf.cast(hidden_states, tf.float32)
            
        if output_attentions:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        if not use_cache:
            return hidden_states
            
        return hidden_states, next_decoder_cache
        
    def _checkpointed_layer(
        self,
        layer,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions,
        use_cache,
        training,
    ):
        """Run layer with gradient checkpointing."""
        
        def create_custom_forward():
            def custom_forward(*inputs):
                return layer(inputs[0], 
                           attention_mask=inputs[1],
                           position_ids=inputs[2],
                           output_attentions=output_attentions,
                           use_cache=use_cache,
                           training=training)
            return custom_forward
            
        layer_outputs = tf.recompute_grad(create_custom_forward())(
            hidden_states,
            attention_mask,
            position_ids,
        )
        
        return layer_outputs
        
    def _regular_layer(
        self,
        layer,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions,
        use_cache,
        training,
    ):
        """Run layer normally."""
        return layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            training=training,
        )


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
        self.mlp = OmniGenMLP(config, name="mlp")
        
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
        # Memory optimization: use mixed precision
        policy = tf.keras.mixed_precision.global_policy()
        if policy.name == 'mixed_float16':
            hidden_states = tf.cast(hidden_states, tf.float16)
            
        residual = hidden_states
        
        # Layer norm
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self attention
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            training=training,
        )
        
        # Add residual connection
        hidden_states = attn_outputs[0]
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Memory optimization: process MLP in chunks if needed
        if tf.reduce_prod(tf.shape(hidden_states)) > 1024 * 1024:
            chunk_size = 1024  # Adjust based on available memory
            chunks = []
            
            for i in range(0, tf.shape(hidden_states)[1], chunk_size):
                chunk = hidden_states[:, i:i+chunk_size, :]
                chunk = self.mlp(chunk, training=training)
                chunks.append(chunk)
                
            hidden_states = tf.concat(chunks, axis=1)
        else:
            hidden_states = self.mlp(hidden_states, training=training)
        
        # Add residual connection
        hidden_states = residual + hidden_states
        
        # Convert back to original dtype if needed
        if policy.name == 'mixed_float16':
            hidden_states = tf.cast(hidden_states, tf.float32)
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (attn_outputs[1],)
            
        if past_key_value is not None:
            outputs += (attn_outputs[2],)
            
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
        training=False,
    ):
        """Forward pass."""
        bsz, q_len, _ = tf.shape(hidden_states)
        
        # Get query, key, value projections
        qkv = self.qkv_proj(hidden_states)
        qkv = tf.reshape(qkv, (bsz, q_len, 3, self.num_attention_heads, self.head_dim))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = tf.unstack(qkv, axis=0)
        
        # Handle past key values
        if past_key_value is not None:
            k = tf.concat([past_key_value[0], k], axis=2)
            v = tf.concat([past_key_value[1], v], axis=2)
            
        if past_key_value is not None:
            present_key_value = (k, v)
            
        # Compute attention
        attn_weights = tf.matmul(q, k, transpose_b=True)
        attn_weights = attn_weights * tf.math.rsqrt(tf.cast(self.head_dim, attn_weights.dtype))
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attn_weights = self.attention_dropout(attn_weights, training=training)
        
        attn_output = tf.matmul(attn_weights, v)
        attn_output = tf.transpose(attn_output, (0, 2, 1, 3))
        attn_output = tf.reshape(attn_output, (bsz, q_len, self.hidden_size))
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output, training=training)
        
        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
        if past_key_value is not None:
            outputs += (present_key_value,)
            
        return outputs


class OmniGenMLP(layers.Layer):
    """MLP layer for OmniGen."""
    
    def __init__(self, config: Phi3Config, **kwargs):
        """Initialize MLP layer."""
        super().__init__(**kwargs)
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Initialize components
        self.gate_up_proj = layers.Dense(2 * self.intermediate_size, use_bias=False, name="gate_up_proj")
        self.down_proj = layers.Dense(self.hidden_size, use_bias=False, name="down_proj")
        self.act_fn = tf.keras.activations.swish
        
    def call(self, x, training=False):
        """Forward pass."""
        # Split activation
        gate_up = self.gate_up_proj(x)
        gate, up = tf.split(gate_up, 2, axis=-1)
        
        # Apply activation
        return self.down_proj(self.act_fn(gate) * up)
