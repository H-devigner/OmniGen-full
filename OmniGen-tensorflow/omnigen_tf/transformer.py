"""OmniGen Transformer implementation."""

import math
from typing import Optional, Tuple, Union, Dict
from dataclasses import dataclass, field

import tensorflow as tf
from transformers import PretrainedConfig


class Phi3Config(PretrainedConfig):
    """Configuration class for Phi3 model."""
    model_type = "phi3"
    
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
        self.use_return_dict = use_return_dict
        
        # Compute derived attributes
        self.head_dim = self.hidden_size // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads (got hidden_size={self.hidden_size} "
                f"and num_attention_heads={self.num_attention_heads})"
            )


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
        # Enable mixed precision by default
        if tf.config.list_physical_devices('GPU'):
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
        super().__init__(config, *args, **kwargs)
        
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        
        # Initialize layers with CPU placement and float16 dtype for GPU operations
        with tf.device('/CPU:0'):
            self.wte = layers.Embedding(config.vocab_size, self.embed_dim, dtype='float16')
            self.drop = layers.Dropout(config.hidden_dropout)
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
        past_key_values: Optional[List[tf.Tensor]] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        offload_model: Optional[bool] = False,
        training: bool = False,
    ) -> Union[Tuple, dict]:
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


class OmniGenTransformer(tf.keras.layers.Layer):
    """Transformer model for OmniGen."""
    
    def __init__(self, config, **kwargs):
        """Initialize transformer.
        
        Args:
            config: Model configuration
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        # Initialize layers
        self.layers = [OmniGenLayer(config, name=f"layer_{i}") for i in range(self.num_hidden_layers)]
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm")
        
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[tf.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, ...], Dict[str, tf.Tensor]]:
        """Forward pass."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Initialize outputs
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[i] if past_key_values is not None else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
                training=training,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
                
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        if not return_dict:
            return tuple(v for v in [
                hidden_states,
                next_decoder_cache,
                all_hidden_states,
                all_self_attentions,
            ] if v is not None)
            
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_decoder_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attentions,
        }


class OmniGenLayer(tf.keras.layers.Layer):
    """Transformer layer for OmniGen."""
    
    def __init__(self, config, **kwargs):
        """Initialize layer."""
        super().__init__(**kwargs)
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        # Initialize components
        self.input_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="input_layernorm")
        self.self_attn = OmniGenAttention(config, name="self_attn")
        self.post_attention_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="post_attention_layernorm")
        self.mlp = OmniGenMLP(config, name="mlp")
        
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_value: Optional[Tuple[tf.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor, ...]:
        """Forward pass."""
        residual = hidden_states
        
        # Self attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            training=training,
        )
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
            
        return outputs


class OmniGenAttention(tf.keras.layers.Layer):
    """Multi-head attention layer for OmniGen."""
    
    def __init__(self, config, **kwargs):
        """Initialize attention layer."""
        super().__init__(**kwargs)
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        
        if self.head_dim * self.num_attention_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_attention_heads`: {self.num_attention_heads})."
            )
            
        # Initialize components
        self.qkv_proj = tf.keras.layers.Dense(3 * self.hidden_size, use_bias=False, name="qkv_proj")
        self.o_proj = tf.keras.layers.Dense(self.hidden_size, use_bias=False, name="o_proj")
        
        self.attention_dropout = tf.keras.layers.Dropout(config.attention_dropout)
        self.resid_dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        
    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        """Reshape tensor for attention computation."""
        return tf.transpose(
            tf.reshape(tensor, (bsz, seq_len, self.num_attention_heads, self.head_dim)),
            (0, 2, 1, 3)
        )
        
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_value: Optional[Tuple[tf.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor], Optional[Tuple[tf.Tensor]]]:
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
            
        if use_cache:
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
        if use_cache:
            outputs += (present_key_value,)
            
        return outputs


class OmniGenMLP(tf.keras.layers.Layer):
    """MLP layer for OmniGen."""
    
    def __init__(self, config, **kwargs):
        """Initialize MLP layer."""
        super().__init__(**kwargs)
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Initialize components
        self.gate_up_proj = tf.keras.layers.Dense(2 * self.intermediate_size, use_bias=False, name="gate_up_proj")
        self.down_proj = tf.keras.layers.Dense(self.hidden_size, use_bias=False, name="down_proj")
        self.act_fn = tf.keras.activations.swish
        
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Forward pass."""
        # Split activation
        gate_up = self.gate_up_proj(x)
        gate, up = tf.split(gate_up, 2, axis=-1)
        
        # Apply activation
        return self.down_proj(self.act_fn(gate) * up)
