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
    def update(self, key: str, key_value: Tuple[tf.Tensor, tf.Tensor], layer_idx: int):
        """Concatenate new key-value pairs with existing ones."""
        if layer_idx not in self.cache:
            self.cache[layer_idx] = {}
            self.cache[layer_idx][key] = key_value
        else:
            k, v = key_value
            if key in self.cache[layer_idx]:
                old_k, old_v = self.cache[layer_idx][key]
                k = tf.concat([old_k, k], axis=1)
                v = tf.concat([old_v, v], axis=1)
            self.cache[layer_idx][key] = (k, v)


class StaticCache(Cache):
    """Static cache with fixed size."""
    def update(self, key: str, key_value: Tuple[tf.Tensor, tf.Tensor], layer_idx: int):
        """Update cache without growing."""
        if layer_idx not in self.cache:
            self.cache[layer_idx] = {}
        self.cache[layer_idx][key] = key_value


class Phi3Transformer(tf.keras.Model):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers.
    Each layer is a [`Phi3DecoderLayer`]. We only modified the attention mask.
    
    Args:
        config: Phi3Config configuration object
    """
    def __init__(self, config: Phi3Config, **kwargs):
        super().__init__(**kwargs)
        print("Initializing Phi3Transformer with config")
        
        self.config = config
        # Get layer norm epsilon, default to 1e-5 if not present
        layer_norm_eps = getattr(config, 'layer_norm_eps', 1e-5)
        
        # Renamed `layers` to `decoder_layers` to avoid conflict with TensorFlow's reserved `Model.layers`
        self.decoder_layers = [Phi3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = layers.LayerNormalization(epsilon=layer_norm_eps)
        self.use_cache = config.use_cache
        
        print("Phi3Transformer initialization completed")
        print(f"Number of decoder layers: {len(self.decoder_layers)}")
        print(f"Hidden size: {config.hidden_size}")
        print(f"Layer norm epsilon: {layer_norm_eps}")

    def prefetch_layer(self, layer_idx: int, device: str):
        """Prefetch layer to device (TensorFlow handles this automatically)."""
        logger.info(f"Layer {layer_idx} prefetch handled by TensorFlow")

    def evict_previous_layer(self, layer_idx: int):
        """Evict layer from device (TensorFlow handles this automatically)."""
        logger.info(f"Layer {layer_idx} eviction handled by TensorFlow")

    def get_offload_layer(self, layer_idx: int, device: str):
        """Handle layer offloading (TensorFlow manages memory automatically)."""
        self.evict_previous_layer(layer_idx - 1)
        self.prefetch_layer((layer_idx + 1) % len(self.decoder_layers), device)

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
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Past key-value states for caching
            inputs_embeds: Pre-computed input embeddings
            use_cache: Whether to use caching
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a ModelOutput object
            cache_position: Position in cache
            offload_model: Whether to offload model to CPU
            training: Whether in training mode
            
        Returns:
            Model outputs with hidden states and optional cache
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if attention_mask is not None and len(tf.shape(attention_mask)) == 3:
            dtype = inputs_embeds.dtype
            min_dtype = tf.dtypes.as_dtype(dtype).min
            attention_mask = (1 - attention_mask) * min_dtype
            attention_mask = attention_mask[:, None, :, :]
        else:
            attention_mask = self._update_causal_mask(
                attention_mask,
                tf.shape(inputs_embeds),
                inputs_embeds.dtype,
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for idx, decoder_layer in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if offload_model and not training:
                self.get_offload_layer(idx, device="GPU")

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[idx] if past_key_values is not None else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                training=training,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Phi3DecoderLayer(tf.keras.Model):
    """Decoder layer for Phi3 model with self-attention and feed-forward networks."""
    
    def __init__(self, config: Phi3Config, **kwargs):
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
        cache_position: Optional[tf.Tensor] = None,
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
            cache_position: Position in cache
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
