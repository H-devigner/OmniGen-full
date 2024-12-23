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
    We only modified the attention mask
    
    Args:
        config: Phi3Config
    """
    def __init__(self, config, dtype=None):
        super().__init__()
        self.config = config
        self._dtype = dtype or tf.float32
        self.num_hidden_layers = config.num_hidden_layers
        
        # Initialize layers
        self.decoder_layers = []
        for i in range(config.num_hidden_layers):
            layer = Phi3DecoderLayer(config, dtype=self._dtype)
            self.decoder_layers.append(layer)
            
        self.norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps,
            dtype=self._dtype
        )
        
        # Memory management
        self._current_device = None
        self._active_layer_idx = None
        self._layer_weights_cpu = [None] * len(self.decoder_layers)
        self._prefetch_queue = []
        
    def _get_device_strategy(self):
        """Get current device strategy."""
        if self._current_device is None:
            # Default to GPU if available
            gpus = tf.config.list_physical_devices('GPU')
            self._current_device = '/GPU:0' if gpus else '/CPU:0'
        return self._current_device
        
    def prefetch_layer(self, layer_idx: int):
        """Prefetch next layer weights to device."""
        if layer_idx >= len(self.decoder_layers):
            return
            
        # Skip if weights already on device
        if self._active_layer_idx == layer_idx:
            return
            
        device = self._get_device_strategy()
        with tf.device(device):
            # If weights are on CPU, restore them
            if self._layer_weights_cpu[layer_idx] is not None:
                weights = self._layer_weights_cpu[layer_idx]
                self.decoder_layers[layer_idx].set_weights([
                    tf.convert_to_tensor(w, dtype=self._dtype) 
                    for w in weights
                ])
                self._layer_weights_cpu[layer_idx] = None
                
        self._prefetch_queue.append(layer_idx)
                
    def evict_layer(self, layer_idx: int):
        """Move layer weights to CPU."""
        if layer_idx < 0 or self._layer_weights_cpu[layer_idx] is not None:
            return
            
        # Store weights on CPU
        weights = self.decoder_layers[layer_idx].get_weights()
        self._layer_weights_cpu[layer_idx] = [w.numpy() for w in weights]
        
        # Clear from device
        with tf.device('/CPU:0'):
            self.decoder_layers[layer_idx].set_weights([
                tf.zeros_like(w, dtype=self._dtype)
                for w in weights
            ])
            
        if self._active_layer_idx == layer_idx:
            self._active_layer_idx = None
            
    def manage_layer_memory(self, current_idx: int):
        """Manage layer memory by prefetching and evicting."""
        # Evict layers we don't need
        if self._active_layer_idx is not None and self._active_layer_idx != current_idx:
            self.evict_layer(self._active_layer_idx)
            
        # Prefetch current layer
        self.prefetch_layer(current_idx)
        
        # Prefetch next layer
        next_idx = current_idx + 1
        if next_idx < len(self.decoder_layers):
            self.prefetch_layer(next_idx)
            
        self._active_layer_idx = current_idx
        
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
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        offload_model=False,
        training=False,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify either input_ids or inputs_embeds")
            
        # Handle attention mask format
        if attention_mask is not None and len(tf.shape(attention_mask)) == 3:
            dtype = inputs_embeds.dtype
            min_val = tf.experimental.numpy.finfo(dtype.as_numpy_dtype).min
            attention_mask = (1.0 - tf.cast(attention_mask, dtype)) * min_val
            attention_mask = tf.expand_dims(attention_mask, axis=1)
        else:
            raise Exception("attention_mask parameter was unavailable or invalid")
            
        hidden_states = inputs_embeds
        
        # Initialize caches
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        
        # Process through layers
        for idx, decoder_layer in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            # Memory management
            if offload_model and not training:
                self.manage_layer_memory(idx)
                
            # Layer forward pass
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                training=training
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
                
            if output_attentions:
                all_self_attns = all_self_attns + (layer_outputs[1],)
                
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Add final hidden states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None)
            
        return {
            'last_hidden_state': hidden_states,
            'past_key_values': next_decoder_cache,
            'hidden_states': all_hidden_states,
            'attentions': all_self_attns,
        }

    def initialize_weights(self):
        """Initialize transformer weights."""
        def _init_weights(layer):
            if isinstance(layer, tf.keras.layers.Dense):
                # Initialize weight matrices with truncated normal
                kernel_shape = layer.kernel.shape
                stddev = 1.0 / tf.sqrt(tf.cast(kernel_shape[-1], tf.float32))
                layer.kernel.assign(
                    tf.random.truncated_normal(kernel_shape, stddev=stddev)
                )
                
                # Initialize biases to zero if present
                if layer.use_bias:
                    layer.bias.assign(tf.zeros_like(layer.bias))
                    
            elif isinstance(layer, tf.keras.layers.LayerNormalization):
                # Initialize scale (gamma) to ones and offset (beta) to zeros
                # Only initialize if they are tf.Variables (not booleans)
                if hasattr(layer, 'scale') and isinstance(layer.scale, tf.Variable):
                    layer.scale.assign(tf.ones_like(layer.scale))
                if hasattr(layer, 'offset') and isinstance(layer.offset, tf.Variable):
                    layer.offset.assign(tf.zeros_like(layer.offset))
                    
            elif isinstance(layer, tf.keras.layers.Embedding):
                # Initialize embeddings with truncated normal
                kernel_shape = layer.embeddings.shape
                stddev = 1.0 / tf.sqrt(tf.cast(kernel_shape[-1], tf.float32))
                layer.embeddings.assign(
                    tf.random.truncated_normal(kernel_shape, stddev=stddev)
                )
        
        # Initialize all layers recursively
        for layer in self.decoder_layers:
            if hasattr(layer, 'layers'):  # For nested layers/models
                for sublayer in layer.layers:
                    _init_weights(sublayer)
            else:
                _init_weights(layer)
                
        print("Transformer weights initialized successfully")


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
