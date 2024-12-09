import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Optional, Tuple, List, Union
from transformers import Phi3Config

class Phi3Transformer(Model):
    def __init__(self, config, **kwargs):
        super(Phi3Transformer, self).__init__(**kwargs)
        self.config = config
        # Renamed `layers` to `decoder_layers` to avoid conflict with TensorFlow's reserved `Model.layers`
        self.decoder_layers = [Phi3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = layers.LayerNormalization(epsilon=1e-5)
        self.use_cache = config.use_cache

    def prefetch_layer(self, layer_idx: int, device: str):
        # Prefetch logic is adapted since TensorFlow GPU memory management is automatic
        print(f"Prefetching layer {layer_idx} to device {device} (handled by TensorFlow)")

    def evict_previous_layer(self, layer_idx: int):
        # TensorFlow manages memory automatically; this is a no-op placeholder
        print(f"Evicting layer {layer_idx} (not explicitly required in TensorFlow)")

    def get_offload_layer(self, layer_idx: int, device: str):
        self.evict_previous_layer(layer_idx - 1)
        self.prefetch_layer((layer_idx + 1) % len(self.decoder_layers), device)

    def call(
        self,
        inputs_embeds: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_values: Optional[List[tf.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        offload_model: Optional[bool] = False,
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        # Default values
        output_attentions = output_attentions or self.config.output_attentions
        output_hidden_states = output_hidden_states or self.config.output_hidden_states
        use_cache = use_cache or self.config.use_cache

        hidden_states = inputs_embeds
        all_hidden_states = [] if output_hidden_states else None
        all_self_attentions = [] if output_attentions else None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.decoder_layers):  # Use renamed `decoder_layers`
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            if offload_model:
                self.get_offload_layer(layer_idx, device="GPU")

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[layer_idx] if past_key_values else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[1]

            if output_attentions:
                all_self_attentions.append(layer_outputs[1])

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return hidden_states, next_decoder_cache


class Phi3DecoderLayer(layers.Layer):
    def __init__(self, config, **kwargs):
        super(Phi3DecoderLayer, self).__init__(**kwargs)
        self.self_attention = layers.MultiHeadAttention(
            num_heads=config.num_attention_heads,
            key_dim=config.hidden_size // config.num_attention_heads,
        )
        self.feed_forward = tf.keras.Sequential([
            layers.Dense(config.intermediate_size, activation="gelu"),
            layers.Dense(config.hidden_size),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_value: Optional[tf.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        # Self-attention
        attn_output = self.self_attention(
            hidden_states,
            hidden_states,
            hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = self.norm1(hidden_states + attn_output)

        # Feed-forward
        ff_output = self.feed_forward(hidden_states)
        hidden_states = self.norm2(hidden_states + ff_output)

        return hidden_states, None  # Cache support not implemented in this basic example


# Example config for demonstration
class Config:
    def __init__(self):
        self.num_hidden_layers = 4
        self.num_attention_heads = 8
        self.hidden_size = 512
        self.intermediate_size = 2048
        self.use_cache = True
        self.output_hidden_states = True
        self.output_attentions = True

# Initialize and use the model
config = Config()
model = Phi3Transformer(config)
inputs = tf.random.uniform((2, 10, config.hidden_size))  # Batch size of 2, sequence length of 10
outputs = model(inputs)
