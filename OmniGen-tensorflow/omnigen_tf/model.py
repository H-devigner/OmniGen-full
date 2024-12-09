import os
import tensorflow as tf
import numpy as np
from .transformer import Phi3Transformer
from typing import Optional, List, Tuple

# Helper function to modulate tensor values
def modulate(x, shift, scale):
    return x * (1 + tf.expand_dims(scale, 1)) + tf.expand_dims(shift, 1)

# Timestep embedding
class TimestepEmbedder(tf.keras.layers.Layer):
    def __init__(self, hidden_size, frequency_embedding_size=256, **kwargs):
        super().__init__(**kwargs)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation='silu'),
            tf.keras.layers.Dense(hidden_size)
        ])
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = tf.exp(
            -tf.math.log(float(max_period)) * tf.range(half, dtype=tf.float32) / half
        )
        args = tf.cast(t[:, None], tf.float32) * freqs[None]
        embedding = tf.concat([tf.cos(args), tf.sin(args)], axis=-1)
        if dim % 2:
            embedding = tf.concat([embedding, tf.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def call(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)

# Final processing layer
class FinalLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_size, patch_size, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.norm_final = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.linear = tf.keras.layers.Dense(patch_size * patch_size * out_channels)
        self.adaLN_modulation = tf.keras.Sequential([
            tf.keras.layers.Activation('silu'),
            tf.keras.layers.Dense(2 * hidden_size)
        ])

    def call(self, x, c):
        shift, scale = tf.split(self.adaLN_modulation(c), 2, axis=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)

# Patch embedder for multi-resolution processing
class PatchEmbedMR(tf.keras.layers.Layer):
    def __init__(self, patch_size: int, in_chans: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.proj = tf.keras.layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, use_bias=True)

    def call(self, x):
        x = self.proj(x)
        x = tf.reshape(x, [x.shape[0], -1, x.shape[-1]])
        return x

# Main model class
class OmniGen(tf.keras.Model):
    def __init__(self, transformer_config, patch_size=2, in_channels=4, pos_embed_max_size=192, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.pos_embed_max_size = pos_embed_max_size

        hidden_size = transformer_config.hidden_size

        # Embedding layers
        self.x_embedder = PatchEmbedMR(patch_size, in_channels, hidden_size)
        self.input_x_embedder = PatchEmbedMR(patch_size, in_channels, hidden_size)
        self.time_token = TimestepEmbedder(hidden_size)

        # Positional embeddings
        self.pos_embed = tf.Variable(
            tf.random.uniform([1, pos_embed_max_size * pos_embed_max_size, hidden_size], dtype=tf.float32),
            trainable=False
        )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        # Load pre-trained transformer model
        self.llm = Phi3Transformer.from_pretrained(transformer_config.name_or_path)

        self._initialize_weights()

    def _initialize_weights(self):
        initializer = tf.keras.initializers.GlorotUniform()

        # Initialize patch embedders
        for embedder in [self.x_embedder, self.input_x_embedder]:
            embedder.proj.kernel.assign(initializer(embedder.proj.kernel.shape))
            embedder.proj.bias.assign(tf.zeros_like(embedder.proj.bias))

    def cropped_pos_embed(self, height, width):
        h, w = height // self.patch_size, width // self.patch_size
        pos_embed = self.pos_embed[:, :h * w, :]
        return pos_embed

    def call(
        self,
        x,
        timestep,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        training=False
    ):
        B, C, H, W = x.shape
        x_embed = self.x_embedder(x)

        # Timestep embedding
        t_embed = self.time_token(timestep)

        # Positional embeddings
        pos_embed = self.cropped_pos_embed(H, W)

        # Combine embeddings and pass through the transformer
        transformer_input = x_embed + pos_embed
        outputs = self.llm(
            inputs_embeds=transformer_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            training=training
        )
        transformer_out = outputs.last_hidden_state

        # Process through the final layer
        final_out = self.final_layer(transformer_out, t_embed)
        return final_out

# Usage Example
if __name__ == "__main__":
    # Define a mock transformer configuration
    from transformers import AutoConfig

    transformer_config = AutoConfig.from_pretrained("EleutherAI/phi-1_5")
    model = OmniGen(transformer_config)

    # Mock data
    batch_size = 2
    num_channels = 4
    img_height, img_width = 64, 64
    x = tf.random.normal([batch_size, img_height, img_width, num_channels])
    timestep = tf.random.uniform([batch_size], minval=0, maxval=1000, dtype=tf.int32)

    output = model(x, timestep)
    print("Output shape:", output.shape)
