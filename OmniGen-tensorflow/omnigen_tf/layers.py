import tensorflow as tf
from tensorflow.keras import layers



class TimestepEmbedder(layers.Layer):
    """Embeds scalar timesteps into vector representations."""
    
    def __init__(self, hidden_size, frequency_embedding_size=256, **kwargs):
        super().__init__(**kwargs)
        self.mlp = tf.keras.Sequential([
            layers.Dense(hidden_size, activation=None),
            layers.Activation('swish'),
            layers.Dense(hidden_size, activation=None),
        ])
        self.frequency_embedding_size = frequency_embedding_size
        
    def timestep_embedding(self, t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = tf.exp(
            -tf.math.log(float(max_period)) * 
            tf.range(0, half, dtype=tf.float32) / float(half)
        )
        args = tf.cast(t[:, None], tf.float32) * freqs[None]
        embedding = tf.concat([tf.cos(args), tf.sin(args)], axis=-1)
        if dim % 2:
            embedding = tf.concat([embedding, tf.zeros_like(embedding[:, :1])], axis=-1)
        return embedding
        
    def call(self, t, training=False):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class FinalLayer(layers.Layer):
    """The final layer of OmniGen."""
    
    def __init__(self, hidden_size, patch_size, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.norm_final = layers.LayerNormalization(
            epsilon=1e-6,
            center=False,
            scale=False,
            dtype=tf.float16
        )
        self.proj = layers.Dense(
            patch_size * patch_size * out_channels,
            use_bias=True,
            dtype=tf.float16
        )
        self.adaLN_modulation = tf.keras.Sequential([
            layers.Activation('swish'),
            layers.Dense(2 * hidden_size, use_bias=True)
        ])
        
    def call(self, x, c, training=False):
        shift, scale = tf.split(self.adaLN_modulation(c), 2, axis=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.proj(x)
        return x


class PatchEmbedMR(layers.Layer):
    """2D Image to Patch Embedding with mixed precision support."""
    
    def __init__(self, patch_size=2, in_channels=4, embed_dim=768, bias=True, **kwargs):
        super().__init__(**kwargs)
        self.proj = layers.Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            use_bias=bias,
            dtype=tf.float16
        )
        
    def call(self, x, training=False):
        # Rearrange input to [B, H, W, C]
        if len(tf.shape(x)) == 3:
            B, L, C = tf.shape(x)
            H = W = int(tf.sqrt(tf.cast(L, tf.float32)))
            x = tf.reshape(x, [B, H, W, C])
            
        x = self.proj(x)  # [B, H', W', D]
        B, H, W, D = tf.shape(x)
        x = tf.reshape(x, [B, H*W, D])  # [B, L, D]
        return x
