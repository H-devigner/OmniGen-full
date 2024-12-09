import os
import tensorflow as tf
import numpy as np
import math
from tensorflow.keras import layers
from diffusers.loaders import PeftAdapterMixin
from huggingface_hub import snapshot_download
from safetensors.tensorflow import load_file

from .transformer import Phi3Config, Phi3Transformer


def modulate(x, shift, scale):
    return x * (1 + scale[:, tf.newaxis]) + shift[:, tf.newaxis]
 

class TimestepEmbedder(tf.keras.Model):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = tf.keras.Sequential([
            layers.Dense(hidden_size, use_bias=True, activation='gelu'),
            layers.Dense(hidden_size, use_bias=True)
        ])
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                  These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = tf.exp(
            -math.log(max_period) * tf.range(half, dtype=tf.float32) / half
        )
        args = t[:, None] * freqs[None]
        embedding = tf.concat([tf.cos(args), tf.sin(args)], axis=-1)
        if dim % 2:
            embedding = tf.concat([embedding, tf.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def call(self, t, dtype=tf.float32):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class FinalLayer(tf.keras.layers.Layer):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = layers.LayerNormalization(epsilon=1e-6)
        self.linear = layers.Dense(patch_size * patch_size * out_channels)
        self.adaLN_modulation = tf.keras.Sequential([
            layers.GELU(),
            layers.Dense(2 * hidden_size)
        ])

    def call(self, x, c):
        shift, scale = tf.split(self.adaLN_modulation(c), 2, axis=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, interpolation_scale=1.0, base_size=1):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / interpolation_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PatchEmbedMR(tf.keras.layers.Layer):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, patch_size: int = 2, in_chans: int = 4, embed_dim: int = 768, bias: bool = True):
        super().__init__()
        self.proj = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, use_bias=bias)

    def call(self, x):
        x = self.proj(x)
        x = tf.reshape(x, (x.shape[0], -1, x.shape[-1]))  # NCHW -> NLC
        return x


class OmniGen(tf.keras.Model, PeftAdapterMixin):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(self, transformer_config: Phi3Config, patch_size=2, in_channels=4, pe_interpolation: float = 1.0, pos_embed_max_size: int = 192):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.pos_embed_max_size = pos_embed_max_size

        hidden_size = transformer_config.hidden_size

        self.x_embedder = PatchEmbedMR(patch_size, in_channels, hidden_size)
        self.input_x_embedder = PatchEmbedMR(patch_size, in_channels, hidden_size)

        self.time_token = TimestepEmbedder(hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        self.pe_interpolation = pe_interpolation
        pos_embed = get_2d_sincos_pos_embed(hidden_size, pos_embed_max_size, interpolation_scale=self.pe_interpolation, base_size=64)
        self.pos_embed = tf.Variable(pos_embed[None], trainable=False)

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

        self.llm = Phi3Transformer(config=transformer_config)
        self.llm.config.use_cache = False
    
    @classmethod
    def from_pretrained(cls, model_name, **args):
        if not os.path.exists(model_name):
            cache_folder = os.getenv('HF_HUB_CACHE')
            model_name = snapshot_download(repo_id=model_name, cache_dir=cache_folder, ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5'])
        config = Phi3Config.from_pretrained(model_name)
        model = cls(config)
        if os.path.exists(os.path.join(model_name, 'model.safetensors')):
            print("Loading safetensors")
            ckpt = load_file(os.path.join(model_name, 'model.safetensors'))
        else:
            ckpt = tf.saved_model.load(os.path.join(model_name, 'model.pt'))
        model.set_weights(ckpt)
        return model

    def initialize_weights(self):
        # Similar to PyTorch's Xavier uniform initialization:
        def _basic_init(layer):
            if isinstance(layer, layers.Dense):
                layer.kernel_initializer = 'glorot_uniform'
                if layer.bias is not None:
                    layer.bias_initializer = 'zeros'

        for layer in self.layers:
            _basic_init(layer)

    def unpatchify(self, x, h, w):
        c = self.out_channels
        x = tf.reshape(x, (x.shape[0], h // self.patch_size, w // self.patch_size, self.patch_size, self.patch_size, c))
        x = tf.einsum('nhwpqc->nchpwq', x)
        imgs = tf.reshape(x, (x.shape[0], c, h, w))
        return imgs


    def cropped_pos_embed(self, height, width):
        """Crops positional embeddings for SD3 compatibility."""
        if self.pos_embed_max_size is None:
            raise ValueError("pos_embed_max_size must be set to crop embeddings.")
        pos_embed = self.pos_embed
        _, h, w, _ = pos_embed.shape
        if h == height and w == width:
            return pos_embed
        return pos_embed[:, :height, :width, :]

    def call(self, x, t, c=None, return_loss=False):
        """
        Forward pass.
        """
        B, H, W, _ = x.shape
        x_embed = self.x_embedder(x)
        t_embed = self.t_embedder(t)
        pos_embed = self.cropped_pos_embed(H, W)
        
        x_embed += pos_embed
        x_embed = self.llm(x_embed, t_embed, c)
        x_out = self.final_layer(x_embed, t_embed)

        if return_loss:
            return x_out
        return self.unpatchify(x_out, H, W)
