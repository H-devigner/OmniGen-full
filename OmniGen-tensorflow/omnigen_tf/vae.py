"""VAE model for OmniGen."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, initializers
from typing import Optional, Tuple, Dict, Any
import json
import os
from safetensors import safe_open
from huggingface_hub import snapshot_download
import onnx
from onnx_tf.backend import prepare

from .utils import convert_torch_to_tf

class AutoencoderKL(tf.keras.Model):
    """VAE model with KL regularization."""
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_block_types = down_block_types
        self.up_block_types = up_block_types
        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block
        self.latent_channels = latent_channels
        self.sample_size = sample_size
        self.scaling_factor = scaling_factor
        
        # Initialize with truncated normal
        kernel_init = initializers.TruncatedNormal(stddev=0.02)
        
        # Build encoder
        self.conv_in = layers.Conv2D(
            block_out_channels[0],
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_initializer=kernel_init,
            name='conv_in'
        )
        
        self.encoder_blocks = []
        output_channel = block_out_channels[0]
        
        for i, (block_type, channel) in enumerate(zip(down_block_types, block_out_channels)):
            input_channel = output_channel
            output_channel = channel
            
            # Add encoder block
            for _ in range(layers_per_block):
                block = self._make_encoder_block(
                    block_type,
                    input_channel,
                    output_channel,
                    kernel_init,
                    name=f'encoder_block_{i}'
                )
                self.encoder_blocks.append(block)
                input_channel = output_channel
                
        self.conv_norm_out = layers.LayerNormalization(epsilon=1e-5, name='conv_norm_out')
        if act_fn == "silu":
            self.conv_act = layers.Activation('swish')
        else:
            self.conv_act = layers.Activation(act_fn)
            
        self.conv_out = layers.Conv2D(
            latent_channels,
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_initializer=kernel_init,
            name='conv_out'
        )
        
        # Build decoder
        self.decoder_blocks = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_up_block_types = list(reversed(up_block_types))
        output_channel = reversed_block_out_channels[0]
        
        for i, (block_type, channel) in enumerate(zip(reversed_up_block_types, reversed_block_out_channels)):
            input_channel = output_channel
            output_channel = channel
            
            # Add decoder block
            for _ in range(layers_per_block):
                block = self._make_decoder_block(
                    block_type,
                    input_channel,
                    output_channel,
                    kernel_init,
                    name=f'decoder_block_{i}'
                )
                self.decoder_blocks.append(block)
                input_channel = output_channel
                
        self.conv_norm_out_decoder = layers.LayerNormalization(epsilon=1e-5, name='conv_norm_out_decoder')
        if act_fn == "silu":
            self.conv_act_decoder = layers.Activation('swish')
        else:
            self.conv_act_decoder = layers.Activation(act_fn)
            
        self.conv_out_decoder = layers.Conv2D(
            out_channels,
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_initializer=kernel_init,
            name='conv_out_decoder'
        )
        
    def _make_encoder_block(
        self,
        block_type: str,
        input_channel: int,
        output_channel: int,
        kernel_init,
        name: str
    ) -> tf.keras.Sequential:
        """Create encoder block."""
        block = tf.keras.Sequential(name=name)
        
        # Add residual connection
        block.add(layers.Conv2D(
            output_channel,
            kernel_size=3,
            strides=2,
            padding='same',
            kernel_initializer=kernel_init
        ))
        block.add(layers.LayerNormalization(epsilon=1e-5))
        block.add(layers.Activation('swish'))
        
        return block
        
    def _make_decoder_block(
        self,
        block_type: str,
        input_channel: int,
        output_channel: int,
        kernel_init,
        name: str
    ) -> tf.keras.Sequential:
        """Create decoder block."""
        block = tf.keras.Sequential(name=name)
        
        # Add upsampling
        block.add(layers.UpSampling2D(size=(2, 2), interpolation='nearest'))
        block.add(layers.Conv2D(
            output_channel,
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_initializer=kernel_init
        ))
        block.add(layers.LayerNormalization(epsilon=1e-5))
        block.add(layers.Activation('swish'))
        
        return block
        
    def encode(self, x: tf.Tensor, return_dict: bool = True) -> Dict[str, tf.Tensor]:
        """Encode input."""
        h = self.conv_in(x)
        
        # Apply encoder blocks
        for block in self.encoder_blocks:
            h = block(h)
            
        h = self.conv_norm_out(h)
        h = self.conv_act(h)
        h = self.conv_out(h)
        
        mean, logvar = tf.split(h, 2, axis=-1)
        logvar = tf.clip_by_value(logvar, -30.0, 20.0)
        
        if return_dict:
            return {"mean": mean, "logvar": logvar}
        return mean, logvar
        
    def decode(self, z: tf.Tensor, return_dict: bool = True) -> tf.Tensor:
        """Decode latent."""
        h = z
        
        # Apply decoder blocks
        for block in self.decoder_blocks:
            h = block(h)
            
        h = self.conv_norm_out_decoder(h)
        h = self.conv_act_decoder(h)
        h = self.conv_out_decoder(h)
        
        if return_dict:
            return {"sample": h}
        return h
        
    def call(
        self,
        inputs: tf.Tensor,
        sample_posterior: bool = True,
        return_dict: bool = True,
        training: bool = False
    ) -> Dict[str, tf.Tensor]:
        """Forward pass."""
        posterior = self.encode(inputs, return_dict=True)
        
        if sample_posterior:
            # Reparameterization trick
            noise = tf.random.normal(tf.shape(posterior["mean"]))
            latents = posterior["mean"] + tf.exp(0.5 * posterior["logvar"]) * noise
        else:
            latents = posterior["mean"]
            
        decoded = self.decode(latents, return_dict=True)
        
        if return_dict:
            return {
                "sample": decoded["sample"],
                "mean": posterior["mean"],
                "logvar": posterior["logvar"]
            }
        return decoded["sample"]
        
    @tf.function
    def encode_to_latents(self, inputs: tf.Tensor) -> tf.Tensor:
        """Encode images to latent space."""
        posterior = self.encode(inputs, return_dict=True)
        latents = posterior["mean"]
        return latents * self.scaling_factor
        
    @tf.function
    def decode_from_latents(self, latents: tf.Tensor) -> tf.Tensor:
        """Decode images from latent space."""
        latents = latents / self.scaling_factor
        decoded = self.decode(latents, return_dict=True)
        return decoded["sample"]
        
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs
    ) -> "AutoencoderKL":
        """Load pretrained model."""
        if not os.path.isdir(pretrained_model_name_or_path):
            pretrained_model_name_or_path = snapshot_download(
                pretrained_model_name_or_path,
                allow_patterns=["*.safetensors", "*.json"]
            )
            
        config_file = os.path.join(pretrained_model_name_or_path, "config.json")
        if os.path.isfile(config_file):
            with open(config_file) as f:
                config = json.load(f)
        else:
            config = {}
            
        config.update(kwargs)
        model = cls(**config)
        
        # Load weights
        weight_files = [f for f in os.listdir(pretrained_model_name_or_path) if f.endswith('.safetensors')]
        if not weight_files:
            raise ValueError(f"No safetensors weights found in {pretrained_model_name_or_path}")
            
        weight_file = os.path.join(pretrained_model_name_or_path, weight_files[0])
        with safe_open(weight_file, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                tf_tensor = convert_torch_to_tf(tensor.detach().cpu().numpy())
                
                # Find corresponding layer and weight
                layer_name, weight_type = key.rsplit(".", 1)
                layer = model.layers[-1].adaLN_modulation[1]
                
                if weight_type == "weight":
                    layer.kernel.assign(tf_tensor)
                elif weight_type == "bias":
                    layer.bias.assign(tf_tensor)
                    
        # Load the ONNX model
        onnx_model = onnx.load('your_model.onnx')
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph('your_model_tf')
        
        return model
