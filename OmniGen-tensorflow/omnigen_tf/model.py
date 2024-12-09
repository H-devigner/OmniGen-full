"""TensorFlow implementation of OmniGen"""

import os
import json
import logging
from pathlib import Path
import tensorflow as tf
import numpy as np
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
import torch

logger = logging.getLogger(__name__)

class PatchEmbedMR(tf.keras.layers.Layer):
    """Patch Embedding layer"""
    
    def __init__(self, patch_size, in_chans, embed_dim, bias=True):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.bias = bias
        
        self.proj = tf.keras.layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size)
        if self.bias:
            self.bias_layer = tf.keras.layers.Dense(embed_dim)
    
    def call(self, x):
        """Forward pass"""
        x = self.proj(x)
        if self.bias:
            x = x + self.bias_layer(x)
        return x

class OmniGenTF(tf.keras.Model):
    """TensorFlow implementation of OmniGen"""
    
    def __init__(self, config=None, model_path=None):
        super().__init__()
        self.config = config or self._load_config(model_path)
        self._build_model()
    
    @staticmethod
    def _load_config(model_path):
        """Load model config from path"""
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            return json.load(f)
    
    def _build_model(self):
        """Build the model architecture"""
        # Text encoder
        self.text_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(self.config["hidden_size"]),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('gelu'),
            # Add additional layers as needed to match PyTorch architecture
        ], name='text_encoder')
        
        # Image encoder
        self.image_encoder = PatchEmbedMR(patch_size=self.config["patch_size"],
                                    in_chans=self.config["in_channels"],
                                    embed_dim=self.config["hidden_size"],
                                    bias=True)
        
        # Transformer layers
        self.transformer_layers = []
        for i in range(self.config['num_hidden_layers']):
            layer = tf.keras.layers.MultiHeadAttention(
                num_heads=self.config['num_attention_heads'],
                key_dim=self.config['hidden_size'] // self.config['num_attention_heads'],
                name=f'transformer_layer_{i}'
            )
            self.transformer_layers.append(layer)
        
        # Final layer
        self.final_layer = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(epsilon=1e-6),
            tf.keras.layers.Dense(self.config['patch_size'] * self.config['patch_size'] * self.config['in_channels']),
            # Add any additional layers or modifications as needed
        ], name='final_layer')
        
        # Decoder
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(4, 3, padding='same'),
            tf.keras.layers.LayerNormalization()
        ], name='decoder')
    
    def call(self, inputs):
        """Forward pass"""
        # Unpack inputs
        x = inputs['x']
        timestep = inputs['timestep']
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = inputs['position_ids']
        
        # Process timestep embedding
        timestep = tf.cast(timestep, tf.float32)[:, None]
        timestep_emb = self.get_timestep_embedding(timestep)
        
        # Process text
        text_features = self.text_encoder(input_ids)
        text_features = text_features * tf.cast(attention_mask[:, :, None], tf.float32)
        
        # Process image
        image_features = self.image_encoder(x)
        
        # Combine features
        logger.debug(f"Image features shape: {image_features.shape}")
        logger.debug(f"Text features shape before addition: {text_features.shape}")

        # Ensure text_features has the correct shape for addition
        text_features = tf.expand_dims(text_features, axis=1)  # Add a dimension
        text_features = tf.expand_dims(text_features, axis=1)  # Add another dimension

        # Check if the shapes are compatible for addition
        if image_features.shape[-1] != text_features.shape[-1]:
            # Adjust text_features to match image_features shape
            text_features = tf.reshape(text_features, (tf.shape(text_features)[0], -1))
            logger.debug(f"Adjusted Text features shape: {text_features.shape}")

        # Ensure the shapes are compatible for addition
        if image_features.shape[1:] != text_features.shape[1:]:
            raise ValueError(f"Incompatible shapes for addition: {image_features.shape} + {text_features.shape}")

        features = image_features + text_features
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            features = layer(features, features)
        
        # Apply final layer
        features = self.final_layer(features)
        
        # Decode
        output = self.decoder(features)
        
        return output
    
    def get_timestep_embedding(self, timesteps, dim=320, max_period=10000):
        """Create sinusoidal timestep embeddings"""
        max_period = tf.cast(max_period, tf.float32)  # Ensure max_period is float32
        half = dim // 2
        freqs = tf.exp(
            -tf.math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half
        )
        args = timesteps * freqs[None]
        embedding = tf.concat([tf.cos(args), tf.sin(args)], axis=-1)
        if dim % 2:
            embedding = tf.pad(embedding, [[0, 0], [0, 1]])
        return embedding
    
    @classmethod
    def from_pretrained(cls, model_name_or_path, local_files_only=False):
        """Load pretrained model"""
        if not os.path.exists(model_name_or_path):
            logger.info(f"Downloading model {model_name_or_path}")
            model_path = snapshot_download(
                model_name_or_path,
                local_files_only=local_files_only
            )
        else:
            model_path = model_name_or_path
        
        # Load config
        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        # Set default values if keys are missing
        config.setdefault('patch_size', 2)
        config.setdefault('in_channels', 4)
        config.setdefault('hidden_size', 768)
        config.setdefault('num_hidden_layers', 12)
        config.setdefault('num_attention_heads', 12)

        # Create model
        model = cls(config=config)
        
        # Load weights
        try:
            # Load weights from safetensors or PyTorch
            weights_path = os.path.join(model_path, "model.safetensors")
            if os.path.exists(weights_path):
                state_dict = load_file(weights_path)
            else:
                weights_path = os.path.join(model_path, "pytorch_model.bin")
                state_dict = torch.load(weights_path)
            
            # Convert weights
            tf_weights = {}
            for name, param in state_dict.items():
                weight = param.cpu().numpy()
                
                # Handle convolution weights
                if 'conv' in name and 'weight' in name:
                    weight = np.transpose(weight, (2, 3, 1, 0))
                # Handle linear/dense weights
                elif 'weight' in name and len(param.shape) == 2:
                    weight = np.transpose(weight)
                
                tf_weights[name] = weight
            
            # Set weights
            for name, weight in tf_weights.items():
                try:
                    layer = model.get_layer(name.split('.')[0])
                    layer.set_weights([weight])
                except Exception as e:
                    logger.warning(f"Could not set weights for {name}: {str(e)}")
    
        except Exception as e:
            logger.error(f"Error loading weights: {str(e)}")
            raise
        
        return model
