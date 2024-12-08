import torch
import os
from huggingface_hub import hf_hub_download
import logging
import json
from pathlib import Path
import tensorflow as tf
import numpy as np
from transformers import AutoConfig
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class TFOmniGen(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._build_layers()
    
    def _build_layers(self):
        """Initialize the model layers"""
        logger.info("Building TensorFlow model layers")
        # Initialize layers based on config
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.config["hidden_size"], 3, padding='same'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('gelu')
        ])
        
        # Add transformer layers
        self.transformer_layers = []
        for _ in range(self.config["num_hidden_layers"]):
            self.transformer_layers.append(
                tf.keras.layers.MultiHeadAttention(
                    num_heads=self.config["num_attention_heads"],
                    key_dim=self.config["hidden_size"] // self.config["num_attention_heads"]
                )
            )
        
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(4, 3, padding='same'),
            tf.keras.layers.LayerNormalization()
        ])
    
    def call(self, inputs):
        x = inputs['x']
        timestep = inputs['timestep']
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = inputs['position_ids']
        
        # Encode
        x = self.encoder(x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, x)
        
        # Decode
        x = self.decoder(x)
        
        return x

def download_pytorch_model(model_name):
    """Download PyTorch model from HuggingFace"""
    logger.info(f"Downloading PyTorch model from {model_name}")
    
    # Download config
    config_path = hf_hub_download(model_name, filename="config.json")
    with open(config_path) as f:
        config = json.load(f)
    
    # Try to load from safetensors first
    try:
        weights_path = hf_hub_download(model_name, filename="model.safetensors")
        from safetensors.torch import load_file
        state_dict = load_file(weights_path)
        logger.info("Loaded weights from safetensors")
    except:
        weights_path = hf_hub_download(model_name, filename="pytorch_model.bin")
        state_dict = torch.load(weights_path)
        logger.info("Loaded weights from PyTorch checkpoint")
    
    return config, state_dict

def convert_weights(state_dict):
    """Convert PyTorch state dict to TensorFlow compatible format"""
    logger.info("Converting weights to TensorFlow format")
    tf_weights = {}
    
    for name, param in state_dict.items():
        # Convert tensor to numpy
        param_numpy = param.cpu().numpy()
        
        # Handle convolution weights
        if 'conv' in name and 'weight' in name:
            # PyTorch conv weights are (out_channels, in_channels, height, width)
            # TF conv weights are (height, width, in_channels, out_channels)
            param_numpy = np.transpose(param_numpy, (2, 3, 1, 0))
        
        # Handle linear/dense weights
        elif 'weight' in name and len(param.shape) == 2:
            # PyTorch linear weights are (out_features, in_features)
            # TF dense weights are (in_features, out_features)
            param_numpy = np.transpose(param_numpy)
        
        tf_weights[name] = param_numpy
    
    return tf_weights

def create_tf_model(config, tf_weights):
    """Create and initialize TensorFlow model"""
    logger.info("Creating TensorFlow model")
    model = TFOmniGen(config)
    
    # Build model with dummy inputs
    dummy_inputs = {
        'x': tf.random.normal((1, 4, 64, 64)),
        'timestep': tf.zeros((1,), dtype=tf.int32),
        'input_ids': tf.zeros((1, 77), dtype=tf.int32),
        'attention_mask': tf.ones((1, 77), dtype=tf.int32),
        'position_ids': tf.range(77)[None]
    }
    _ = model(dummy_inputs)
    
    # Set weights
    logger.info("Setting model weights")
    for name, weight in tf_weights.items():
        try:
            model.get_layer(name).set_weights([weight])
            logger.debug(f"Set weights for layer: {name}")
        except Exception as e:
            logger.warning(f"Could not set weights for {name}: {str(e)}")
    
    return model

def download_and_convert(model_name="Shitao/omnigen-v1", output_dir="./"):
    """Download PyTorch model and convert to TensorFlow"""
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Step 1: Download PyTorch model
        logger.info("Step 1: Downloading PyTorch model")
        config, state_dict = download_pytorch_model(model_name)
        
        # Step 2: Convert weights
        logger.info("Step 2: Converting weights")
        tf_weights = convert_weights(state_dict)
        
        # Step 3: Create and initialize TF model
        logger.info("Step 3: Creating TensorFlow model")
        tf_model = create_tf_model(config, tf_weights)
        
        # Step 4: Save model
        logger.info("Step 4: Saving TensorFlow model")
        tf_path = output_dir / "tf_model"
        tf.saved_model.save(tf_model, str(tf_path))
        
        logger.info("Conversion completed successfully")
        return tf_path
        
    except Exception as e:
        logger.error(f"Error during conversion: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting model conversion")
        result = download_and_convert()
        logger.info(f"Conversion completed: {result}")
    except Exception as e:
        logger.error("Conversion failed:", exc_info=True)
        raise
