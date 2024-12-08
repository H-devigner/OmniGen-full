"""Weight conversion utilities for OmniGen."""

import numpy as np
import tensorflow as tf
import torch
import logging

logger = logging.getLogger(__name__)

class WeightConverter:
    def __init__(self, state_dict):
        self.state_dict = state_dict

    def convert_weights(self):
        """Convert PyTorch state dict to TensorFlow compatible format"""
        logger.info("Converting weights to TensorFlow format")
        tf_weights = {}
        
        for name, param in self.state_dict.items():
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
            logger.debug(f"Converted weights for layer: {name}")
        
        return tf_weights

    @staticmethod
    def load_weights(model, tf_weights):
        """Load converted weights into TensorFlow model"""
        logger.info("Loading converted weights into TensorFlow model")
        for name, weight in tf_weights.items():
            try:
                layer = model.get_layer(name.split('.')[0])  # Get base layer name
                layer.set_weights([weight])
                logger.debug(f"Set weights for layer: {name}")
            except Exception as e:
                logger.warning(f"Could not set weights for {name}: {str(e)}")

        logger.info("Weights loaded successfully")
