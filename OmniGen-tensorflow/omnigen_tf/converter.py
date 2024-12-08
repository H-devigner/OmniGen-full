"""Weight converter for PyTorch to TensorFlow."""
from __future__ import annotations

import numpy as np
import tensorflow as tf
import torch
from typing import Dict, Any, Union, List

class WeightConverter:
    """Converts PyTorch weights to TensorFlow format."""
    
    def __init__(self):
        self.name_map = {
            'weight': 'kernel',
            'running_mean': 'moving_mean',
            'running_var': 'moving_variance'
        }
    
    def _convert_name(self, pt_name: str) -> str:
        """Convert PyTorch parameter names to TensorFlow names."""
        tf_name = pt_name
        for pt_key, tf_key in self.name_map.items():
            tf_name = tf_name.replace(pt_key, tf_key)
        return tf_name
    
    def _convert_tensor(self, weight_np: np.ndarray, pt_name: str) -> np.ndarray:
        """Convert tensor based on layer type and shape."""
        # Skip conversion for 1D tensors (biases, etc.)
        if len(weight_np.shape) <= 1:
            return weight_np
            
        try:
            # Handle convolution weights
            if 'conv' in pt_name.lower() or 'downsample' in pt_name.lower():
                if len(weight_np.shape) == 4:
                    # PyTorch: [out_channels, in_channels, height, width]
                    # TensorFlow: [height, width, in_channels, out_channels]
                    return np.transpose(weight_np, (2, 3, 1, 0))
                elif len(weight_np.shape) == 3:
                    # For 1D convolutions
                    return np.transpose(weight_np, (2, 1, 0))
                    
            # Handle attention/linear layer weights
            elif any(x in pt_name.lower() for x in ['linear', 'dense', 'attention', 'mlp']):
                if len(weight_np.shape) == 2:
                    # PyTorch: [out_features, in_features]
                    # TensorFlow: [in_features, out_features]
                    return np.transpose(weight_np, (1, 0))
                elif len(weight_np.shape) == 3:
                    # For attention layers with extra dimension
                    return np.transpose(weight_np, (0, 2, 1))
                    
            # Handle batch norm weights
            elif 'batch' in pt_name.lower() or 'bn' in pt_name.lower():
                # Keep original shape for batch norm
                return weight_np
                
        except Exception as e:
            print(f"Warning: Failed to convert weights for {pt_name} with shape {weight_np.shape}: {str(e)}")
            # Return original weights if conversion fails
            return weight_np
            
        # Default: return original weights if no conversion rule matches
        return weight_np
    
    def convert_torch_to_tf(self, state_dict: Dict[str, Any]) -> List[np.ndarray]:
        """Convert PyTorch state dict to TensorFlow weights."""
        tf_weights = []
        
        for pt_name, weight in state_dict.items():
            # Convert to numpy array
            if isinstance(weight, torch.Tensor):
                weight_np = weight.detach().cpu().numpy()
            else:
                weight_np = np.array(weight)
            
            # Convert tensor format
            weight_np = self._convert_tensor(weight_np, pt_name)
            
            tf_weights.append(weight_np)
            
        return tf_weights
        
    def convert_tf_to_torch(self, tf_weights: List[np.ndarray], state_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert TensorFlow weights back to PyTorch format."""
        torch_state_dict = {}
        
        for (pt_name, orig_tensor), tf_weight in zip(state_dict.items(), tf_weights):
            # Convert back to PyTorch format
            if len(tf_weight.shape) == 4:
                # Convert back from TF to PT conv format
                weight_np = np.transpose(tf_weight, (3, 2, 0, 1))
            elif len(tf_weight.shape) == 2:
                # Convert back from TF to PT linear format
                weight_np = np.transpose(tf_weight, (1, 0))
            else:
                weight_np = tf_weight
                
            torch_state_dict[pt_name] = torch.from_numpy(weight_np)
            
        return torch_state_dict
