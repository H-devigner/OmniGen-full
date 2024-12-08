"""Utility functions for OmniGen TensorFlow implementation."""
import tensorflow as tf
import numpy as np
from typing import Union, Dict, Any

def convert_torch_to_tf(
    tensor: Union[np.ndarray, Dict[str, Any]]
) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
    """Convert PyTorch tensor to TensorFlow tensor.
    
    Args:
        tensor: PyTorch tensor as numpy array or dict
        
    Returns:
        TensorFlow tensor or dict of tensors
    """
    if isinstance(tensor, dict):
        return {k: convert_torch_to_tf(v) for k, v in tensor.items()}
        
    # Convert numpy array
    if isinstance(tensor, np.ndarray):
        # Handle different tensor types
        if len(tensor.shape) == 4:  # NCHW -> NHWC
            tensor = np.transpose(tensor, (0, 2, 3, 1))
        elif len(tensor.shape) == 3:  # CHW -> HWC
            tensor = np.transpose(tensor, (1, 2, 0))
            
        return tf.convert_to_tensor(tensor)
        
    raise ValueError(f"Unsupported tensor type: {type(tensor)}")

def convert_tf_to_torch(
    tensor: Union[tf.Tensor, Dict[str, Any]]
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Convert TensorFlow tensor to PyTorch tensor.
    
    Args:
        tensor: TensorFlow tensor or dict
        
    Returns:
        PyTorch tensor as numpy array or dict
    """
    if isinstance(tensor, dict):
        return {k: convert_tf_to_torch(v) for k, v in tensor.items()}
        
    # Convert TensorFlow tensor
    if isinstance(tensor, tf.Tensor):
        tensor = tensor.numpy()
        
        # Handle different tensor types
        if len(tensor.shape) == 4:  # NHWC -> NCHW
            tensor = np.transpose(tensor, (0, 3, 1, 2))
        elif len(tensor.shape) == 3:  # HWC -> CHW
            tensor = np.transpose(tensor, (2, 0, 1))
            
        return tensor
        
    raise ValueError(f"Unsupported tensor type: {type(tensor)}")

def get_activation(activation_fn: str) -> tf.keras.layers.Layer:
    """Get activation function by name.
    
    Args:
        activation_fn: Name of activation function
        
    Returns:
        Activation function layer
    """
    if activation_fn == "silu":
        return tf.keras.layers.Activation(tf.nn.silu)
    elif activation_fn == "gelu":
        return tf.keras.layers.Activation(tf.nn.gelu)
    elif activation_fn == "relu":
        return tf.keras.layers.ReLU()
    else:
        raise ValueError(f"Unsupported activation function: {activation_fn}")

def build_position_ids(
    sequence_length: int,
    batch_size: int = 1,
    dtype: tf.DType = tf.int32
) -> tf.Tensor:
    """Build position IDs tensor.
    
    Args:
        sequence_length: Length of sequence
        batch_size: Batch size
        dtype: Data type of output tensor
        
    Returns:
        Position IDs tensor of shape (batch_size, sequence_length)
    """
    position_ids = tf.range(sequence_length, dtype=dtype)
    position_ids = tf.expand_dims(position_ids, axis=0)
    position_ids = tf.repeat(position_ids, batch_size, axis=0)
    return position_ids

def build_causal_attention_mask(
    sequence_length: int,
    dtype: tf.DType = tf.float32
) -> tf.Tensor:
    """Build causal attention mask.
    
    Args:
        sequence_length: Length of sequence
        dtype: Data type of output tensor
        
    Returns:
        Causal attention mask of shape (sequence_length, sequence_length)
    """
    # Create mask of shape (sequence_length, sequence_length)
    mask = 1 - tf.linalg.band_part(
        tf.ones((sequence_length, sequence_length), dtype=dtype), -1, 0
    )
    
    # Set masked positions to -inf
    mask = mask * -1e9
    return mask

def get_shape_list(
    tensor: tf.Tensor,
    expected_rank: Union[int, list] = None
) -> list:
    """Get shape of tensor as list with static or dynamic values.
    
    Args:
        tensor: Input tensor
        expected_rank: Expected rank of tensor
        
    Returns:
        Shape of tensor as list
    """
    if expected_rank is not None:
        assert_rank(tensor, expected_rank)
        
    shape = tensor.shape.as_list()
    
    # Replace None values with dynamic values
    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)
            
    if not non_static_indexes:
        return shape
        
    dynamic_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dynamic_shape[index]
        
    return shape

def assert_rank(tensor: tf.Tensor, expected_rank: Union[int, list]):
    """Raises ValueError if tensor rank does not match expected_rank.
    
    Args:
        tensor: Input tensor
        expected_rank: Expected rank of tensor
    """
    expected_rank_dict = {}
    if isinstance(expected_rank, int):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True
            
    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        raise ValueError(
            f"For tensor {tensor}, expected rank to be in "
            f"{expected_rank_dict.keys()}, but was {actual_rank}"
        )

def get_initializer(
    initializer_range: float = 0.02,
    seed: int = None
) -> tf.keras.initializers.TruncatedNormal:
    """Get weight initializer.
    
    Args:
        initializer_range: Standard deviation for truncated normal initializer
        seed: Random seed
        
    Returns:
        Weight initializer
    """
    return tf.keras.initializers.TruncatedNormal(
        mean=0.0, stddev=initializer_range, seed=seed
    )
