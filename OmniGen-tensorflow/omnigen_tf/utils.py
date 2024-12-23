"""Utility functions for OmniGen TensorFlow implementation.

This module provides utility functions for the TensorFlow implementation of OmniGen,
including both converted PyTorch utilities and TensorFlow-specific utilities.
"""

import os
import logging
from typing import Union, Dict, Any, Optional, List

import tensorflow as tf
import numpy as np
from PIL import Image

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

def create_logger(logging_dir: str) -> logging.Logger:
    """Create a logger that writes to a log file and stdout.
    
    Args:
        logging_dir: Directory to write log file to
        
    Returns:
        Logger instance
    """
    os.makedirs(logging_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger

@tf.function
def update_ema(ema_model: tf.keras.Model, model: tf.keras.Model, decay: float = 0.9999) -> None:
    """Update exponential moving average (EMA) of model weights.
    
    Args:
        ema_model: EMA model
        model: Current model
        decay: EMA decay rate
    """
    for ema_var, model_var in zip(ema_model.trainable_variables, model.trainable_variables):
        ema_var.assign(decay * ema_var + (1 - decay) * model_var)

def requires_grad(model: tf.keras.Model, flag: bool = True) -> None:
    """Set trainable flag for all parameters in a model.
    
    Args:
        model: Model to update
        flag: Whether parameters should be trainable
    """
    model.trainable = flag

def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """Center crop an image to specified size.
    
    Implementation from ADM:
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    
    Args:
        pil_image: Input PIL image
        image_size: Target size
        
    Returns:
        Center-cropped PIL image
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def crop_arr(pil_image: Image.Image, max_image_size: int) -> Image.Image:
    """Crop and resize image to fit within maximum size while maintaining aspect ratio.
    
    Args:
        pil_image: Input PIL image
        max_image_size: Maximum size in pixels
        
    Returns:
        Cropped and resized PIL image
    """
    while min(*pil_image.size) >= 2 * max_image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    if max(*pil_image.size) > max_image_size:
        scale = max_image_size / max(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
    
    if min(*pil_image.size) < 16:
        scale = 16 / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
    
    arr = np.array(pil_image)
    crop_y1 = (arr.shape[0] % 16) // 2
    crop_y2 = arr.shape[0] % 16 - crop_y1
    crop_x1 = (arr.shape[1] % 16) // 2
    crop_x2 = arr.shape[1] % 16 - crop_x1
    
    arr = arr[crop_y1:arr.shape[0]-crop_y2, crop_x1:arr.shape[1]-crop_x2]    
    return Image.fromarray(arr)

def vae_encode(vae: tf.keras.Model, x: tf.Tensor, dtype: tf.dtypes.DType) -> tf.Tensor:
    """Encode images using VAE.
    
    Args:
        vae: VAE model
        x: Input tensor
        dtype: Output data type
        
    Returns:
        Encoded tensor
    """
    if x is not None:
        if hasattr(vae.config, 'shift_factor'):
            x = vae.encode(x).sample()
            x = (x - vae.config.shift_factor) * vae.config.scaling_factor
        else:
            x = vae.encode(x).sample() * vae.config.scaling_factor
        return tf.cast(x, dtype)
    return None

def vae_encode_list(
    vae: tf.keras.Model, 
    x: List[tf.Tensor], 
    dtype: tf.dtypes.DType
) -> List[tf.Tensor]:
    """Encode a list of images using VAE.
    
    Args:
        vae: VAE model
        x: List of input tensors
        dtype: Output data type
        
    Returns:
        List of encoded tensors
    """
    return [vae_encode(vae, img, dtype) for img in x]

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
    dtype: tf.dtypes.DType = tf.int32
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
    dtype: tf.dtypes.DType = tf.float32
) -> tf.Tensor:
    """Build causal attention mask.
    
    Args:
        sequence_length: Length of sequence
        dtype: Data type of output tensor
        
    Returns:
        Causal attention mask tensor
    """
    # Create mask of shape [1, 1, sequence_length, sequence_length]
    mask = 1 - tf.linalg.band_part(tf.ones((sequence_length, sequence_length)), -1, 0)
    mask = tf.cast(mask, dtype)
    
    # Add batch and head dimensions
    mask = tf.expand_dims(tf.expand_dims(mask, 0), 0)
    
    return mask

def get_shape_list(tensor: tf.Tensor, expected_rank: Optional[Union[int, List[int]]] = None) -> List[int]:
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
    
    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)
            
    if not non_static_indexes:
        return shape
        
    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
        
    return shape

def assert_rank(tensor: tf.Tensor, expected_rank: Union[int, List[int]]) -> None:
    """Raises ValueError if tensor rank does not match expected_rank.
    
    Args:
        tensor: Input tensor
        expected_rank: Expected rank of tensor
    """
    if isinstance(expected_rank, int):
        expected_rank = [expected_rank]
        
    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank:
        raise ValueError(
            f"For tensor {tensor}, expected rank to be one of {expected_rank} "
            f"but was actually {actual_rank}"
        )

def get_initializer(
    initializer_range: float = 0.02,
    seed: Optional[int] = None
) -> tf.keras.initializers.Initializer:
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
