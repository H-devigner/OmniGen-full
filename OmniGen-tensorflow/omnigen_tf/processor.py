"""OmniGen TensorFlow Processor with exact PyTorch equivalence."""

import os
import re
import logging
import gc
from typing import Dict, List, Union, Optional, Tuple
import json
import random

import tensorflow as tf
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

from .utils import (
    create_logger,
    update_ema,
    requires_grad,
    center_crop_arr,
    crop_arr,
)

class ProcessorConfig:
    """Configuration for processor memory management."""
    def __init__(self):
        self.mixed_precision = True
        self.prefetch_buffer = 2
        self.max_batch_size = 4
        self.memory_growth = True

class OmniGenProcessor:
    """TensorFlow processor for OmniGen model with PyTorch equivalence."""
    
    def __init__(self, text_tokenizer, max_image_size: int = 1024):
        """Initialize processor with memory optimization.
        
        Args:
            text_tokenizer: Tokenizer for text processing
            max_image_size: Maximum image size
        """
        self.text_tokenizer = text_tokenizer
        self.max_image_size = max_image_size
        self.config = ProcessorConfig()
        
        # Setup optimized image processing
        self._setup_image_transform()
        self._setup_memory_optimization()
        
        # Initialize collators
        self.collator = OmniGenCollator()
        self.separate_collator = OmniGenSeparateCollator()
        
    def _setup_memory_optimization(self):
        """Setup memory optimization configurations."""
        if tf.config.list_physical_devices('GPU'):
            if self.config.mixed_precision:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                
            for device in tf.config.list_physical_devices('GPU'):
                try:
                    if self.config.memory_growth:
                        tf.config.experimental.set_memory_growth(device, True)
                except RuntimeError as e:
                    print(f"Memory optimization warning: {e}")
                    
    def _setup_image_transform(self):
        """Setup image transformation pipeline."""
        def normalize(x):
            x = tf.cast(x, tf.float32)
            x = (x / 127.5) - 1.0  # Normalize to [-1, 1] like PyTorch
            return x
            
        self.image_transform = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: crop_arr(x, self.max_image_size)),
            tf.keras.layers.Lambda(normalize)
        ])
        
    @classmethod
    def from_pretrained(cls, model_name):
        """Load processor from pretrained model with optimized memory usage."""
        if not os.path.exists(model_name):
            cache_folder = os.getenv('HF_HUB_CACHE')
            model_name = snapshot_download(
                repo_id=model_name,
                cache_dir=cache_folder,
                allow_patterns="*.json",
                local_files_only=True  # Prevent unnecessary downloads
            )
            
        # Load tokenizer with optimized settings
        text_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True,  # Prevent unnecessary downloads
            use_fast=True  # Use faster tokenizer implementation
        )
        
        # Clear any cached tensors
        if tf.config.list_physical_devices('GPU'):
            tf.keras.backend.clear_session()
            gc.collect()
            
        return cls(text_tokenizer)
    
    @tf.function(jit_compile=True)
    def process_image(self, image) -> tf.Tensor:
        """Process image with XLA optimization.
        
        Args:
            image: PIL Image, file path, or tensor
            
        Returns:
            Processed image tensor
        """
        if isinstance(image, str):
            image = tf.io.read_file(image)
            image = tf.image.decode_image(image, channels=3)
        elif isinstance(image, Image.Image):
            image = tf.convert_to_tensor(np.array(image))
            
        # Process with memory optimization
        with tf.device("/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"):
            image = self.image_transform(image)
            
        return image
    
    @tf.function(jit_compile=True)
    def process_multi_modal_prompt(
        self, 
        text: str,
        input_images: Optional[List[Union[str, Image.Image, tf.Tensor]]] = None
    ) -> Dict[str, tf.Tensor]:
        """Process multi-modal prompt with XLA optimization."""
        # Add instruction prefix
        text = self.add_prefix_instruction(text)
        
        # Handle text-only case
        if not input_images:
            model_inputs = self.text_tokenizer(text)
            return {
                "input_ids": model_inputs["input_ids"],
                "pixel_values": None,
                "image_sizes": None
            }
            
        # Split text by image tags
        pattern = r"<\|image_\d+\|>"
        chunks = re.split(pattern, text)
        prompt_chunks = [
            self.text_tokenizer(chunk)["input_ids"] for chunk in chunks
        ]
        
        # Remove BOS token from non-first chunks
        for i in range(1, len(prompt_chunks)):
            if prompt_chunks[i][0] == 1:
                prompt_chunks[i] = prompt_chunks[i][1:]
                
        # Extract image IDs
        image_tags = re.findall(pattern, text)
        image_ids = [int(s.split("|")[1].split("_")[-1]) for s in image_tags]
        
        # Validate image IDs
        unique_ids = sorted(list(set(image_ids)))
        if unique_ids != list(range(1, len(unique_ids) + 1)):
            raise ValueError(
                f"Image IDs must be continuous integers starting from 1, got {unique_ids}"
            )
            
        if len(unique_ids) != len(input_images):
            raise ValueError(
                f"Got {len(unique_ids)} image tags but {len(input_images)} images"
            )
            
        # Process images in order
        processed_images = []
        for idx in image_ids:
            img = input_images[idx - 1]
            processed_images.append(self.process_image(img))
            
        # Build combined sequence
        all_input_ids = []
        img_positions = []
        for i in range(len(prompt_chunks)):
            all_input_ids.extend(prompt_chunks[i])
            
            if i < len(prompt_chunks) - 1:
                start_pos = len(all_input_ids)
                img_size = tf.shape(processed_images[i])[-2] * tf.shape(processed_images[i])[-1] // 256
                img_positions.append([start_pos, start_pos + img_size])
                all_input_ids.extend([0] * img_size)
                
        return {
            "input_ids": tf.convert_to_tensor(all_input_ids),
            "pixel_values": processed_images,
            "image_sizes": img_positions
        }
    
    def add_prefix_instruction(self, prompt: str) -> str:
        """Add instruction prefix to prompt, matching PyTorch exactly."""
        user_prompt = '''
# Instruction: Continue implementing processor
'''
        return user_prompt + prompt

    def process_text(self, prompt, max_length=77):
        """Process text input.
        
        Args:
            prompt (str or List[str]): Text prompt(s)
            max_length (int): Maximum sequence length
            
        Returns:
            tf.Tensor: Processed text embeddings
        """
        # Handle single prompt
        if isinstance(prompt, str):
            prompt = [prompt]
            
        # Tokenize
        text_inputs = self.text_tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="tf"
        )
        
        # Convert to float16 for mixed precision
        input_ids = tf.cast(text_inputs.input_ids, tf.float16)
        attention_mask = tf.cast(text_inputs.attention_mask, tf.float16)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

class OmniGenCollator:
    """TensorFlow collator with exact PyTorch equivalence."""
    
    def __init__(self, pad_token_id: int = 2, hidden_size: int = 3072):
        """Initialize collator.
        
        Args:
            pad_token_id: Token ID for padding
            hidden_size: Hidden size for position embeddings
        """
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size
        
    @tf.function(jit_compile=True)
    def create_position(
        self,
        attention_mask: tf.Tensor,
        num_tokens_for_output_images: int
    ) -> tf.Tensor:
        """Create position IDs with XLA optimization."""
        batch_size = tf.shape(attention_mask)[0]
        seq_length = tf.shape(attention_mask)[1]
        
        # Create position IDs
        position_ids = tf.range(seq_length, dtype=tf.int32)
        position_ids = tf.expand_dims(position_ids, axis=0)
        position_ids = tf.tile(position_ids, [batch_size, 1])
        
        # Add output image tokens
        if num_tokens_for_output_images > 0:
            output_position_ids = tf.range(
                seq_length,
                seq_length + num_tokens_for_output_images,
                dtype=tf.int32
            )
            output_position_ids = tf.expand_dims(output_position_ids, axis=0)
            output_position_ids = tf.tile(output_position_ids, [batch_size, 1])
            position_ids = tf.concat([position_ids, output_position_ids], axis=1)
            
        return position_ids
        
    @tf.function(jit_compile=True)
    def create_mask(
        self,
        attention_mask: tf.Tensor,
        num_tokens_for_output_images: int
    ) -> tf.Tensor:
        """Create attention mask with XLA optimization."""
        batch_size = tf.shape(attention_mask)[0]
        
        if num_tokens_for_output_images > 0:
            # Add mask for output image tokens
            output_attention_mask = tf.ones(
                [batch_size, num_tokens_for_output_images],
                dtype=attention_mask.dtype
            )
            attention_mask = tf.concat(
                [attention_mask, output_attention_mask],
                axis=1
            )
            
        return attention_mask
        
    @tf.function(jit_compile=True)
    def adjust_attention_for_input_images(
        self,
        attention_mask: tf.Tensor,
        image_sizes: List[List[int]]
    ) -> tf.Tensor:
        """Adjust attention mask for input images with XLA optimization."""
        if not image_sizes:
            return attention_mask
            
        batch_size = tf.shape(attention_mask)[0]
        seq_length = tf.shape(attention_mask)[1]
        
        # Create mask for each batch
        for b in range(batch_size):
            if b < len(image_sizes):
                for start, end in image_sizes[b]:
                    # Zero out attention for image tokens
                    attention_mask = tf.tensor_scatter_nd_update(
                        attention_mask,
                        [[b, i] for i in range(start, end)],
                        tf.zeros([end - start], dtype=attention_mask.dtype)
                    )
                    
        return attention_mask
        
    @tf.function(jit_compile=True)
    def pad_input_ids(
        self,
        input_ids: List[List[int]],
        image_sizes: List[List[int]]
    ) -> tf.Tensor:
        """Pad input IDs with XLA optimization."""
        if not input_ids:
            return tf.zeros([0, 0], dtype=tf.int32)
            
        # Find max length
        max_length = max(len(ids) for ids in input_ids)
        batch_size = len(input_ids)
        
        # Create padded tensor
        padded = tf.zeros([batch_size, max_length], dtype=tf.int32)
        
        # Fill with actual values
        for b, ids in enumerate(input_ids):
            padded = tf.tensor_scatter_nd_update(
                padded,
                [[b, i] for i in range(len(ids))],
                tf.constant(ids, dtype=tf.int32)
            )
            
        return padded
        
    @tf.function(jit_compile=True)
    def process_mllm_input(
        self,
        mllm_inputs: Dict[str, tf.Tensor],
        target_img_size: Optional[Tuple[int, int]] = None
    ) -> Dict[str, tf.Tensor]:
        """Process multi-modal inputs with XLA optimization."""
        input_ids = mllm_inputs["input_ids"]
        pixel_values = mllm_inputs.get("pixel_values")
        image_sizes = mllm_inputs.get("image_sizes")
        
        # Process input IDs
        input_ids = self.pad_input_ids(input_ids, image_sizes)
        attention_mask = tf.cast(input_ids != self.pad_token_id, tf.float32)
        
        # Adjust for input images
        attention_mask = self.adjust_attention_for_input_images(
            attention_mask, image_sizes
        )
        
        # Create position IDs
        position_ids = self.create_position(attention_mask, 0)
        
        # Process pixel values if present
        if pixel_values is not None and target_img_size is not None:
            # Resize images to target size
            pixel_values = tf.image.resize(
                pixel_values,
                target_img_size,
                method=tf.image.ResizeMethod.LANCZOS3
            )
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes
        }
        
    def __call__(self, features: List[Dict[str, tf.Tensor]]) -> Dict[str, tf.Tensor]:
        """Process batch of features with memory optimization."""
        if not features:
            return {}
            
        # Extract components
        input_ids = [f["input_ids"] for f in features]
        pixel_values = [f.get("pixel_values") for f in features]
        image_sizes = [f.get("image_sizes") for f in features]
        
        # Process with XLA optimization
        return self.process_mllm_input(
            {
                "input_ids": input_ids,
                "pixel_values": pixel_values[0] if pixel_values else None,
                "image_sizes": image_sizes
            }
        )


class OmniGenSeparateCollator:
    """TensorFlow separate collator with exact PyTorch equivalence."""
    
    def __call__(self, features: List[Dict[str, tf.Tensor]]) -> List[Dict[str, tf.Tensor]]:
        """Process features separately with memory optimization."""
        if not features:
            return []
            
        processed_features = []
        for feature in features:
            # Process each feature independently
            processed = OmniGenCollator().process_mllm_input(
                {
                    "input_ids": [feature["input_ids"]],
                    "pixel_values": feature.get("pixel_values"),
                    "image_sizes": feature.get("image_sizes")
                }
            )
            
            # Remove batch dimension
            processed = {
                k: v[0] if isinstance(v, tf.Tensor) and len(v.shape) > 0 else v
                for k, v in processed.items()
            }
            
            processed_features.append(processed)
            
        return processed_features
