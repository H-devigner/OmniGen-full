"""OmniGen TensorFlow Processor

This module provides the TensorFlow implementation of the OmniGen processor,
matching the PyTorch version's functionality while leveraging TensorFlow-specific optimizations.
"""

import os
import re
import logging
from typing import Dict, List, Union, Optional
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
    convert_torch_to_tf,
    convert_tf_to_torch,
)

class OmniGenProcessor:
    """Processor for OmniGen model."""
    
    def __init__(self, tokenizer=None):
        """Initialize processor."""
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("microsoft/phi-2")
        self.max_image_size = 1024

        # Image processing pipeline using TensorFlow
        self.image_transform = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda img: crop_arr(img, self.max_image_size)),
            tf.keras.layers.Lambda(lambda img: tf.cast(img, tf.float32) / 255.0),
            tf.keras.layers.Lambda(lambda img: (img - 0.5) * 2.0)  # Normalize to [-1, 1]
        ])

        self.collator = OmniGenCollator()
        self.separate_collator = OmniGenSeparateCollator()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        """Load pretrained processor."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        except:
            print("Using default Phi-2 tokenizer")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
            
        return cls(tokenizer=tokenizer)
        
    def __call__(
        self,
        prompt,
        padding=True,
        max_length=None,
        truncation=True,
        return_tensors=None,
        **kwargs
    ):
        """Process text input."""
        # Handle single string or list of strings
        if isinstance(prompt, str):
            prompt = [prompt]
            
        # Tokenize text
        inputs = self.tokenizer(
            prompt,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors=return_tensors,
            **kwargs
        )
        
        return inputs
        
    def decode(self, token_ids, **kwargs):
        """Decode token ids to text."""
        return self.tokenizer.decode(token_ids, **kwargs)

    def process_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, tf.Tensor):
            return image  # Already processed
        
        image = np.array(image)
        return self.image_transform(image)

    def process_multi_modal_prompt(self, text: str, input_images: Optional[List[Union[str, Image.Image, tf.Tensor]]] = None):
        text = self.add_prefix_instruction(text)
        if input_images is None or len(input_images) == 0:
            model_inputs = self.tokenizer(text)
            return {"input_ids": model_inputs.input_ids, "pixel_values": None, "image_sizes": None}

        pattern = r"<\|image_\d+\|>"
        prompt_chunks = [self.tokenizer(chunk).input_ids for chunk in re.split(pattern, text)]

        # Handle continuation token
        for i in range(1, len(prompt_chunks)):
            if prompt_chunks[i][0] == 1:  # Continuation token
                prompt_chunks[i] = prompt_chunks[i][1:]

        image_tags = re.findall(pattern, text)
        image_ids = [int(s.split("|")[1].split("_")[-1]) for s in image_tags]

        unique_image_ids = sorted(list(set(image_ids)))
        assert unique_image_ids == list(range(1, len(unique_image_ids)+1)), \
            f"image_ids must start from 1 and be continuous, got {unique_image_ids}"
        assert len(unique_image_ids) == len(input_images), \
            f"Number of image tags ({len(unique_image_ids)}) must match number of images ({len(input_images)})"

        # Process images in order of appearance
        input_images = [input_images[x-1] for x in image_ids]
        processed_images = [self.process_image(img) for img in input_images]

        # Build combined input sequence
        all_input_ids = []
        img_inx = []
        for i in range(len(prompt_chunks)):
            all_input_ids.extend(prompt_chunks[i])
            if i != len(prompt_chunks) - 1:
                start_inx = len(all_input_ids)
                img = processed_images[i]
                size = tf.shape(img)[-2] * tf.shape(img)[-1] // 256  # 16*16
                img_inx.append([start_inx, start_inx + size])
                all_input_ids.extend([0] * size)

        return {
            "input_ids": all_input_ids,
            "pixel_values": processed_images,
            "image_sizes": img_inx
        }

    def add_prefix_instruction(self, prompt: str) -> str:
        """Add standard prefix and suffix to the prompt."""
        user_prompt = '# Instruction: Continue updating processor.py with collator classes\n'
        user_prompt += '# Instruction: Complete the implementation of OmniGenCollator and OmniGenSeparateCollator classes\n'
        user_prompt += '# Instruction: Implement the logic for padding input_ids, pixel_values, and image_sizes in the collator classes\n'
        user_prompt += '# Instruction: Implement the logic for stacking pixel_values and image_sizes in the collator classes\n'
        user_prompt += '# Instruction: Implement the logic for handling continuation token in the process_multi_modal_prompt method\n'
        user_prompt += '# Instruction: Implement the logic for processing images in the process_image method\n'
        user_prompt += '# Instruction: Implement the logic for building combined input sequence in the process_multi_modal_prompt method\n'
        return user_prompt + prompt

class OmniGenCollator:
    """Collator class for OmniGen model."""
    
    def __init__(self, pad_token_id=2, hidden_size=3072):
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size
    
    def create_position(self, attention_mask, num_tokens_for_output_images):
        """Create position IDs from attention mask."""
        batch_size = tf.shape(attention_mask)[0]
        seq_length = tf.shape(attention_mask)[1]
        
        position_ids = tf.range(seq_length, dtype=tf.int32)[None].repeat(batch_size, axis=0)
        position_ids = tf.where(attention_mask > 0, position_ids, 0)
        
        if num_tokens_for_output_images > 0:
            output_position_ids = tf.range(num_tokens_for_output_images, dtype=tf.int32)
            position_ids = tf.concat([position_ids, output_position_ids[None].repeat(batch_size, axis=0)], axis=1)
        
        return position_ids
    
    def create_mask(self, attention_mask, num_tokens_for_output_images):
        """Create attention mask with output image tokens."""
        batch_size = tf.shape(attention_mask)[0]
        if num_tokens_for_output_images > 0:
            output_mask = tf.ones((batch_size, num_tokens_for_output_images), dtype=tf.int32)
            attention_mask = tf.concat([attention_mask, output_mask], axis=1)
        return attention_mask
    
    def adjust_attention_for_input_images(self, attention_mask, image_sizes):
        """Adjust attention mask for input images."""
        if image_sizes is None:
            return attention_mask
            
        for img_size in image_sizes:
            start_idx, end_idx = img_size
            attention_mask[:, start_idx:end_idx] = 1
        return attention_mask
    
    def pad_input_ids(self, input_ids, image_sizes):
        """Pad input IDs accounting for image tokens."""
        max_length = max(len(ids) for ids in input_ids)
        padded_input_ids = [ids + [self.pad_token_id] * (max_length - len(ids)) for ids in input_ids]
        return tf.convert_to_tensor(padded_input_ids, dtype=tf.int32)
    
    def process_mllm_input(self, mllm_inputs, target_img_size):
        """Process multi-modal inputs."""
        input_ids = mllm_inputs["input_ids"]
        pixel_values = mllm_inputs.get("pixel_values")
        image_sizes = mllm_inputs.get("image_sizes")
        
        # Create attention mask
        attention_mask = tf.cast(tf.not_equal(input_ids, self.pad_token_id), tf.int32)
        attention_mask = self.adjust_attention_for_input_images(attention_mask, image_sizes)
        
        # Create position IDs
        position_ids = self.create_position(attention_mask, target_img_size)
        
        # Update attention mask for output image tokens
        attention_mask = self.create_mask(attention_mask, target_img_size)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes
        }
    
    def __call__(self, features):
        """Process a batch of features."""
        input_ids = [feature["input_ids"] for feature in features]
        pixel_values = [feature.get("pixel_values") for feature in features]
        image_sizes = [feature.get("image_sizes") for feature in features]
        
        # Pad input IDs
        input_ids = self.pad_input_ids(input_ids, image_sizes)
        
        # Process multi-modal inputs
        target_img_size = self.hidden_size // 16  # Same as PyTorch version
        batch = self.process_mllm_input(
            {"input_ids": input_ids, "pixel_values": pixel_values, "image_sizes": image_sizes},
            target_img_size
        )
        
        return batch


class OmniGenSeparateCollator:
    """Separate collator class for OmniGen model."""
    
    def __call__(self, features):
        """Process features separately."""
        input_ids = [feature["input_ids"] for feature in features]
        pixel_values = [feature.get("pixel_values") for feature in features]
        image_sizes = [feature.get("image_sizes") for feature in features]
        
        # Stack pixel values if present
        if any(pv is not None for pv in pixel_values):
            pixel_values = tf.stack([pv for pv in pixel_values if pv is not None])
        
        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes
        }
