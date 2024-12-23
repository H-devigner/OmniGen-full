"""OmniGen TensorFlow Processor

This module provides the TensorFlow implementation of the OmniGen processor,
matching the PyTorch version's functionality while leveraging TensorFlow-specific optimizations.
"""

import os
import re
from typing import Dict, List, Union, Optional
import json

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

class OmniGenProcessor:
    """Processor class for OmniGen model that handles text and image inputs."""
    
    def __init__(self, text_tokenizer, max_image_size: int = 1024):
        self.text_tokenizer = text_tokenizer
        self.max_image_size = max_image_size

        # Image processing pipeline using TensorFlow
        self.image_transform = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda img: crop_arr(img, max_image_size)),
            tf.keras.layers.Lambda(lambda img: tf.cast(img, tf.float32) / 255.0),
            tf.keras.layers.Lambda(lambda img: (img - 0.5) * 2.0)  # Normalize to [-1, 1]
        ])

        self.collator = OmniGenCollator()
        self.separate_collator = OmniGenSeparateCollator()

    @classmethod
    def from_pretrained(cls, model_name):
        if not os.path.exists(model_name):
            cache_folder = os.getenv('HF_HUB_CACHE')
            model_name = snapshot_download(
                repo_id=model_name,
                cache_dir=cache_folder,
                allow_patterns="*.json"
            )
        text_tokenizer = AutoTokenizer.from_pretrained(model_name)
        return cls(text_tokenizer)

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
            model_inputs = self.text_tokenizer(text)
            return {"input_ids": model_inputs.input_ids, "pixel_values": None, "image_sizes": None}

        pattern = r"<\|image_\d+\|>"
        prompt_chunks = [self.text_tokenizer(chunk).input_ids for chunk in re.split(pattern, text)]

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
    def __call__(self, features):
        # Implement collator logic here
        input_ids = [feature['input_ids'] for feature in features]
        pixel_values = [feature['pixel_values'] for feature in features]
        image_sizes = [feature['image_sizes'] for feature in features]

        # Pad input_ids to the longest sequence length
        max_length = max(len(ids) for ids in input_ids)
        padded_input_ids = [ids + [0] * (max_length - len(ids)) for ids in input_ids]

        # Stack pixel_values and image_sizes
        pixel_values = tf.stack(pixel_values, axis=0)
        image_sizes = tf.stack(image_sizes, axis=0)

        return {
            "input_ids": padded_input_ids,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes
        }

class OmniGenSeparateCollator:
    """Separate collator class for OmniGen model."""
    def __call__(self, features):
        # Implement separate collator logic here
        input_ids = [feature['input_ids'] for feature in features]
        pixel_values = [feature['pixel_values'] for feature in features]
        image_sizes = [feature['image_sizes'] for feature in features]

        # Pad input_ids to the longest sequence length
        max_length = max(len(ids) for ids in input_ids)
        padded_input_ids = [ids + [0] * (max_length - len(ids)) for ids in input_ids]

        # Stack pixel_values and image_sizes
        pixel_values = tf.stack(pixel_values, axis=0)
        image_sizes = tf.stack(image_sizes, axis=0)

        return {
            "input_ids": padded_input_ids,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes
        }
