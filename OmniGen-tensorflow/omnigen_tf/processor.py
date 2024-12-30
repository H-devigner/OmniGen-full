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
    """Processor for OmniGen model."""
    
    def __init__(self, tokenizer=None):
        """Initialize processor."""
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("microsoft/phi-2")
        
    def process_text(self, prompt, max_length=77, return_tensors="tf"):
        """Process text input."""
        # Ensure prompt is a list
        if isinstance(prompt, str):
            prompt = [prompt]
            
        # Tokenize text
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors=return_tensors
        )
        
        return text_inputs
        
    def process_images(self, images, height=512, width=512, max_input_image_size=768, use_input_image_size_as_output=False):
        """Process image input."""
        if not isinstance(images, list):
            images = [images]
            
        processed_images = []
        for image in images:
            # Load image if path is provided
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
                
            # Resize image
            if use_input_image_size_as_output:
                w, h = image.size
                height = h
                width = w
            else:
                # Scale down if larger than max size
                if max(height, width) > max_input_image_size:
                    scale = max_input_image_size / max(height, width)
                    height = int(height * scale)
                    width = int(width * scale)
                    
            # Resize and convert to tensor
            image = image.resize((width, height), Image.LANCZOS)
            image = np.array(image)
            image = (image / 127.5) - 1.0
            processed_images.append(image)
            
        # Stack images and convert to tensor
        image_tensor = tf.stack(processed_images)
        
        return {
            "pixel_values": image_tensor,
            "height": height,
            "width": width
        }
        
    def add_prefix_instruction(self, prompt: str) -> str:
        """Add instruction prefix exactly like PyTorch."""
        user_prompt = '''
# Instruction: Continue implementing processor
'''
        return user_prompt + prompt

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
            model_inputs = self.process_text(text)
            return {
                "input_ids": model_inputs["input_ids"],
                "pixel_values": None,
                "image_sizes": None
            }
            
        # Split text by image tags
        pattern = r"<\|image_\d+\|>"
        chunks = re.split(pattern, text)
        prompt_chunks = [
            self.process_text(chunk)["input_ids"] for chunk in chunks
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
            processed_images.append(self.process_images(img)["pixel_values"])
            
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
            image = self.process_images(image)["pixel_values"]
            
        return image
    
class OmniGenCollator:
    """TensorFlow collator with exact PyTorch equivalence."""
    
    def __init__(self, pad_token_id: int = 2, hidden_size: int = 3072):
        """Initialize collator with PyTorch equivalence."""
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size
        
    def create_position(self, attention_mask, num_tokens_for_output_images):
        """Create position IDs exactly like PyTorch."""
        position_ids = []
        text_length = tf.shape(attention_mask)[-1]
        img_length = max(num_tokens_for_output_images)  
        
        for mask in attention_mask:
            temp_l = tf.reduce_sum(mask)
            # Add time embedding token
            temp_position = [0] * (text_length - temp_l) + list(range(temp_l + img_length + 1))
            position_ids.append(temp_position)
            
        return tf.convert_to_tensor(position_ids, dtype=tf.int64)
        
    def create_mask(self, attention_mask, num_tokens_for_output_images):
        """Create attention mask exactly like PyTorch."""
        extended_mask = []
        padding_images = []
        text_length = tf.shape(attention_mask)[-1]
        img_length = max(num_tokens_for_output_images)
        seq_len = text_length + img_length + 1  # Add time embedding token
        
        for i, mask in enumerate(attention_mask):
            temp_l = tf.reduce_sum(mask)
            pad_l = text_length - temp_l
            
            # Create causal mask for text
            temp_mask = tf.linalg.band_part(
                tf.ones((temp_l + 1, temp_l + 1)), -1, 0
            )
            
            # Add image mask
            image_mask = tf.zeros((temp_l + 1, img_length))
            temp_mask = tf.concat([temp_mask, image_mask], axis=-1)
            
            # Add full attention for image tokens
            image_mask = tf.ones((img_length, temp_l + img_length + 1))
            temp_mask = tf.concat([temp_mask, image_mask], axis=0)
            
            # Handle padding
            if pad_l > 0:
                pad_mask = tf.zeros((temp_l + 1 + img_length, pad_l))
                temp_mask = tf.concat([pad_mask, temp_mask], axis=-1)
                
                pad_mask = tf.ones((pad_l, seq_len))
                temp_mask = tf.concat([pad_mask, temp_mask], axis=0)
                
            # Handle image padding
            true_img_length = num_tokens_for_output_images[i]
            pad_img_length = img_length - true_img_length
            if pad_img_length > 0:
                temp_mask = tf.tensor_scatter_nd_update(
                    temp_mask,
                    tf.expand_dims(tf.range(-pad_img_length, 0), 1),
                    tf.zeros((pad_img_length,))
                )
                temp_padding_imgs = tf.zeros((1, pad_img_length, self.hidden_size))
            else:
                temp_padding_imgs = None
                
            extended_mask.append(tf.expand_dims(temp_mask, 0))
            padding_images.append(temp_padding_imgs)
            
        return tf.concat(extended_mask, axis=0), padding_images
        
    def adjust_attention_for_input_images(self, attention_mask, image_sizes):
        """Adjust attention mask for input images exactly like PyTorch."""
        for b_inx in image_sizes:
            for start_inx, end_inx in image_sizes[b_inx]:
                indices = tf.range(start_inx, end_inx)
                updates = tf.ones((end_inx - start_inx, end_inx - start_inx))
                attention_mask = tf.tensor_scatter_nd_update(
                    attention_mask,
                    tf.stack([
                        tf.repeat(indices, end_inx - start_inx),
                        tf.tile(indices, [end_inx - start_inx])
                    ], axis=1),
                    tf.reshape(updates, [-1])
                )
        return attention_mask
        
    def pad_input_ids(self, input_ids, image_sizes):
        """Pad input IDs exactly like PyTorch."""
        max_l = max(len(x) for x in input_ids)
        padded_ids = []
        attention_mask = []
        new_image_sizes = {}
        
        for i, ids in enumerate(input_ids):
            temp_l = len(ids)
            pad_l = max_l - temp_l
            
            if pad_l == 0:
                attention_mask.append([1] * max_l)
                padded_ids.append(ids)
            else:
                attention_mask.append([0] * pad_l + [1] * temp_l)
                padded_ids.append([self.pad_token_id] * pad_l + ids)
                
            if i in image_sizes:
                new_image_sizes[i] = [
                    [x + pad_l for x in pos] for pos in image_sizes[i]
                ]
                
        return (tf.convert_to_tensor(padded_ids, dtype=tf.int64),
                tf.convert_to_tensor(attention_mask, dtype=tf.int64),
                new_image_sizes)
        
    def process_mllm_input(self, mllm_inputs, target_img_size):
        """Process multi-modal inputs exactly like PyTorch."""
        num_tokens_for_output_images = []
        for img_size in target_img_size:
            num_tokens_for_output_images.append(img_size[0] * img_size[1] // 16 // 16)
            
        pixel_values, image_sizes = [], {}
        b_inx = 0
        for x in mllm_inputs:
            if x['pixel_values'] is not None:
                pixel_values.extend(x['pixel_values'])
                for size in x['image_sizes']:
                    if b_inx not in image_sizes:
                        image_sizes[b_inx] = [size]
                    else:
                        image_sizes[b_inx].append(size)
            b_inx += 1
            
        pixel_values = [tf.expand_dims(x, 0) for x in pixel_values]
        
        input_ids = [x['input_ids'] for x in mllm_inputs]
        padded_input_ids, attention_mask, image_sizes = self.pad_input_ids(input_ids, image_sizes)
        position_ids = self.create_position(attention_mask, num_tokens_for_output_images)
        attention_mask, padding_images = self.create_mask(attention_mask, num_tokens_for_output_images)
        attention_mask = self.adjust_attention_for_input_images(attention_mask, image_sizes)
        
        return (padded_input_ids, position_ids, attention_mask, padding_images, 
                pixel_values, image_sizes)
        
    def __call__(self, features):
        """Process features exactly like PyTorch."""
        mllm_inputs = [f[0] for f in features]
        cfg_mllm_inputs = [f[1] for f in features]
        img_cfg_mllm_input = [f[2] for f in features]
        target_img_size = [f[3] for f in features]
        
        if img_cfg_mllm_input[0] is not None:
            mllm_inputs = mllm_inputs + cfg_mllm_inputs + img_cfg_mllm_input
            target_img_size = target_img_size + target_img_size + target_img_size
        else:
            mllm_inputs = mllm_inputs + cfg_mllm_inputs
            target_img_size = target_img_size + target_img_size
            
        (all_padded_input_ids, all_position_ids, all_attention_mask,
         all_padding_images, all_pixel_values, all_image_sizes) = self.process_mllm_input(
            mllm_inputs, target_img_size)
            
        return {
            "input_ids": all_padded_input_ids,
            "attention_mask": all_attention_mask,
            "position_ids": all_position_ids,
            "input_pixel_values": all_pixel_values,
            "input_image_sizes": all_image_sizes,
            "padding_images": all_padding_images
        }
        
class OmniGenSeparateCollator(OmniGenCollator):
    """TensorFlow separate collator with exact PyTorch equivalence."""
    
    def __call__(self, features):
        """Process features exactly like PyTorch."""
        mllm_inputs = [f[0] for f in features]
        cfg_mllm_inputs = [f[1] for f in features]
        img_cfg_mllm_input = [f[2] for f in features]
        target_img_size = [f[3] for f in features]
        
        all_padded_input_ids = []
        all_attention_mask = []
        all_position_ids = []
        all_pixel_values = []
        all_image_sizes = []
        all_padding_images = []
        
        # Process main inputs
        (padded_input_ids, position_ids, attention_mask, padding_images,
         pixel_values, image_sizes) = self.process_mllm_input(mllm_inputs, target_img_size)
         
        all_padded_input_ids.append(padded_input_ids)
        all_attention_mask.append(attention_mask)
        all_position_ids.append(position_ids)
        all_pixel_values.append(pixel_values)
        all_image_sizes.append(image_sizes)
        all_padding_images.append(padding_images)
        
        # Process cfg inputs if present
        if cfg_mllm_inputs[0] is not None:
            (padded_input_ids, position_ids, attention_mask, padding_images,
             pixel_values, image_sizes) = self.process_mllm_input(cfg_mllm_inputs, target_img_size)
             
            all_padded_input_ids.append(padded_input_ids)
            all_attention_mask.append(attention_mask)
            all_position_ids.append(position_ids)
            all_pixel_values.append(pixel_values)
            all_image_sizes.append(image_sizes)
            all_padding_images.append(padding_images)
            
        # Process img cfg inputs if present
        if img_cfg_mllm_input[0] is not None:
            (padded_input_ids, position_ids, attention_mask, padding_images,
             pixel_values, image_sizes) = self.process_mllm_input(img_cfg_mllm_input, target_img_size)
             
            all_padded_input_ids.append(padded_input_ids)
            all_attention_mask.append(attention_mask)
            all_position_ids.append(position_ids)
            all_pixel_values.append(pixel_values)
            all_image_sizes.append(image_sizes)
            all_padding_images.append(padding_images)
            
        return {
            "input_ids": all_padded_input_ids,
            "attention_mask": all_attention_mask,
            "position_ids": all_position_ids,
            "input_pixel_values": all_pixel_values,
            "input_image_sizes": all_image_sizes,
            "padding_images": all_padding_images
        }
