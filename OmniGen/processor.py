import os
import re
from typing import Dict, List
import json

import torch
import numpy as np
import random
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

from OmniGen.utils import (
    create_logger,
    update_ema,
    requires_grad,
    center_crop_arr,
    crop_arr,
)

class OmniGenProcessor:
    """Memory-efficient processor for OmniGen."""
    
    def __init__(self, 
                text_tokenizer, 
                max_image_size: int=1024,
                device=None):
        """Initialize processor."""
        self.text_tokenizer = text_tokenizer
        self.max_image_size = max_image_size
        
        # Set up device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Processor using device: {self.device}")
        
        # Configure GPU settings if available
        if torch.cuda.is_available():
            # Enable TF32 for better performance
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            
            # Configure TensorFlow GPU memory growth
            for device in tf.config.list_physical_devices('GPU'):
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                except:
                    pass
        
        # Enable mixed precision
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        self.image_transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: crop_arr(pil_image, max_image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

        self.collator = OmniGenCollator()
        self.separate_collator = OmniGenSeparateCollator()

    @classmethod
    def from_pretrained(cls, model_name):
        if not os.path.exists(model_name):
            cache_folder = os.getenv('HF_HUB_CACHE')
            model_name = snapshot_download(repo_id=model_name,
                                           cache_dir=cache_folder,
                                           allow_patterns="*.json")
        text_tokenizer = AutoTokenizer.from_pretrained(model_name)

        return cls(text_tokenizer)


    def process_image(self, image):
        image = Image.open(image).convert('RGB')
        return self.image_transform(image)
    
    def process_multi_modal_prompt(self, text, input_images):
        text = self.add_prefix_instruction(text)
        if input_images is None or len(input_images) == 0:
            model_inputs = self.text_tokenizer(text)
            return {"input_ids": model_inputs.input_ids, "pixel_values": None, "image_sizes": None}

        pattern = r"<\|image_\d+\|>"
        prompt_chunks = [self.text_tokenizer(chunk).input_ids for chunk in re.split(pattern, text)] 

        for i in range(1, len(prompt_chunks)):
            if prompt_chunks[i][0] == 1:
                prompt_chunks[i] = prompt_chunks[i][1:]

        image_tags = re.findall(pattern, text) 
        image_ids = [int(s.split("|")[1].split("_")[-1]) for s in image_tags]

        unique_image_ids = sorted(list(set(image_ids)))
        assert unique_image_ids == list(range(1, len(unique_image_ids)+1)), f"image_ids must start from 1, and must be continuous int, e.g. [1, 2, 3], cannot be {unique_image_ids}"
        # total images must be the same as the number of image tags
        assert len(unique_image_ids) == len(input_images), f"total images must be the same as the number of image tags, got {len(unique_image_ids)} image tags and {len(input_images)} images"
        
        input_images = [input_images[x-1] for x in image_ids]

        all_input_ids = []
        img_inx = []
        idx = 0
        for i in range(len(prompt_chunks)):
            all_input_ids.extend(prompt_chunks[i])
            if i != len(prompt_chunks) -1:
                start_inx = len(all_input_ids)
                size = input_images[i].size(-2) *  input_images[i].size(-1) // 16 // 16
                img_inx.append([start_inx, start_inx+size])
                all_input_ids.extend([0]*size)

        return {"input_ids": all_input_ids, "pixel_values": input_images, "image_sizes": img_inx}


    def add_prefix_instruction(self, prompt):
        user_prompt = '
