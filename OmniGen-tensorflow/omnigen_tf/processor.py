import os
import re
import tensorflow as tf
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from typing import Dict, List

class OmniGenProcessor:
    def __init__(self, text_tokenizer, max_image_size: int = 1024):
        self.text_tokenizer = text_tokenizer
        self.max_image_size = max_image_size

        # Image processing pipeline using TensorFlow/Keras
        self.image_transform = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda img: tf.image.resize(img, (max_image_size, max_image_size))),
            tf.keras.layers.Lambda(lambda img: img / 255.0),  # Normalize to [0, 1]
            tf.keras.layers.Lambda(lambda img: img * 2 - 1),  # Normalize to [-1, 1]
        ])

    @classmethod
    def from_pretrained(cls, model_name):
        if not os.path.exists(model_name):
            cache_folder = os.getenv('HF_HUB_CACHE', './')
            model_name = snapshot_download(repo_id=model_name, cache_dir=cache_folder)
        text_tokenizer = AutoTokenizer.from_pretrained(model_name)
        return cls(text_tokenizer)

    def process_image(self, image_path: str):
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        return self.image_transform(image)

    def process_multi_modal_prompt(self, text: str, input_images: List[tf.Tensor]):
        text = self.add_prefix_instruction(text)
        if not input_images:
            model_inputs = self.text_tokenizer(text)
            return {
                "input_ids": model_inputs.input_ids,
                "pixel_values": None,
            }

        # Process images
        input_images = [self.process_image(img) for img in input_images]

        # Tokenize text chunks
        pattern = r"<\|image_\d+\|>"
        prompt_chunks = [
            self.text_tokenizer(chunk).input_ids for chunk in re.split(pattern, text)
        ]
        image_tags = re.findall(pattern, text)
        all_input_ids = []

        for i, chunk in enumerate(prompt_chunks):
            all_input_ids.extend(chunk)
            if i < len(image_tags):
                all_input_ids.extend([0] * input_images[i].numpy().size)

        return {
            "input_ids": all_input_ids,
            "pixel_values": input_images,
        }

    def add_prefix_instruction(self, prompt: str):
        user_prompt = '<|user|>\n'
        generation_prompt = 'Generate an image according to the following instructions\n'
        assistant_prompt = '<|assistant|>\n<|diffusion|>'
        prompt_suffix = "<|end|>\n"
        return f"{user_prompt}{generation_prompt}{prompt}{prompt_suffix}{assistant_prompt}"

    def __call__(
        self, instructions: List[str], input_images: List[List[str]] = None
    ) -> Dict:
        if isinstance(instructions, str):
            instructions = [instructions]
            input_images = [input_images]

        data = []
        for idx, instruction in enumerate(instructions):
            cur_images = input_images[idx] if input_images else []
            processed_data = self.process_multi_modal_prompt(instruction, cur_images)
            data.append(processed_data)
        return data
