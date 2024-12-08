"""Text and image processor for OmniGen."""
from __future__ import annotations

import os
from typing import Optional, Union, List, Dict, Any
import tensorflow as tf
from PIL import Image
import numpy as np
from transformers import PreTrainedTokenizer, AutoTokenizer
from huggingface_hub import snapshot_download

class OmniGenProcessor:
    """Processor for text and images in OmniGen."""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        image_size: int = 512,
        **kwargs
    ):
        """Initialize processor."""
        self.tokenizer = tokenizer
        self.image_size = image_size
        
    @classmethod
    def from_pretrained(cls, pretrained_path: str, **kwargs) -> "OmniGenProcessor":
        """Load processor from pretrained path."""
        try:
            # Download if needed
            if not os.path.exists(pretrained_path):
                cache_folder = os.getenv('HF_HUB_CACHE')
                model_path = snapshot_download(
                    repo_id=pretrained_path,
                    cache_dir=cache_folder
                )
            else:
                model_path = pretrained_path
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            return cls(tokenizer=tokenizer, **kwargs)
            
        except Exception as e:
            print(f"Error loading processor: {str(e)}")
            raise
    
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> Dict[str, tf.Tensor]:
        """Process text and optional image inputs."""
        # Handle text inputs
        text_inputs = self.process_text(prompt, negative_prompt)
        
        # Set image dimensions
        height = height or self.image_size
        width = width or self.image_size
        
        return {
            "text_embeddings": text_inputs,
            "height": height,
            "width": width,
            **kwargs
        }
    
    def process_text(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None
    ) -> tf.Tensor:
        """Process text inputs."""
        if isinstance(prompt, str):
            prompt = [prompt]
            
        if negative_prompt is None:
            negative_prompt = [""] * len(prompt)
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
            
        # Tokenize text
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="tf"
        )
        
        # Tokenize negative prompt
        uncond_inputs = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="tf"
        )
        
        # Concatenate for classifier-free guidance
        return tf.concat([uncond_inputs.input_ids, text_inputs.input_ids], axis=0)
    
    def process_image(
        self,
        image: Union[Image.Image, tf.Tensor, np.ndarray],
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> tf.Tensor:
        """Process image input."""
        height = height or self.image_size
        width = width or self.image_size
        
        # Convert to PIL Image if needed
        if isinstance(image, (tf.Tensor, np.ndarray)):
            image = Image.fromarray(image.astype('uint8'))
            
        # Resize image
        image = image.resize((width, height), Image.LANCZOS)
        
        # Convert to tensor
        image = tf.convert_to_tensor(np.array(image))
        image = tf.cast(image, tf.float32) / 127.5 - 1.0
        
        return image
    
    def postprocess_images(
        self,
        images: tf.Tensor,
        output_type: str = "pil"
    ) -> Union[List[Image.Image], np.ndarray]:
        """Post-process generated images."""
        # Denormalize
        images = ((images + 1) * 127.5).numpy().astype(np.uint8)
        
        if output_type == "pil":
            return [Image.fromarray(img) for img in images]
        return images
