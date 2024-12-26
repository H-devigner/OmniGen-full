import tensorflow as tf
from PIL import Image
import numpy as np
from omnigen_tf.pipeline import OmniGenPipeline

def generate_image(prompt, output_path=None, show_image=False):
    """Generate an image from a text prompt.
    
    Args:
        prompt (str): Text prompt to generate image from
        output_path (str, optional): Path to save generated image
        show_image (bool): Whether to display the image
        
    Returns:
        PIL.Image: Generated image
    """
    # Initialize pipeline
    pipeline = OmniGenPipeline.from_pretrained("OmniGen")
    
    # Generate image
    image = pipeline(
        prompt=prompt,
        height=128,  # Reduced height for faster generation
        width=128,   # Reduced width for faster generation
        num_inference_steps=50,
        guidance_scale=7.5
    )
    
    # Save image if output path provided
    if output_path:
        image.save(output_path)
        
    # Show image if requested
    if show_image:
        image.show()
        
    return image

if __name__ == "__main__":
    # Example usage
    prompt = "a beautiful mountain landscape at sunset"
    image = generate_image(
        prompt=prompt,
        output_path="mountain.png",  # Optional
        show_image=True  # Optional
    )
