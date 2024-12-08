import torch
from OmniGen.pipeline import OmniGenPipeline
import matplotlib.pyplot as plt

def generate_image(prompt: str, output_path: str = "generated_image.png", show_image: bool = True):
    """
    Generate an image from a text prompt using OmniGen and optionally display it.
    
    Args:
        prompt (str): The text prompt describing the image to generate
        output_path (str): Path where to save the generated image
        show_image (bool): Whether to display the image using matplotlib
    """
    print(f"Generating image for prompt: '{prompt}'...")
    
    # Initialize the pipeline
    pipeline = OmniGenPipeline.from_pretrained("Salesforce/omnigen")
    
    # Generate the image
    # Using minimal arguments with smaller dimensions
    images = pipeline(
        prompt=prompt,
        height=256,   # Reduced height for faster generation
        width=256,    # Reduced width for faster generation
        num_inference_steps=30,  # Reduced number of steps
        guidance_scale=3.0,      # Default guidance scale
    )
    
    # Get the first image if we got a list
    image = images[0] if isinstance(images, list) else images
    
    # Save the generated image
    image.save(output_path)
    print(f"Image saved to: {output_path}")
    
    # Display the image if requested
    if show_image:
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis('off')  # Hide axes
        plt.title(f"Generated Image\nPrompt: {prompt}")
        plt.show()

if __name__ == "__main__":
    # Example usage
    prompt = "A beautiful sunset over a mountain landscape, digital art style"
    generate_image(prompt)
