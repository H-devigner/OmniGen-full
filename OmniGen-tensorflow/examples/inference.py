import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omnigen_tf.pipeline import OmniGenPipeline
from PIL import Image
import argparse

def main():
    parser = argparse.ArgumentParser(description='OmniGen TensorFlow Inference')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for generation')
    parser.add_argument('--output', type=str, default='output.png', help='Output image path')
    parser.add_argument('--height', type=int, default=1024, help='Output image height')
    parser.add_argument('--width', type=int, default=1024, help='Output image width')
    parser.add_argument('--guidance-scale', type=float, default=3.0, help='Text guidance scale')
    parser.add_argument('--steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--input-image', type=str, default=None, help='Optional input image for image-guided generation')
    parser.add_argument('--img-guidance-scale', type=float, default=1.6, help='Image guidance scale')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    print("Initializing OmniGen pipeline...")
    pipeline = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")
    
    # Prepare input image if provided
    input_images = None
    if args.input_image:
        if not os.path.exists(args.input_image):
            raise ValueError(f"Input image not found: {args.input_image}")
        print(f"Loading input image: {args.input_image}")
        input_images = [Image.open(args.input_image).convert('RGB')]
    
    # Generate image
    print(f"Generating image with prompt: {args.prompt}")
    images = pipeline(
        prompt=args.prompt,
        input_images=input_images,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        img_guidance_scale=args.img_guidance_scale if input_images else 0.0,
        num_inference_steps=args.steps,
        seed=args.seed
    )
    
    # Save output
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    images[0].save(args.output)
    print(f"Image saved to: {args.output}")

if __name__ == '__main__':
    main()
