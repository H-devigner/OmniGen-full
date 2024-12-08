import torch
import os
from huggingface_hub import hf_hub_download
import logging
import json
from pathlib import Path
import tensorflow as tf
import onnx
import onnx_tf
from transformers import AutoConfig
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def download_pytorch_model(model_name):
    """Download PyTorch model from HuggingFace"""
    logger.info(f"Downloading PyTorch model from {model_name}")
    
    # Download config
    config_path = hf_hub_download(model_name, filename="config.json")
    with open(config_path) as f:
        config = json.load(f)
    
    # Try to load from safetensors first
    try:
        weights_path = hf_hub_download(model_name, filename="model.safetensors")
        from safetensors.torch import load_file
        state_dict = load_file(weights_path)
        logger.info("Loaded weights from safetensors")
    except:
        weights_path = hf_hub_download(model_name, filename="pytorch_model.bin")
        state_dict = torch.load(weights_path)
        logger.info("Loaded weights from PyTorch checkpoint")
    
    return config, state_dict

def create_pytorch_model(config, state_dict):
    """Create PyTorch model from config and state dict"""
    logger.info("Creating PyTorch model")
    from OmniGen.model import OmniGen
    
    model = OmniGen(config)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def convert_to_onnx(model, output_path):
    """Convert PyTorch model to ONNX format"""
    logger.info("Converting PyTorch model to ONNX")
    
    # Create dummy inputs
    batch_size = 1
    dummy_inputs = {
        'x': torch.randn(batch_size, 4, 64, 64),
        'timestep': torch.zeros(batch_size, dtype=torch.long),
        'input_ids': torch.zeros(batch_size, 77, dtype=torch.long),
        'attention_mask': torch.ones(batch_size, 77, dtype=torch.long),
        'position_ids': torch.arange(77)[None].expand(batch_size, -1)
    }
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_inputs,),
        output_path,
        input_names=['x', 'timestep', 'input_ids', 'attention_mask', 'position_ids'],
        output_names=['output'],
        dynamic_axes={
            'x': {0: 'batch_size'},
            'timestep': {0: 'batch_size'},
            'input_ids': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size'},
            'position_ids': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=12
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verified successfully")
    return output_path

def convert_onnx_to_tensorflow(onnx_path, output_path):
    """Convert ONNX model to TensorFlow format"""
    logger.info("Converting ONNX model to TensorFlow")
    
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    
    # Convert to TensorFlow
    tf_rep = onnx_tf.backend.prepare(onnx_model)
    
    # Export TensorFlow model
    tf_rep.export_graph(output_path)
    logger.info(f"TensorFlow model saved to {output_path}")
    return output_path

def download_and_convert(model_name="Shitao/omnigen-v1", output_dir="./"):
    """
    Download PyTorch model and convert to TensorFlow through ONNX
    
    Args:
        model_name (str): HuggingFace model name
        output_dir (str): Output directory for converted model
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Step 1: Download PyTorch model
        logger.info("Step 1: Downloading PyTorch model")
        config, state_dict = download_pytorch_model(model_name)
        
        # Step 2: Create PyTorch model
        logger.info("Step 2: Creating PyTorch model")
        pytorch_model = create_pytorch_model(config, state_dict)
        
        # Step 3: Convert to ONNX
        logger.info("Step 3: Converting to ONNX")
        onnx_path = output_dir / "model.onnx"
        convert_to_onnx(pytorch_model, str(onnx_path))
        
        # Step 4: Convert ONNX to TensorFlow
        logger.info("Step 4: Converting ONNX to TensorFlow")
        tf_path = output_dir / "tf_model"
        tf_path = convert_onnx_to_tensorflow(str(onnx_path), str(tf_path))
        
        logger.info("Conversion completed successfully")
        return tf_path
        
    except Exception as e:
        logger.error(f"Error during conversion: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting model conversion")
        result = download_and_convert()
        logger.info(f"Conversion completed: {result}")
    except Exception as e:
        logger.error("Conversion failed:", exc_info=True)
        raise
