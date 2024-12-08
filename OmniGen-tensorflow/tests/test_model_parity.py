"""Tests to verify TensorFlow implementation matches PyTorch."""
import os
import sys
import pytest
import numpy as np
import tensorflow as tf
import torch

# Add parent directories to path to import OmniGen modules
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tf_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(tf_parent_dir)

from omnigen_tf.model import (
    OmniGenTF, 
    TimestepEmbedder as TFTimestepEmbedder,
    FinalLayer as TFFinalLayer,
    PatchEmbedMR as TFPatchEmbedMR
)
from OmniGen.model import (
    OmniGen,
    TimestepEmbedder as PTTimestepEmbedder,
    FinalLayer as PTFinalLayer,
    PatchEmbedMR as PTPatchEmbedMR
)

def convert_torch_to_tf(tensor):
    """Convert PyTorch tensor to TensorFlow tensor."""
    if isinstance(tensor, torch.Tensor):
        return tf.convert_to_tensor(tensor.detach().cpu().numpy())
    return tf.convert_to_tensor(tensor)

def convert_tf_to_torch(tensor):
    """Convert TensorFlow tensor to PyTorch tensor."""
    if isinstance(tensor, tf.Tensor):
        return torch.from_numpy(tensor.numpy())
    return torch.from_numpy(tensor)

class TestModelParity:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test configurations."""
        self.hidden_size = 768
        self.batch_size = 2
        self.image_size = 64
        self.in_channels = 4
        self.patch_size = 2
        
        # Common config for both implementations
        self.config = {
            'hidden_size': self.hidden_size,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': self.hidden_size * 4
        }
        
        # Random seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        tf.random.set_seed(42)
        
    def test_timestep_embedder(self):
        """Test TimestepEmbedder parity."""
        # Create models
        pt_embedder = PTTimestepEmbedder(self.hidden_size)
        tf_embedder = TFTimestepEmbedder(self.hidden_size)
        
        # Build TensorFlow layer
        dummy_input = tf.random.uniform((1,), dtype=tf.float32)
        _ = tf_embedder(dummy_input)
        
        # Copy weights from PyTorch to TensorFlow
        tf_embedder.mlp.layers[0].kernel.assign(
            convert_torch_to_tf(pt_embedder.mlp[0].weight.t().float())
        )
        tf_embedder.mlp.layers[0].bias.assign(
            convert_torch_to_tf(pt_embedder.mlp[0].bias.float())
        )
        tf_embedder.mlp.layers[2].kernel.assign(
            convert_torch_to_tf(pt_embedder.mlp[2].weight.t().float())
        )
        tf_embedder.mlp.layers[2].bias.assign(
            convert_torch_to_tf(pt_embedder.mlp[2].bias.float())
        )
        
        # Test input
        timesteps = np.array([1, 10, 100, 1000], dtype=np.float32)
        pt_timesteps = torch.from_numpy(timesteps)
        tf_timesteps = tf.convert_to_tensor(timesteps)
        
        # Get outputs
        pt_output = pt_embedder(pt_timesteps)
        tf_output = tf_embedder(tf_timesteps)
        
        # Compare outputs
        np.testing.assert_allclose(
            pt_output.detach().numpy(),
            tf_output.numpy(),
            rtol=1e-5,
            atol=1e-5
        )
        
    def test_patch_embed(self):
        """Test PatchEmbedMR parity."""
        # Create models
        pt_embed = PTPatchEmbedMR(
            patch_size=self.patch_size,
            in_chans=self.in_channels,
            embed_dim=self.hidden_size
        )
        tf_embed = TFPatchEmbedMR(
            patch_size=self.patch_size,
            in_chans=self.in_channels,
            embed_dim=self.hidden_size
        )
        
        # Build TensorFlow layer
        dummy_input = tf.random.uniform((1, self.image_size, self.image_size, self.in_channels), dtype=tf.float32)
        _ = tf_embed(dummy_input)
        
        # Copy weights
        # PyTorch: [out_channels, in_channels, kernel_size, kernel_size]
        # TensorFlow: [kernel_size, kernel_size, in_channels, out_channels]
        pt_weight = pt_embed.proj.weight.detach().float().numpy()
        tf_weight = np.transpose(pt_weight, (2, 3, 1, 0))
        tf_embed.proj.kernel.assign(tf_weight)
        tf_embed.proj.bias.assign(convert_torch_to_tf(pt_embed.proj.bias.float()))
        
        # Test input
        x = np.random.randn(self.batch_size, self.image_size, self.image_size, self.in_channels).astype(np.float32)
        pt_x = torch.from_numpy(x).permute(0, 3, 1, 2)  # PyTorch uses channels first
        tf_x = tf.convert_to_tensor(x)  # TensorFlow uses channels last
        
        # Get outputs
        pt_output = pt_embed(pt_x)
        tf_output = tf_embed(tf_x)
        
        # Compare outputs (convert PyTorch output to channels last)
        pt_output = pt_output.permute(0, 2, 3, 1).detach().numpy()
        np.testing.assert_allclose(
            pt_output,
            tf_output.numpy(),
            rtol=1e-5,
            atol=1e-5
        )
        
    def test_final_layer(self):
        """Test FinalLayer parity."""
        # Create models
        pt_final = PTFinalLayer(self.hidden_size, self.patch_size, self.in_channels)
        tf_final = TFFinalLayer(self.hidden_size, self.patch_size, self.in_channels)
        
        # Build TensorFlow layers
        dummy_x = tf.random.uniform((1, 256, self.hidden_size), dtype=tf.float32)
        dummy_c = tf.random.uniform((1, self.hidden_size), dtype=tf.float32)
        _ = tf_final(dummy_x, dummy_c)
        
        # Copy weights
        # AdaLN modulation
        tf_final.adaLN_modulation.layers[1].kernel.assign(
            convert_torch_to_tf(pt_final.adaLN_modulation[1].weight.t().float())
        )
        tf_final.adaLN_modulation.layers[1].bias.assign(
            convert_torch_to_tf(pt_final.adaLN_modulation[1].bias.float())
        )
        
        # Linear layer
        tf_final.linear.kernel.assign(convert_torch_to_tf(pt_final.linear.weight.t().float()))
        tf_final.linear.bias.assign(convert_torch_to_tf(pt_final.linear.bias.float()))
        
        # Test inputs
        x = np.random.randn(self.batch_size, 256, self.hidden_size).astype(np.float32)
        c = np.random.randn(self.batch_size, self.hidden_size).astype(np.float32)
        
        pt_x = torch.from_numpy(x)
        pt_c = torch.from_numpy(c)
        tf_x = tf.convert_to_tensor(x)
        tf_c = tf.convert_to_tensor(c)
        
        # Get outputs
        pt_output = pt_final(pt_x, pt_c)
        tf_output = tf_final(tf_x, tf_c)
        
        # Compare outputs
        np.testing.assert_allclose(
            pt_output.detach().numpy(),
            tf_output.numpy(),
            rtol=1e-5,
            atol=1e-5
        )
        
    def test_full_forward_pass(self):
        """Test full model forward pass parity."""
        # Create models
        from transformers import Phi3Config
        config = Phi3Config(**self.config)
        pt_model = OmniGen(config)
        tf_model = OmniGenTF(self.config)
        
        # Test inputs
        x = np.random.randn(self.batch_size, self.image_size, self.image_size, self.in_channels).astype(np.float32)
        timesteps = np.array([100, 500], dtype=np.float32)
        
        # Convert inputs
        pt_x = torch.from_numpy(x).permute(0, 3, 1, 2)
        pt_timesteps = torch.from_numpy(timesteps)
        tf_x = tf.convert_to_tensor(x)
        tf_timesteps = tf.convert_to_tensor(timesteps)
        
        # Get outputs
        with torch.no_grad():
            pt_output = pt_model(pt_x, pt_timesteps)
        tf_output = tf_model(tf_x, tf_timesteps)
        
        # Convert PyTorch output to channels last and compare
        pt_output = pt_output.permute(0, 2, 3, 1).detach().numpy()
        np.testing.assert_allclose(
            pt_output,
            tf_output.numpy(),
            rtol=1e-4,
            atol=1e-4
        )

if __name__ == '__main__':
    pytest.main([__file__])
