import os
import sys
import unittest
import torch
import tensorflow as tf
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omnigen_tf import OmniGenTF, WeightConverter

class TestWeightConversion(unittest.TestCase):
    def setUp(self):
        self.model_name = "Shitao/OmniGen-v1"
        self.converter = WeightConverter()
        
    def test_download_weights(self):
        """Test downloading weights from HuggingFace."""
        try:
            pytorch_weights, config = self.converter.download_pytorch_weights(self.model_name)
            self.assertIsNotNone(pytorch_weights)
            self.assertIsNotNone(config)
        except Exception as e:
            self.fail(f"Failed to download weights: {str(e)}")
            
    def test_linear_conversion(self):
        """Test conversion of linear layer weights."""
        # Create sample PyTorch weights
        torch_weight = torch.randn(64, 32)
        torch_bias = torch.randn(64)
        
        # Convert weights
        tf_weight, tf_bias = self.converter._convert_linear(torch_weight, torch_bias)
        
        # Check shapes
        self.assertEqual(tf_weight.shape, (32, 64))  # TF uses (in_features, out_features)
        self.assertEqual(tf_bias.shape, (64,))
        
        # Check values
        np.testing.assert_allclose(
            tf_weight,
            torch_weight.transpose(0, 1).numpy(),
            rtol=1e-5
        )
        np.testing.assert_allclose(
            tf_bias,
            torch_bias.numpy(),
            rtol=1e-5
        )
        
    def test_conv2d_conversion(self):
        """Test conversion of Conv2D layer weights."""
        # Create sample PyTorch weights
        torch_weight = torch.randn(64, 32, 3, 3)  # (out_channels, in_channels, height, width)
        torch_bias = torch.randn(64)
        
        # Convert weights
        tf_weight, tf_bias = self.converter._convert_conv2d(torch_weight, torch_bias)
        
        # Check shapes
        self.assertEqual(tf_weight.shape, (3, 3, 32, 64))  # TF uses (height, width, in_channels, out_channels)
        self.assertEqual(tf_bias.shape, (64,))
        
        # Check values
        np.testing.assert_allclose(
            tf_weight,
            torch_weight.permute(2, 3, 1, 0).numpy(),
            rtol=1e-5
        )
        np.testing.assert_allclose(
            tf_bias,
            torch_bias.numpy(),
            rtol=1e-5
        )
        
    def test_batch_norm_conversion(self):
        """Test conversion of BatchNorm layer weights."""
        # Create sample PyTorch weights
        size = 64
        weight = torch.randn(size)
        bias = torch.randn(size)
        running_mean = torch.randn(size)
        running_var = torch.randn(size)
        
        # Convert weights
        tf_weights = self.converter._convert_batch_norm(
            weight, bias, running_mean, running_var
        )
        
        # Check keys and shapes
        self.assertIn('gamma', tf_weights)
        self.assertIn('beta', tf_weights)
        self.assertIn('moving_mean', tf_weights)
        self.assertIn('moving_variance', tf_weights)
        
        for k, v in tf_weights.items():
            self.assertEqual(v.shape, (size,))
            
    def test_layer_norm_conversion(self):
        """Test conversion of LayerNorm weights."""
        # Create sample PyTorch weights
        size = 64
        weight = torch.randn(size)
        bias = torch.randn(size)
        
        # Convert weights
        tf_weights = self.converter._convert_layer_norm(weight, bias)
        
        # Check keys and shapes
        self.assertIn('gamma', tf_weights)
        self.assertIn('beta', tf_weights)
        
        self.assertEqual(tf_weights['gamma'].shape, (size,))
        self.assertEqual(tf_weights['beta'].shape, (size,))
        
    def test_full_model_conversion(self):
        """Test conversion of full model weights."""
        try:
            # Create TF model
            model = OmniGenTF.from_pretrained(self.model_name)
            self.assertIsNotNone(model)
            
            # Try a forward pass
            batch_size = 1
            height, width = 64, 64
            timestep = tf.zeros((batch_size,))
            x = tf.random.normal((batch_size, height, width, 4))
            input_ids = None
            input_img_latents = None
            input_image_sizes = {}
            attention_mask = tf.ones((batch_size, height * width // 4))
            position_ids = tf.range(height * width // 4)[None, :]
            
            inputs = [x, timestep, input_ids, input_img_latents, 
                     input_image_sizes, attention_mask, position_ids]
            
            output = model(inputs, training=False)
            self.assertIsNotNone(output)
            
        except Exception as e:
            self.fail(f"Failed to convert and test full model: {str(e)}")

if __name__ == '__main__':
    unittest.main()
