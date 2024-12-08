"""PEFT adapter implementation for TensorFlow."""
import tensorflow as tf
from typing import Optional, Dict, Any, List
import os
import json
from safetensors import safe_open

class PeftAdapterMixin:
    """Mixin class for PEFT adapter support."""
    
    def load_adapter(
        self,
        adapter_path: str,
        adapter_name: Optional[str] = None,
        is_trainable: bool = True
    ):
        """Load LoRA adapter weights.
        
        Args:
            adapter_path: Path to adapter weights
            adapter_name: Name of adapter (optional)
            is_trainable: Whether adapter should be trainable
        """
        # Load adapter config
        config_path = os.path.join(adapter_path, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                adapter_config = json.load(f)
        else:
            raise ValueError(f"No adapter config found at {config_path}")
            
        # Create adapter layers
        self._create_adapter_layers(adapter_config, is_trainable)
        
        # Load adapter weights
        weight_files = [f for f in os.listdir(adapter_path) if f.endswith(".safetensors")]
        if not weight_files:
            raise ValueError(f"No .safetensors files found in {adapter_path}")
            
        # Load and set weights
        for weight_file in weight_files:
            weight_path = os.path.join(adapter_path, weight_file)
            with safe_open(weight_path, framework="pt") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    tf_tensor = convert_torch_to_tf(tensor)
                    
                    # Find corresponding adapter layer and set weights
                    layer_name = key.split(".")[0]
                    if hasattr(self, f"{layer_name}_adapter"):
                        layer = getattr(self, f"{layer_name}_adapter")
                        if isinstance(layer, tf.keras.layers.Layer):
                            layer.set_weights([tf_tensor])
                            
    def _create_adapter_layers(
        self,
        config: Dict[str, Any],
        is_trainable: bool = True
    ):
        """Create adapter layers based on config.
        
        Args:
            config: Adapter configuration
            is_trainable: Whether adapter should be trainable
        """
        r = config.get("r", 8)  # LoRA rank
        alpha = config.get("alpha", 8)  # LoRA alpha for scaling
        dropout = config.get("dropout", 0.0)
        
        # Create adapter for each transformer block
        for i in range(self.num_layers):
            # Query adapter
            setattr(
                self,
                f"transformer_block_{i}_q_adapter",
                LoRALayer(
                    in_features=self.hidden_size,
                    out_features=self.hidden_size,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                    trainable=is_trainable,
                    name=f"transformer_block_{i}_q_adapter"
                )
            )
            
            # Key adapter
            setattr(
                self,
                f"transformer_block_{i}_k_adapter",
                LoRALayer(
                    in_features=self.hidden_size,
                    out_features=self.hidden_size,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                    trainable=is_trainable,
                    name=f"transformer_block_{i}_k_adapter"
                )
            )
            
            # Value adapter
            setattr(
                self,
                f"transformer_block_{i}_v_adapter",
                LoRALayer(
                    in_features=self.hidden_size,
                    out_features=self.hidden_size,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                    trainable=is_trainable,
                    name=f"transformer_block_{i}_v_adapter"
                )
            )
            
            # Output adapter
            setattr(
                self,
                f"transformer_block_{i}_o_adapter",
                LoRALayer(
                    in_features=self.hidden_size,
                    out_features=self.hidden_size,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                    trainable=is_trainable,
                    name=f"transformer_block_{i}_o_adapter"
                )
            )
            
    def get_adapter_state_dict(self) -> Dict[str, tf.Tensor]:
        """Get adapter weights as state dict.
        
        Returns:
            Dictionary of adapter weights
        """
        state_dict = {}
        for name, layer in self.layers:
            if isinstance(layer, LoRALayer):
                state_dict[name] = layer.get_weights()
        return state_dict
        
    def save_adapter(
        self,
        save_path: str,
        adapter_name: Optional[str] = None
    ):
        """Save adapter weights.
        
        Args:
            save_path: Path to save adapter weights
            adapter_name: Name of adapter (optional)
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Save adapter config
        config = {
            "r": 8,  # TODO: Get from layer config
            "alpha": 8,
            "dropout": 0.0,
            "bias": "none"
        }
        
        config_path = os.path.join(save_path, "adapter_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            
        # Save adapter weights
        state_dict = self.get_adapter_state_dict()
        weight_path = os.path.join(save_path, "adapter_model.safetensors")
        save_file(weight_path, state_dict)

class LoRALayer(tf.keras.layers.Layer):
    """LoRA adapter layer implementation."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: int = 8,
        dropout: float = 0.0,
        **kwargs
    ):
        """Initialize LoRA layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            r: LoRA rank
            alpha: LoRA alpha for scaling
            dropout: Dropout rate
        """
        super().__init__(**kwargs)
        
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        self.lora_down = tf.keras.layers.Dense(
            r,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1/r),
            name="lora_down"
        )
        
        self.lora_up = tf.keras.layers.Dense(
            out_features,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.Zeros(),
            name="lora_up"
        )
        
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        # Down project
        down = self.lora_down(x)
        
        # Apply dropout
        down = self.dropout(down, training=training)
        
        # Up project with scaling
        up = self.lora_up(down) * self.scaling
        
        return up
