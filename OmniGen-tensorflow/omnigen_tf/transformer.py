import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from huggingface_hub import snapshot_download

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers import Phi3Config, Phi3Model
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.utils import logging
import tensorflow as tf

logger = logging.get_logger(__name__)


class Phi3Transformer(Phi3Model):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Phi3DecoderLayer`]
    Modified to handle TensorFlow inputs and outputs while using PyTorch internally.
    """
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prefetch_stream = None
        if torch.cuda.is_available():
            self.prefetch_stream = torch.cuda.Stream()

    def _convert_to_pytorch(self, tensor):
        """Convert TensorFlow tensor to PyTorch."""
        if isinstance(tensor, (list, tuple)):
            return [self._convert_to_pytorch(t) for t in tensor]
        if tensor is None:
            return None
        return torch.from_numpy(tensor.numpy()).to(self.device)

    def _convert_to_tensorflow(self, tensor):
        """Convert PyTorch tensor to TensorFlow."""
        if isinstance(tensor, (list, tuple)):
            return [self._convert_to_tensorflow(t) for t in tensor]
        if tensor is None:
            return None
        return tf.convert_to_tensor(tensor.detach().cpu().numpy())

    def prefetch_layer(self, layer_idx: int, device: torch.device):
        """Starts prefetching the next layer cache"""
        if self.prefetch_stream is None:
            return
        with torch.cuda.stream(self.prefetch_stream):
            for name, param in self.layers[layer_idx].named_parameters():
                param.data = param.data.to(device, non_blocking=True)

    def evict_previous_layer(self, layer_idx: int):
        """Moves the previous layer cache to the CPU"""
        prev_layer_idx = layer_idx - 1
        if prev_layer_idx < 0:
            return
        for name, param in self.layers[prev_layer_idx].named_parameters():
            param.data = param.data.to("cpu", non_blocking=True)
            
    def get_offload_layer(self, layer_idx: int, device: torch.device):
        """Manages layer offloading for memory efficiency"""
        torch.cuda.current_stream().synchronize()
        self.evict_previous_layer(layer_idx)
        self.prefetch_layer((layer_idx + 1) % len(self.layers), device)

    def forward(
        self,
        input_ids: tf.Tensor = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_values: Optional[List[tf.Tensor]] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[tf.Tensor] = None,
        offload_model: Optional[bool] = False,
    ) -> Union[Tuple, Dict]:
        """
        Forward pass that handles TensorFlow inputs and outputs.
        """
        # Convert TensorFlow inputs to PyTorch
        pt_inputs = {
            "input_ids": self._convert_to_pytorch(input_ids) if input_ids is not None else None,
            "attention_mask": self._convert_to_pytorch(attention_mask) if attention_mask is not None else None,
            "position_ids": self._convert_to_pytorch(position_ids) if position_ids is not None else None,
            "past_key_values": self._convert_to_pytorch(past_key_values) if past_key_values is not None else None,
            "inputs_embeds": self._convert_to_pytorch(inputs_embeds) if inputs_embeds is not None else None,
            "cache_position": self._convert_to_pytorch(cache_position) if cache_position is not None else None,
        }

        # Run PyTorch forward pass
        outputs = super().forward(
            **pt_inputs,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Convert PyTorch outputs to TensorFlow
        if isinstance(outputs, (tuple, list)):
            return tuple(self._convert_to_tensorflow(o) for o in outputs)
        else:
            return {
                k: self._convert_to_tensorflow(v) 
                for k, v in outputs.items()
            }
