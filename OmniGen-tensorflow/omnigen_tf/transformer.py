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

logger = logging.get_logger(__name__)


class Phi3Transformer(PreTrainedModel):
    """Memory-efficient Phi3 transformer implementation."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Set up GPU and memory config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        # Initialize components and move to GPU
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size).to(self.device)
        self.layers = nn.ModuleList([Phi3Layer(config) for _ in range(config.num_hidden_layers)]).to(self.device)
        self.final_layernorm = nn.LayerNorm(config.hidden_size).to(self.device)
        
        # Enable gradient checkpointing if available
        self.gradient_checkpointing = False
        self._use_flash_attention_2 = config.use_flash_attention_2 if hasattr(config, 'use_flash_attention_2') else False
        
        # Initialize weights
        self.post_init()
        
        # Move entire model to GPU
        self.to(self.device)
        
    def prefetch_layer(self, layer_idx: int, device: torch.device):
        "Starts prefetching the next layer cache"
        with torch.cuda.stream(self.prefetch_stream):
            # Prefetch next layer tensors to GPU
            for name, param in self.layers[layer_idx].named_parameters():
                param.data = param.data.to(device, non_blocking=True)

    def evict_previous_layer(self, layer_idx: int):
        "Moves the previous layer cache to the CPU"
        prev_layer_idx = layer_idx - 1
        for name, param in self.layers[prev_layer_idx].named_parameters():
            param.data = param.data.to("cpu", non_blocking=True)
            
    def get_offlaod_layer(self, layer_idx: int, device: torch.device):
        # init stream
        if not hasattr(self, "prefetch_stream"):
            self.prefetch_stream = torch.cuda.Stream()

        # delete previous layer
        torch.cuda.current_stream().synchronize()
        self.evict_previous_layer(layer_idx)
        
        # make sure the current layer is ready
        torch.cuda.synchronize(self.prefetch_stream)

        # load next layer
        self.prefetch_layer((layer_idx + 1) % len(self.layers), device)
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        offload_model=False,
    ):
        try:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # Move inputs to GPU and handle dtype
            if input_ids is not None:
                input_ids = input_ids.to(self.device)
                input_shape = input_ids.size()
                batch_size = input_ids.shape[0]
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
                batch_size = inputs_embeds.shape[0]
                inputs_embeds = inputs_embeds.to(self.device)
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

            if position_ids is not None:
                position_ids = position_ids.to(self.device)

            if past_key_values is None:
                past_length = 0
                past_key_values = tuple([None] * len(self.layers))
            else:
                past_length = past_key_values[0][0].size(-2)
                past_key_values = tuple(
                    tuple(p.to(self.device) if p is not None else None for p in layer_past)
                    for layer_past in past_key_values
                )

            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    past_length, input_shape[-1] + past_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0)

            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
                if batch_size <= 0:
                    raise ValueError("batch_size has to be defined and > 0")
                attention_mask = self._prepare_decoder_attention_mask(
                    attention_mask, input_shape, inputs_embeds, past_length
                )

            # Prepare head mask if needed
            head_mask = [None] * self.config.num_hidden_layers
            
            if inputs_embeds is None:
                with torch.cuda.amp.autocast():
                    inputs_embeds = self.embed_tokens(input_ids)

            hidden_states = inputs_embeds

            # Optimization: release memory of input tensors no longer needed
            del input_ids, inputs_embeds
            torch.cuda.empty_cache()

            # Initialize variables for outputs
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None
            next_decoder_cache = () if use_cache else None

            # Process through transformer layers with memory optimization
            layer_idx = -1
            for idx, (decoder_layer, layer_past) in enumerate(zip(self.layers, past_key_values)):
                layer_idx += 1

                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                with torch.cuda.amp.autocast():
                    if self.gradient_checkpointing and self.training:
                        layer_outputs = self._gradient_checkpointing_func(
                            decoder_layer.__call__,
                            hidden_states,
                            attention_mask,
                            position_ids,
                            layer_past,
                            output_attentions,
                            use_cache,
                            None,
                        )
                    else:
                        if offload_model and not self.training:
                            self.get_offlaod_layer(layer_idx, device=inputs_embeds.device)
                        layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_value=layer_past,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                        )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

                # Clear layer outputs to free memory
                del layer_outputs
                torch.cuda.empty_cache()

            # Final layer norm
            with torch.cuda.amp.autocast():
                hidden_states = self.final_layernorm(hidden_states)

            # Add last hidden state
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Prepare output
            next_cache = next_decoder_cache if use_cache else None
            if not return_dict:
                return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )

        except Exception as e:
            print(f"Error in transformer forward pass: {str(e)}")
            torch.cuda.empty_cache()
            raise
        
        finally:
            # Final cleanup
            if offload_model:
                self.cpu()
                torch.cuda.empty_cache()

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_length):
        # Create causal mask
        batch_size, seq_length = input_shape
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=self.device)
            
        # Convert mask to float and expand
        attention_mask = attention_mask.to(dtype=inputs_embeds.dtype)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.expand(batch_size, 1, seq_length, seq_length + past_length)
        
        return attention_mask
