import torch
import torch.nn as nn
from typing import Any, Optional, Tuple

from ToMe.tome.merge import bipartite_soft_matching, merge_wavg
from ToMe.tome.utils import parse_r

class ToMeBlock(nn.Module):
    def __init__(self, num_tokens, dim=1024, r_token=0.5):
        super().__init__()
        self.r_token = parse_r(num_tokens, r_token)
        self.dim = dim
        self.size = None
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Parse the token reduction rate
        r = self.r_token.pop(0)
        if not (0 <= r < 1):
            return hidden_states
        
        # Ensure the input is in the correct shape
        if len(hidden_states.shape) != 3:
            raise ValueError(f"Expected input to be 3D, got {len(hidden_states.shape)}D")

        
        # Perform bipartite soft matching to determine which tokens to merge
        merge, _ = bipartite_soft_matching(
            hidden_states, 
            r,
            class_token=True,
            distill_token=False
        )
        
        # Merge the tokens using weighted average
        merged_states, self.size = merge_wavg(
            merge,
            hidden_states,
            size=self.size
        )
        
        return merged_states

class ToMeCLIPEncoderLayer(nn.Module):
    def __init__(self, vision_tower_layer, num_tokens, r_token, device):
        super().__init__()

        self.device = device

        self.self_attn = vision_tower_layer.self_attn
        self.layer_norm1 = vision_tower_layer.layer_norm1
        self.mlp = vision_tower_layer.mlp
        self.layer_norm2 = vision_tower_layer.layer_norm2
        
        self.tome = ToMeBlock(num_tokens=num_tokens, dim=1024, r_token=r_token)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        hidden_states = hidden_states.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if causal_attention_mask is not None:
            causal_attention_mask = causal_attention_mask.to(self.device)

        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        hidden_states = self.tome(hidden_states)

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs