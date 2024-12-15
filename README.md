# LLaVa with ToMe Token Merging

## Overview

This repository implements the Token Merging (ToMe) algorithm for the vision tower of the LLaVa (Large Language and Vision Assistant) model. The ToMe algorithm provides an efficient method for reducing computational complexity while maintaining model performance by strategically merging tokens during vision processing.

## Utilizing this repository

In order ot use this code, one must install both the LLaVA (https://github.com/haotian-liu/LLaVA) and ToMe (https://github.com/facebookresearch/ToMe) repositories and place the repository in the LLaVa root directory. For enchmakring, this also includes following the LLaVa benchmakring installation process. Afterwords, replace the files in the ./llava/model/multimodal_encoder directory with the ones in this repository to add in token merging to the LLaVa vision tower.

## Implementation Details

The integration of Token Merging (ToMe) into the CLIP vision tower involves a strategic modification of the vision encoder's architecture. The implementation focuses on selectively reducing token complexity while preserving the core representational capabilities of the vision transformer.

**ToMeBlock**

The ToMeBlock class is the core mechanism for token merging. It implements the token reduction algorithm with several critical features:

- Configurable token reduction rate (r_token)
- Uses bipartite soft matching to determine token merging
- Applies weighted average merging to maintain feature representational quality

**ToMeCLIPEncoderLayer**

This custom encoder layer wraps the original CLIP vision tower layer and introduces the ToMe token merging:
```
class ToMeCLIPEncoderLayer(nn.Module):
    def __init__(self, vision_tower_layer, num_tokens, r_token, device):
        super().__init__()
        self.self_attn = vision_tower_layer.self_attn
        self.layer_norm1 = vision_tower_layer.layer_norm1
        self.mlp = vision_tower_layer.mlp
        self.layer_norm2 = vision_tower_layer.layer_norm2
        
        self.tome = ToMeBlock(num_tokens=num_tokens, dim=1024, r_token=r_token)
```

The integration process occurs in the ```add_tome_to_clip_vision_tower``` method of the CLIPVisionTower class:

```
def add_tome_to_clip_vision_tower(self, vision_tower, num_tokens, r_token, device):
    vision_tower_layers = vision_tower.vision_model.encoder.layers
    vision_tower_layers = nn.ModuleList([
        ToMeCLIPEncoderLayer(layer, num_tokens, r_token, device) 
        for layer in vision_tower_layers
    ])
    vision_tower.vision_model.encoder.layers = vision_tower_layers
    return vision_tower
```

The token merging process if meant to fit within the existing vision tower forward process, seamlessly intercepting and modifying the hidden states between the self-attention and MLP layers of each transformer block. More specifically, we perform bipartite soft matching on the existing tokens and merge them accoridngly before sending it to the MLP.

## Current benchmarking progress

As of right now, the code is implemented, but there are issues with CUDA memory allocation when using CUDA. This could be because the original implementation for the vision tower utilized a device map, whereas the ToMe implementation cannot due to the fact that we are adding a layer to the existing CLIP Vision Tower. Specifically, the device mapping strategy in the original vision tower likely distributed model layers across multiple GPUs or GPU memory segments, which allowed for more efficient memory management. In contrast, the current ToMe implementation may be forcing the entire model onto a single devices, resulting in CUDA memory issues. To resolve this, we would need to carefully modify the ToMe integration to respect the original device mapping strategy, potentially by implementing custom CUDA memory allocation routines or restructuring how tokens are merged to maintain the original memory distribution pattern.
