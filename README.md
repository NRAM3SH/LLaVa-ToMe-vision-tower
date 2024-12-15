# LLaVa with ToMe Token Merging

## Overview

This repository implements the Token Merging (ToMe) algorithm for the vision tower of the LLaVa (Large Language and Vision Assistant) model. The ToMe algorithm provides an efficient method for reducing computational complexity while maintaining model performance by strategically merging tokens during vision processing.

## Utilizing this repository

In order ot use this code, one must install both the LLaVA (https://github.com/haotian-liu/LLaVA) and ToMe (https://github.com/facebookresearch/ToMe) repositories and place the repository in the LLaVa root directory. For enchmakring, this also includes following the LLaVa benchmakring installation process. Afterwords, replace the files in the ./llava/model/multimodal_encoder directory with the ones in this repository to add in token merging to the LLaVa vision tower.

## Current progress

As of right now, the code is implemented, but there are issues with CUDA memory allocation when using CUDA. This could be because the original implementation for the vision tower utilized a device map, whereas the ToMe implementation cannot due to the fact that we are adding a layer to the existing CLIP Vision Tower. Specifically, the device mapping strategy in the original vision tower likely distributed model layers across multiple GPUs or GPU memory segments, which allowed for more efficient memory management. In contrast, the current ToMe implementation may be forcing the entire model onto a single devices, resulting in CUDA memory issues. To resolve this, we would need to carefully modify the ToMe integration to respect the original device mapping strategy, potentially by implementing custom CUDA memory allocation routines or restructuring how tokens are merged to maintain the original memory distribution pattern.
