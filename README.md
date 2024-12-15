# LLaVa with ToMe Token Merging

## Overview

This repository implements the Token Merging (ToMe) algorithm for the vision tower of the LLaVa (Large Language and Vision Assistant) model. The ToMe algorithm provides an efficient method for reducing computational complexity while maintaining model performance by strategically merging tokens during vision processing.

## Utilizing this repository

In order ot use this code, one must install both the LLaVA (https://github.com/haotian-liu/LLaVA) and ToMe (https://github.com/facebookresearch/ToMe) repositories and place the repository in the LLaVa root directory. For enchmakring, this also includes following the LLaVa benchmakring installation process. Afterwords, replace the files in the ./llava/model/multimodal_encoder directory with the ones in this repository to add in token merging to the LLaVa vision tower.
