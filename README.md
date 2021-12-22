# pytorch_clip: Pytorch API to work with CLIP models.
[![Downloads](https://pepy.tech/badge/pytorch_clip)](https://pepy.tech/project/pytorch_clip)
[![Downloads](https://pepy.tech/badge/pytorch_clip/month)](https://pepy.tech/project/pytorch_clip)
[![Downloads](https://pepy.tech/badge/pytorch_clip/week)](https://pepy.tech/project/pytorch_clip)


## Install package

```bash
pip install pytorch_clip
```

## Install the latest version

```bash
pip install --upgrade git+https://github.com/bes-dev/pytorch_clip.git
```

## Features
- Supports original CLIP models by OpenAI and ruCLIP model by SberAI.

## Usage

### Simple code

```python
import torch
from pytorch_clip import get_models_list, get_clip_model

print(get_models_list())

model, text_processor, image_processor = get_clip_model()
```