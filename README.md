# pytorch_clip: Pytorch API to work with CLIP models.

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