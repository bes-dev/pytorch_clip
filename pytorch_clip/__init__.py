"""
Copyright 2021 by Sergei Belousov
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import typing
import torch.nn as nn
from omegaconf import OmegaConf
from pytorch_clip.utils import res_path
from pytorch_clip.models import CLIP, ruCLIP
from pytorch_clip.processor import TextProcessor


def get_models_list() -> typing.List[str]:
    """ Get list of available CLIP models.
    Returns:
        models_list (list[str]): list of available models.
    """
    return [name.split(".")[0] for name in os.listdir(res_path(f"configs"))]


def get_clip_model(clip_type: str = "clip_vit_b32", input_range: typing.Tuple[float, float] = (-1.0, 1.0), cache_dir: str = "/tmp/") -> typing.Tuple[nn.Module, TextProcessor, nn.Module, OmegaConf]:
    """Build CLIP model from config file.
    Arguments:
        cfg (OmegaConf): configuration of the model.
        input_range (tuple[float, float]): input range.
        cache_dir (str): path to cache dir.
    Returns:
        model (nn.Module): CLIP model.
        text_processor (TextProcessor): text processor.
        image_processor (nn.Module): image processor.
        cfg (OmegaConf): configuration of the CLIP model.
    """
    assert clip_type in get_models_list(), f"Unknown clip_type: {clip_type}"
    cfg = OmegaConf.load(res_path(f"configs/{clip_type}.yml"))
    if cfg.type == "clip":
        return *CLIP.from_pretrained(cfg, input_range, cache_dir=cache_dir), cfg
    elif cfg.type  == "ruclip":
        return *ruCLIP.from_pretrained(cfg, input_range, cache_dir=cache_dir), cfg
    else:
        raise ValueError(f"Unknown model type: {cfg.type}. Available model types: [clip, ruclip]")
