from pathlib import Path
from typing import Union

from omegaconf import OmegaConf, DictConfig


def load(path: Union[str, Path]) -> DictConfig:
    return OmegaConf.load(path)
