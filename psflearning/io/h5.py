from pathlib import Path
from typing import Union

from omegaconf import OmegaConf, DictConfig

import hdfdict
from dotted_dict import DottedDict
import h5py

def load(path: Union[str, Path]) -> DictConfig:
    f = h5py.File(path, 'r')
    res = DottedDict(hdfdict.load(f,lazy=False))
    params = OmegaConf.create(f.attrs['params'])
    return res, params
