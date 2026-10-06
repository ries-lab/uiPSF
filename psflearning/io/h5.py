from pathlib import Path
from typing import Union

from omegaconf import OmegaConf, DictConfig

from dotted_dict import DottedDict
import h5py


def _read_group(group):
    """Read an HDF5 group into nested dicts of arrays.

    Replaces hdfdict, which does not import with numpy >= 2.
    """
    out = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            out[key] = DottedDict(_read_group(item))
        else:
            out[key] = item[()]
    return out


def load(path: Union[str, Path]) -> DictConfig:
    with h5py.File(path, 'r') as f:
        res = DottedDict(_read_group(f))
        params = OmegaConf.create(f.attrs['params'])
    return res, params
