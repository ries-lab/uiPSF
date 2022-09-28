from pathlib import Path
from omegaconf import OmegaConf

from psflearning import io


def test_param(tmpdir):
    p = Path(tmpdir) / "dummy_conf.yaml"

    cfg_dict = {"a": 42}
    cfg = OmegaConf.create(cfg_dict)

    OmegaConf.save(cfg, p)

    cfg_reloaded = io.param.load(p)
    assert cfg_reloaded == cfg
