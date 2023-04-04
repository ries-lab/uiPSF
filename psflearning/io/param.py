from pathlib import Path
from typing import Union

from omegaconf import OmegaConf, DictConfig


def load(path: Union[str, Path]) -> DictConfig:
    return OmegaConf.load(path)

def combine(basefile,psftype=None,channeltype=None,sysfile=None):
    fparam = load('../demo_config/'+basefile+'.yaml').Params
    if psftype is not None:
        psfparam = load('../demo_config/config_'+psftype+'.yaml').Params
        fparam = redefine(fparam,psfparam)
    if channeltype is not None:
        chparam = load('../demo_config/config_'+channeltype+'.yaml').Params
        fparam = redefine(fparam,chparam)
    if sysfile is not None:
        sysparam = load('../demo_config/'+sysfile+'.yaml').Params
        fparam = redefine(fparam,sysparam)
    if psftype == 'zernike' and channeltype == '4pi':
        fparam.PSFtype = 'zernike'
    if psftype == 'insitu':
        fparam.roi.roi_size[-1] = max([4,fparam.roi.roi_size[-1]])
        fparam.roi.roi_size[-2] = max([4,fparam.roi.roi_size[-2]])
        fparam.roi.gauss_sigma[-1] = max([5,fparam.roi.roi_size[-1]])
        fparam.roi.gauss_sigma[-2] = max([5,fparam.roi.roi_size[-2]])
    if psftype == 'FD':
        fparam.option.model.bin = 1

    return fparam

def redefine(baseparam,userparam):
    for k, v in userparam.items():
        try:
            for k1, v1 in v.items():
                try: 
                    for k2, v2 in v1.items():
                        try: 
                            for k3, v3 in v2.items():
                                baseparam[k][k1][k2][k3] = v3
                        except:
                            baseparam[k][k1][k2] = v2
                except:
                    baseparam[k][k1] = v1
        except:
            baseparam[k]=v

    return baseparam


