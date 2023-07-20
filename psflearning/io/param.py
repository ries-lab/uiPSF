from pathlib import Path
from typing import Union
import os
from omegaconf import OmegaConf, DictConfig


def load(path: Union[str, Path]) -> DictConfig:
    return OmegaConf.load(path)

def combine(basefile,psftype=None,channeltype=None,sysfile=None):
    thispath = os.path.dirname(os.path.abspath(__file__))
    pkgpath = os.path.dirname(os.path.dirname(thispath))
    fparam = load(pkgpath+'/config/'+basefile+'.yaml').Params
    if psftype is not None:
        psfparam = load(pkgpath+'/config/psftype/'+psftype+'.yaml').Params
        fparam = redefine(fparam,psfparam)
    if channeltype is not None:
        chparam = load(pkgpath+'/config/channeltype/'+channeltype+'.yaml').Params
        fparam = redefine(fparam,chparam)
    if sysfile is not None:
        sysparam = load(pkgpath+'/config/systemtype/'+sysfile+'.yaml').Params
        fparam = redefine(fparam,sysparam)
    if psftype == 'zernike' and channeltype == '4pi':
        fparam.PSFtype = 'zernike'
    if 'insitu' in psftype:
        fparam.roi.gauss_sigma[-1] = max([4,fparam.roi.gauss_sigma[-1]])
        fparam.roi.gauss_sigma[-2] = max([4,fparam.roi.gauss_sigma[-2]])
        fparam.roi.max_kernel[-1] = max([5,fparam.roi.max_kernel[-1]])
        fparam.roi.max_kernel[-2] = max([5,fparam.roi.max_kernel[-2]])
    if 'FD' in psftype:
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


