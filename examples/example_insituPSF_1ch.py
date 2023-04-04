#%%
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("..")
from psflearning.psflearninglib import psflearninglib
from psflearning import io
from psflearning.makeplots import *

maindatadir = io.param.load('../config/config_path.yaml').main_data_dir

#%% load parameters
L = psflearninglib()
L.param = io.param.load('../config/config_insitu_tetrapod_yoav.yaml').Params
L.param.datapath = maindatadir + L.param.datapath
L.param.savename = L.param.datapath + L.param.savename
images = L.load_data()

#%%
L.getpsfclass()
dataobj = L.prep_data(images)
resfile = L.iterlearn_psf(dataobj,time=0)

#%%
f,p = io.h5.load(resfile)
#%%
showpsfvsdata_insitu(f,p)
#%%
showpupil(f,p)
try:
    showzernike(f,p)
except:
    print('no Zernike coefficients')
#%%
showlearnedparam_insitu(f,p)





#%%
[0.5215881 , 0.48724937]   #bin=1
[0.4527834, 0.4128101]  #bin=2








# %%
