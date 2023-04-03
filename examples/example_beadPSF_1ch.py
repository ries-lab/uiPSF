# %% imports
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
L.param = io.param.load('../config/config_zernike_1ch.yaml').Params
L.param.datapath = maindatadir + L.param.datapath
L.param.savename = L.param.datapath + L.param.savename
#L.param.datapath = L.param.datapath
images = L.load_data()

#%%
L.getpsfclass()
dataobj = L.prep_data(images)
psfobj,fitter = L.learn_psf(dataobj,time=0)
resfile = L.save_result(psfobj,dataobj,fitter)

# %% show results

f,p = io.h5.load(resfile)
#%%
showpsfvsdata(f,p,index=0)
#%%
showlocalization(f,p)

#%%
try:
    showpupil(f,p)
except:
    print('no pupil')

try:
    showzernike(f,p)
except:
    print('no Zernike coefficients')

showlearnedparam(f,p)
# %%
[0.7478614, 0.7868592] # radius=0.025, bin=3
[0.75469255, 0.7930746 ] # radius=0, bin=3
[0.8040463 , 0.83986473] # radius=0, bin=1
[0.5191372 , 0.51132196] # radius=0.025, bin=3, vector
[0.53423333, 0.5222833 ] # radius=0.025, bin=2, vector
[0.59332156, 0.5808556 ] # radius=0.025, bin=1, vector

[0.8877916 , 1.7708156 , 1.5449411 , 1.7708156 , 0.59193677, 1.8801146 ]
[2.2039149 , 0.9988498 , 1.0983567 , 0.9988498 , 1.3747756 ,0.72415626]

[0.9431401, 1.8442363, 1.6164677, 1.8442363, 0.6229577, 1.9458437] #sigma=0.05
[0.8978031, 1.7640057, 1.5396551, 1.7640057, 0.6069426, 1.869314 ] #sigma=0.1
[0.95390594, 1.7663572 , 1.5249628 , 1.7663572 , 0.649004  ,1.8163725 ]#sigma=0.2
[1.4368327, 2.4023893, 1.9870223, 2.4023893, 1.0608294, 2.3228884] #sigma=0.3
[1.5022764, 2.04701  , 1.601577 , 2.04701  , 1.1209229, 1.8646094] #sigma=0.4

# create genpsfmodel() for each class, then call in post processing