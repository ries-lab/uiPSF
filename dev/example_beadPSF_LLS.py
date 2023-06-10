# %% imports
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

sys.path.append("..")
from psflearning.psflearninglib import psflearninglib
from psflearning import io
from psflearning.makeplots import *

#%% load parameters
L = psflearninglib()
L.param = io.param.load('../config/config_voxel_LLS.yaml').Params
L.param.datapath = r'E:\EMBL files\data for PSF learning\LLS_Data\20220207_SMAP\full_stacks_no_deskew/'
L.param.savename = L.param.datapath + 'test1'
images = L.load_data()

#%%
L.getpsfclass()
dataobj = L.prep_data(images)
#%%
for k in range(0,3): # increase iteration number if necessary
    psfobj,fitter = L.learn_psf(dataobj,time=0)
resfile = L.save_result(psfobj,dataobj,fitter)

# %% show results
f,p = io.h5.load(resfile)

#%%
for i in range(0,np.min([9,f.rois.psf_data.shape[0]])):
    showpsfvsdata(f,p,index=i)
#%%
showlocalization(f,p)
#%%
showpupil(f,p)
#%%
showzernike(f,p)    

# %%
showlearnedparam(f,p)
# %%
