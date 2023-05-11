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
L.param = io.param.load('../config/config_FD_2ch.yaml').Params
L.param.datapath = 'E:/EMBL files/data 2022/230202_Nup96SNAP_NC_M2/beads_30ms_conventional/30ms_conventional/'
L.param.savename = L.param.datapath + 'test'
images = L.load_data()

#%%
L.getpsfclass()
dataobj = L.prep_data(images)
psfobj,fitter = L.learn_psf(dataobj,time=0)
#%%
loc_FD = L.localize_FD(fitter)
#%%
resfile = L.save_result(psfobj,dataobj,fitter)

# %% show results

f,p = io.h5.load(resfile)
#%%
showpsfvsdata(f,p,index=1)
#%%
showlocalization(f,p)
#%%
showpupil(f,p,index=0)
#%%
showzernike(f,p,index=24)    

#%%
showzernikemap(f,p,index=[4,5,6,7,10,11,12,15,16,21,23])    



# %%
showtransform(f)
np.set_printoptions(precision=4,suppress=True)
print(f.res.T)
# %%
showlearnedparam(f,p)
# %%
