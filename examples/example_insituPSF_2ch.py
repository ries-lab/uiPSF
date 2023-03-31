# %% imports
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("..")
from psflearning.psflearninglib import psflearninglib
from psflearning import io
from psflearning.makeplots import *
# only for testing, easier to switch between win and linux
maindatadir = io.param.load('../config/config_path.yaml').main_data_dir 
#%% load parameters
L = psflearninglib()
L.param = io.param.load('../config/config_insitu_2ch.yaml').Params
L.param.datapath = maindatadir + L.param.datapath
L.param.savename = L.param.datapath + L.param.savename
images = L.load_data()

#%%
L.getpsfclass()
dataobj = L.prep_data(images)

#%%
resfile = L.iterlearn_psf(dataobj,time=0)

# %% show results
f,p = io.h5.load(resfile)

# %%
showpsfvsdata_insitu(f,p)
showpupil(f,p)
#%%
showzernike(f,p)
# %%
showtransform(f)
np.set_printoptions(precision=4,suppress=True)
print(f.res.T)

#%%
showlearnedparam_insitu(f,p)
#%%
for i in range(2):
    print(f.res['channel'+str(i)].sigma)




