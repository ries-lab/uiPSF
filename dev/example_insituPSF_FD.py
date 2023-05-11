#%%
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

maindatadir = io.param.load('../config/path/config_path.yaml').main_data_dir

#%% load parameters
L = psflearninglib()
L.param = io.param.load('../config/config_insitu_FD.yaml').Params
L.param.datapath = r'C:\Users\Sheng\Documents\MATLAB\data\UNM-TIRF\05-10-2023\cell7/'
L.param.savename = L.param.datapath + L.param.savename
images = L.load_data()

#%%
L.getpsfclass()
dataobj = L.prep_data(images)
#%%
resfile = L.iterlearn_psf(dataobj,time=0)

#%%
f,p = io.h5.load(resfile)
#%%
showpsfvsdata_insitu(f,p)
#%%
showpupil(f,p,index=0)
try:
    showzernike(f,p,index=0)
except:
    print('no Zernike coefficients')
#%%
showlearnedparam_insitu(f,p)

#%%
showzernikemap(f,index=[4,5,6,7,8,9,10,11,12,15,16])

# %%
cor = f.res.cor
plt.scatter(cor[:,-1],cor[:,-2],f.res.pos[:,0])


# %%
