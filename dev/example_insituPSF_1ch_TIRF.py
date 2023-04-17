#%%
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[1], 'GPU')
tf.config.experimental.set_memory_growth(gpus[1], True)

import sys
import matplotlib.pyplot as plt
import numpy as np
import hdfdict
from dotted_dict import DottedDict
import h5py

sys.path.append("..")
from psflearning.psflearninglib import psflearninglib
from psflearning import io
from psflearning.makeplots import *
maindatadir = io.param.load('../config/config_path.yaml').main_data_dir

#%% load parameters
L = psflearninglib()
L.param = io.param.load('../config/config_insitu_TIRF.yaml').Params
L.param.datapath = '/home/kiwibogo/shares/lidkelab/Personal Folders/Sheng/data/TIRF/'
L.param.savename = L.param.datapath + L.param.savename
images = L.load_data()

#%%
L.getpsfclass()
dataobj = L.prep_data(images)
#%%
L.param.option.backgroundROI = [0,50,150,250] # [ymin, ymax, xmin, xmax]
#%%
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
print(f.res.stagepos)
# %%
