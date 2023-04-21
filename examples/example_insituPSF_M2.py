#%%
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
#tf.config.set_visible_devices(gpus[1], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

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
L.param = io.param.load('../config/config_insitu_M2.yaml').Params


#%%
images = L.load_data()
L.getpsfclass()

#%%
dataobj = L.prep_data(images)
resfile = L.iterlearn_psf(dataobj,time=0)
#%%
#psfobj,fitter = L.learn_psf(dataobj,time=0)
#resfile = L.save_result(psfobj,dataobj,fitter)

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
