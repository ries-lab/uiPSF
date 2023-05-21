#%%
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6600)])
sys.path.append("..")
from psflearning.psflearninglib import psflearninglib
from psflearning import io
from psflearning.makeplots import *

maindatadir = io.param.load('../config/path/config_path.yaml').main_data_dir

#%% load parameters
L = psflearninglib()
L.param = io.param.load('../config/config_insitu_4pi.yaml').Params
L.param.datapath = r'E:\EMBL files\data for PSF learning\02-26-2021 nup96 AF647/'
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
showpupil(f,p)

#%%    
showzernike(f,p,index=0)

#%%
showlearnedparam_insitu(f,p)

# %%
showtransform(f)
np.set_printoptions(precision=4,suppress=True)
print(f.res.T)

# %%
zT = p.fpi.modulation_period/p.pixel_size.z
plt.plot(f.locres.loc.zast,np.mod(f.locres.loc.z,zT),'.')
# %%
