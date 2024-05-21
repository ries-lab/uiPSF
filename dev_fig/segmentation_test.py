#%%
import sys
sys.path.append("..")
from psflearning.psflearninglib import psflearninglib
from psflearning import io
from psflearning.makeplots import *
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import numpy as np
#import scipy as sp

#main_data_dir = io.param.load('datapath.yaml').main_data_dir
try:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('Running on GPU')
except:
    print('Running on CPU')

#%%
L = psflearninglib()
L.param = io.param.combine(basefile='config_base',psftype='insitu',channeltype='1ch',sysfile='SEQ')
#%%
L.param.datapath = r'Y:\Projects\Super Critical Angle Localization Microscopy\Data\10-06-2023\Data4/'
L.param.filelist = [r'Y:\Projects\Super Critical Angle Localization Microscopy\Data\10-06-2023\Data4\Data4-2023-10-7-17-36-53.h5']
#L.param.datapath = r'Y:\Projects\Super Critical Angle Localization Microscopy\Data\01-12-24\Data4\0.35nM Lifeact\Cell_04\Label_01/'
#L.param.filelist = [r'Y:\Projects\Super Critical Angle Localization Microscopy\Data\01-12-24\Data4\0.35nM Lifeact\Cell_04\Label_01\Data_2024-1-13-10-41-6.h5']
L.param.roi.peak_height = 0.4
L.param.insitu.frame_range = []
#%% load psf model
resfile = r'Y:\Projects\Super Critical Angle Localization Microscopy\Data\10-06-2023\Data4\psf_kimm_insitu_zernike_single.h5'
#resfile = r'Y:\Projects\Super Critical Angle Localization Microscopy\Data\01-12-24\Data4\0.35nM Lifeact\Cell_04\Label_01\tiltFD\psf_kimm4_insitu_zernike_single.h5'
f,p = io.h5.load(resfile) 





#%% localize
x,y,z,LL = L.localize(f,datarange=[0,5])
# %% remove outliers
maskll = LL>-300
xf = x[maskll]
yf = y[maskll]
zf = z[maskll]
#%
plt.plot(xf,zf,'.',markersize=1)

# %%
plt.plot(xf,yf,'.',markersize=1)


#%% identify background region
x0,y0,z0 = L.identify_background(xf,yf,zf,zpeak=0.5)
plt.plot(x0,y0,'.',markersize=1)
# %%
plt.plot(y0,z0,'.',markersize=1)
# %%
plt.plot(x0,z0,'.',markersize=1)

# %% estimate tilt
X = np.vstack([y0,x0,np.ones(x0.shape)])
Y = z0*p.pixel_size.z*1e3
beta=np.matmul(np.matmul(np.linalg.inv(np.matmul(X,X.transpose())),X),Y.transpose())
# %%
zcorrected = z0-np.matmul(beta,X)/p.pixel_size.z/1e3
# %%
plt.plot(y0,zcorrected,'.',markersize=1)
# %%
plt.plot(x0,zcorrected,'.',markersize=1)
# %%
