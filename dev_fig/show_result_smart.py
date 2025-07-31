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
import h5py as h5
from psflearning.learning import utilities as util
from psflearning.learning.loclib import localizationlib
try:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('Running on GPU')
except:
    print('Running on CPU')


# %%
pixelsize = 80 # unit: nm
std_limit = 0.2
folder = r'C:\Users\Sheng\Documents\MATLAB\DNA-Paint\ruler/result_1ch/'
filename = '20R-ruler-0.05exp-FocusLock-noPol--2025-06-17_12-47-59_loc_1ch_dme'
resfile = folder+filename+'.h5'
F = h5.File(resfile,'r')
res = F['res']
std = np.swapaxes(res['crlb'],0,1)
hx_1ch = plt.hist(std[0],bins=np.linspace(0,std_limit,100))
hy_1ch = plt.hist(std[1],bins=np.linspace(0,std_limit,100))
mask = (std[0]<std_limit) & (std[1]<std_limit)
mean_1ch = np.mean(std[:,mask],axis=1)*pixelsize

# %%
folder = r'T:\projects\smart-microscope\data\focuslock\Nanostage Position while imaging\2025-07-15\python\result_2ch/'
filename = 'xyDriftTest-noFocusLock-focusAdjustedManually--2025-07-15_21-24-54_loc_2ch_dme'
resfile = folder+filename+'.h5'
F = h5.File(resfile,'r')
res = F['res']
std = np.swapaxes(res['crlb'],0,1)
hx_2ch = plt.hist(std[0],bins=np.linspace(0,std_limit,100))
hy_2ch = plt.hist(std[1],bins=np.linspace(0,std_limit,100))
mask = (std[0]<std_limit) & (std[1]<std_limit)
mean_2ch = np.mean(std[:,mask],axis=1)*pixelsize

# %%
ftsz = 14
plt.figure(figsize=(6, 4))
plt.plot(hx_1ch[1][:-1]*pixelsize,hx_1ch[0]/np.sum(hx_1ch[0]),'r',label='1ch std_x',linewidth=2)
plt.plot(hx_2ch[1][:-1]*pixelsize,hx_2ch[0]/np.sum(hx_2ch[0]),'b',label='2ch std_x',linewidth=2)
plt.xlabel('std_x (nm)',fontsize=ftsz)
plt.ylabel('count',fontsize=ftsz)
plt.xticks(fontsize=ftsz)
plt.yticks(fontsize=ftsz)
plt.legend()
plt.title(f'1ch: mean={mean_1ch[0]:.2f}nm, 2ch: mean={mean_2ch[0]:.2f}nm',fontsize=ftsz)
# %%
plt.figure(figsize=(6, 4))
plt.plot(hy_1ch[1][:-1]*pixelsize,hy_1ch[0]/np.sum(hy_1ch[0]),'r',label='1ch std_y',linewidth=2)
plt.plot(hy_2ch[1][:-1]*pixelsize,hy_2ch[0]/np.sum(hy_2ch[0]),'b',label='2ch std_y',linewidth=2)
plt.xlabel('std_y (nm)',fontsize=ftsz)
plt.ylabel('count',fontsize=ftsz)
plt.xticks(fontsize=ftsz)
plt.yticks(fontsize=ftsz)
plt.legend()
plt.title(f'1ch: mean={mean_1ch[1]:.2f}nm, 2ch: mean={mean_2ch[1]:.2f}nm',fontsize=ftsz)

# %%
folder = r'T:\projects\cells-labeling\Data\DNA PAINT smart\7_8_25\2025-07-08\python\result_2ch/'
filename = 'Cell2_EGF_Hek293_AlfaEGFR_WT_ALFA75pM_laser100percent--2025-07-08_16-59-04_loc_2ch'
resfile = folder+filename+'.h5'
F = h5.File(resfile,'r')
res = F['res']
# %%
ftsz=14
fig = plt.figure(figsize=[14,5])
spec = gridspec.GridSpec(ncols=2, nrows=1,
                        width_ratios=[3, 3], wspace=0.2,
                        hspace=0.3)
ax = fig.add_subplot(spec[0])
h = plt.hist(res['photon'], bins=np.linspace(0, np.quantile(res['photon'],0.99), 100))
plt.xlabel('photon',fontsize=ftsz)
plt.ylabel('count',fontsize=ftsz)
plt.xticks(fontsize=ftsz)
plt.yticks(fontsize=ftsz)
ax = fig.add_subplot(spec[1])
h = plt.hist(res['bg'], bins=np.linspace(0, np.quantile(res['bg'],0.99), 100))
plt.xlabel('bg',fontsize=ftsz)
plt.xticks(fontsize=ftsz)
plt.yticks(fontsize=ftsz)
plt.ylabel('count',fontsize=ftsz)


# %%
