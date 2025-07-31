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
import os
#import scipy as sp
from psflearning.learning import utilities as util
from psflearning.learning.loclib import localizationlib
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
L.param = io.param.combine(basefile='config_base',psftype='insitu',channeltype='1ch',sysfile='smart_tirf')

#%% load psf model
resfile = r'C:\Users\Sheng\Documents\MATLAB\DNA-Paint\ruler\result_1ch\20R-ruler-0.05exp-FocusLock-noPol--2025-06-17_12-47-591_insitu_zernike_single.h5'
f,p = io.h5.load(resfile) 
L.param = p
I_model = f.res.I_model
pz = p.pixel_size.z # unit: um

#%%
L.param.datapath = r'C:\Users\Sheng\Documents\MATLAB\DNA-Paint/ruler/'
filename = '20R-ruler-0.1exp-TIRF-onlyZFocusLockDT10-noPol--2025-07-10_20-02-37'
L.param.filelist = [L.param.datapath + filename+'.h5']
daf = h5.File(L.param.filelist[0],'r')
dat = daf[L.param.varname]
Nframes = dat.shape[0]
batchsize = 1000
ind = list(np.linspace(0,Nframes,Nframes//batchsize+1,dtype=int))
roisize = [9,9]
#%%
start = time.process_time()

dll = localizationlib(usecuda=True)
L.param.roi.roi_size = roisize
L.param.roi.peak_height = 0.1
imsz = 256
startx = 0
starty = 570
x = []
y = []  
z = []
photon = []
bg = []
LL = []
cors = []
crlb = []
frames = []
for i in range(len(ind)-1):
    print('Processing frames %d to %d' % (ind[i], ind[i+1]))    
    L.param.insitu.frame_range = [round(ind[i]),round(ind[i+1])]
    images = L.load_data()
    images = images[:,starty:starty+imsz,startx:startx+imsz]
    L.getpsfclass()
    dataobj = L.prep_data(images)
    _, rois, centers, frame = dataobj.get_image_data()
    cor = centers
    data = rois
    locres = dll.loc_ast(data,I_model,pz,initz=[0.0])
    x.append(cor[:,-1]+locres[-1]['y'].flatten()-roisize[-1]//2)
    y.append(cor[:,-2]+locres[-1]['x'].flatten()-roisize[-2]//2)
    z.append(locres[-1]['z'].flatten())
    LL.append(locres[2].flatten())
    cors.append(centers)
    photon.append(locres[0][2].flatten())
    bg.append(locres[0][3].flatten())
    crlb.append(locres[1])
    frames.append(frame.flatten()+ind[i])
x = np.hstack(x)
y = np.hstack(y)
z = np.hstack(z)
LL = np.hstack(LL)
photon = np.hstack(photon)
bg = np.hstack(bg)
cors = np.vstack(cors)
crlb = np.hstack(crlb)
frames = np.hstack(frames)

end = time.process_time()
print(f"CPU time used: {end - start:.6f} seconds")

#%%
plt.figure(figsize=(10,10))
plt.plot(x,y,'.',markersize=0.1)
# %%
plt.plot(y,z*pz*1e3,'.',markersize=0.1)
# %%
plt.plot(x,z*pz*1e3,'.',markersize=0.1)

#%% show results
llmask = -150
mask = (LL>llmask) & (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(z)) 
h=plt.hist(LL[mask],100)



#%%
h=plt.hist(bg[mask],bins=100)
#%%
h=plt.hist(photon[mask],bins=np.linspace(0,10000,100))
#%%
h=plt.hist(np.sqrt(crlb[0,mask]),bins=np.linspace(0,0.1,100))
#%%
h=plt.hist(np.sqrt(crlb[1,mask]),bins=np.linspace(0,0.1,100))
#%%
h=plt.hist(np.sqrt(crlb[4,mask]),bins=np.linspace(0,1,100))
#%%
x0 = x[mask]
y0 = y[mask]
z0 = z[mask]
#%%
plt.figure(figsize=(10,10))
plt.plot(x0,y0,'.',markersize=0.1)
# %%
plt.plot(y0,z0,'.',markersize=0.1)

# %% save results
res = dict(x=x[mask],y=y[mask],z=z[mask],LL=LL[mask],pz=pz,photon=photon[mask],bg=bg[mask],
           crlb=crlb[:,mask],frames=frames[mask],cors=cors[mask,:])
savepath = L.param.datapath + 'python/result_1ch/'
savename = savepath + filename + '_loc_1ch.h5'
os.makedirs(savepath, exist_ok=True)

with h5.File(savename, "w") as f1:
    g1 = f1.create_group("res")
    for k, v in res.items():
        g1[k] = v
# 