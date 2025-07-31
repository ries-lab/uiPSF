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
from psflearning.learning import utilities as util
from psflearning.learning.loclib import localizationlib
try:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('Running on GPU')
except:
    print('Running on CPU')

#%%
L = psflearninglib()
L.param = io.param.combine(basefile='config_base',psftype='insitu',channeltype='2ch',sysfile='smart_tirf')
#%% load psf model
resfile = r'C:\Users\Sheng\Documents\MATLAB\bead100nm_smart\python\result_2ch\100nmbeads-0.5exp-noFocusLock-8000Frames--2025-07-23_18-57-1811_insitu_zernike_multi.h5'
f,p = io.h5.load(resfile) 
L.param = p
I_model = np.stack([f.res.channel0.I_model, f.res.channel1.I_model])
pz = p.pixel_size.z # unit: um
imgcenter = f.res.imgcenter
T = np.expand_dims(f.res.T,axis=0)
#%%
L.param.datapath = r'C:\Users\Sheng\Documents\MATLAB\bead100nm_smart/'
filename = '100nmbeads-0.5exp-noFocusLock-8000Frames--2025-07-23_18-57-18'
L.param.filelist = [L.param.datapath + filename+'.h5']
daf = h5.File(L.param.filelist[0],'r')
dat = daf[L.param.varname]
Nframes = dat.shape[0]
batchsize = 200
ind = list(np.linspace(0,Nframes,Nframes//batchsize+1,dtype=int))
roisize = [13,13]

#%%
start = time.process_time()
L.param.plotall = False
L.param.roi.roi_size = roisize
L.param.roi.peak_height = 0.6
imsz = 256
startx = 0
starty = 110
dll = localizationlib(usecuda=True)
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
    L.param.channeltype = 'multi'
    L.param.insitu.frame_range = [round(ind[i]),round(ind[i+1])]
    images = L.load_data()
    images = images[:,:,starty:starty+imsz,startx:startx+imsz]
    # L.getpsfclass()
    # dataobj = L.prep_data(images)
    # _, rois, centers, frame = dataobj.get_image_data()
    # cor = np.stack(centers)[...,-2:]
    # data = np.stack(rois)
    L.param.channeltype = 'single'
    L.getpsfclass()
    dataobj = L.prep_data(images[0])
    _, rois, centers, frame = dataobj.get_image_data()
    ref_pos = centers
    cor_ref = np.concatenate((ref_pos, np.ones((ref_pos.shape[0], 1))), axis=1)
    cor_target = np.matmul(cor_ref-imgcenter, f.res.T)[..., :-1]+imgcenter[:-1]
    cor_target = np.int32(cor_target)    
    mask = (cor_target[:,-1]-roisize[-1]//2 >=0) & (cor_target[:,-1]-roisize[-1]//2+roisize[0] <= images.shape[-1]) & (cor_target[:,-2]-roisize[-2]//2 >=0) & (cor_target[:,-2]-roisize[-2]//2+roisize[1] <= images.shape[-2])
    coords = np.hstack([frame.reshape((-1,1)),cor_target])[mask,:]
    rois2 = util.crop_rois(images[1], coords, roisize)
    frame = frame[mask]
    cor = np.stack([ref_pos[mask],cor_target[mask]])
    data = np.stack([rois[mask], rois2])

    locres = dll.loc_ast_dual(data,I_model,pz,cor,imgcenter,T,initz=[0.0])
    x.append(cor[0][:,-1]+locres[-1]['y'].flatten()-roisize[-1]//2)
    y.append(cor[0][:,-2]+locres[-1]['x'].flatten()-roisize[-2]//2)
    z.append(locres[-1]['z'].flatten())
    LL.append(locres[2].flatten())
    cors.append(cor[0])
    photon.append(locres[0][3].flatten())
    bg.append(locres[0][4].flatten())
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

#%% 
llmask = -1200
mask = (LL>llmask) & (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(z)) 
h=plt.hist(LL[mask],100)

#%%
mask = mask & (np.sqrt(crlb[0])<0.025)

#%%
h=plt.hist(bg[mask],bins=100)
#%%
h=plt.hist(photon[mask],bins=np.linspace(100,10000,100))
#%%
h=plt.hist(np.sqrt(crlb[0,mask]),bins=np.linspace(0,0.1,100))
#%%
h=plt.hist(np.sqrt(crlb[1,mask]),bins=np.linspace(0,0.1,100))
#%%
h=plt.hist(np.sqrt(crlb[2,mask]),bins=np.linspace(0,1,100))
#%%
x0 = x[mask]
y0 = y[mask]
z0 = z[mask]
#%%
plt.figure(figsize=(10,10))
plt.plot(x0,y0,'.',markersize=0.1)
# %%
plt.plot(y0,z0,'.',markersize=0.1)

#%%
plt.plot(frames[mask],y0,'.',markersize=0.1)

# %% save results
res = dict(x=x[mask],y=y[mask],z=z[mask],LL=LL[mask],pz=pz,photon=photon[mask],bg=bg[mask],
           crlb=crlb[:,mask],frames=frames[mask],cors=cors[mask,:])
savepath = L.param.datapath + 'python/result_2ch/'
savename = savepath + filename + '_loc_2ch.h5'
# create directory if it does not exist
os.makedirs(savepath, exist_ok=True)
with h5.File(savename, "w") as f1:
    g1 = f1.create_group("res")
    for k, v in res.items():
        g1[k] = v

# %%
