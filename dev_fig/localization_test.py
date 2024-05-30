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
L.param = io.param.combine(basefile='config_base',psftype='insitu',channeltype='1ch',sysfile='TIRF')
#%%
L.param.datapath = r'Y:\Projects\Super Critical Angle Localization Microscopy\Data\10-06-2023\Data4/'
L.param.filelist = [r'Y:\Projects\Super Critical Angle Localization Microscopy\Data\10-06-2023\Data4\Data4-2023-10-7-17-36-53.h5']

#%% load psf model
resfile = r'Y:\Projects\Super Critical Angle Localization Microscopy\Data\10-06-2023\Data4\psf_kmed2_insitu_zernike_single.h5'
f,p = io.h5.load(resfile) 
L.param = p
L.param.roi.peak_height = 0.1
L.param.insitu.frame_range = []
#%% prepare segementation
sigma = L.param.roi.gauss_sigma
poolsize = L.param.roi.max_kernel
roisize = [17,17]
thresh = L.param.roi.peak_height

conv2d,max_pool_2d,conv_uniform = util.gen_layers(f.rois.image_size,sigma,poolsize,roisize)


#%% load data
rois = []
coords = []
for i in range(200):
    L.param.insitu.dataId = i
    images = L.load_data()
    #%
    roi,coord = util.crop_rois(images,conv2d,max_pool_2d,conv_uniform,thresh,roisize)
    rois.append(roi)
    coords.append(coord)    
rois = np.concatenate(rois)
coords = np.concatenate(coords)
#%% localization
nbin = np.ceil(np.abs(f.res.stagetilt.flatten()*f.rois.image_size[1:]/0.01)).astype(int) # in each subregion the stage tilt is within 10 nm
nbin_x = nbin[-1]
nbin_y = nbin[-2]
count_x, edge_x = np.histogram(coords[:,-1], nbin_x)
count_y, edge_y = np.histogram(coords[:,-2], nbin_y)
ind_x = np.digitize(coords[:,-1], edge_x)
ind_y = np.digitize(coords[:,-2], edge_y)

x = []
y = []  
z = []
photon = []
bg = []
LL = []
cors = []
crlb = []
for xx in range(1,nbin_x+1):
    for yy in range(1,nbin_y+1):
        maskid = np.where((ind_x==xx) & (ind_y==yy))
        maskid[0].size

        #%
        stpos = f.res.stagepos + coords[maskid,-1]*f.res.stagetilt[-1] + coords[maskid,-2]*f.res.stagetilt[-2]
        stpos_avg = np.mean(stpos)
        pz = 0.002 # unit: um
        L.param.pixel_size.z = pz
        Nz = 121
        zpos  = np.linspace(0,Nz-1,Nz,dtype=np.complex64).reshape((Nz,1,1))
        f,_ = L.genpsf(f,Nz=Nz,xsz=21,stagepos=stpos_avg,zpos = zpos)
        I_model = f.res.I_model
        #% localize
        data = rois[maskid]
        offset = np.min((np.quantile(data,1e-3),0))
        data = data - offset            
        dll = localizationlib(usecuda=True)
        locres = dll.loc_ast(data,I_model,pz,initz=[0.0])
        #% plot results
        x.append(coords[maskid,-1]+locres[-1]['y'].flatten()-roisize[-1]//2)
        y.append(coords[maskid,-2]+locres[-1]['x'].flatten()-roisize[-2]//2)
        z.append(locres[-1]['z'].flatten())
        LL.append(locres[2].flatten())
        cors.append(coords[maskid])
        photon.append(locres[0][2].flatten())
        bg.append(locres[0][3].flatten())
        crlb.append(locres[1])

x = np.hstack(x)
y = np.hstack(y)
z = np.hstack(z)
LL = np.hstack(LL)
photon = np.hstack(photon)
bg = np.hstack(bg)
cors = np.vstack(cors)
crlb = np.hstack(crlb)
#%% 
x = x.flatten()
y = y.flatten()
z = z.flatten()
LL = LL.flatten()
photon = photon.flatten()
bg = bg.flatten()
#%% show results
llmask = -300
mask = (z>0) & (LL>llmask)
plt.hist(z[mask],100)
#plt.hist(z*pz*1e3,100)


#%% identify background region
mask = (z<22) & (z>0)& (LL>llmask)
plt.hist(z[mask]*pz*1e3,bins=100)
#%%
plt.hist(bg[mask],bins=100)
#%%
plt.hist(photon[mask],bins=np.linspace(1000,5000,100))
#%%
plt.hist(LL[mask],bins=100)
#%%
x0 = x[mask]
y0 = y[mask]
z0 = z[mask]
#%%
plt.figure(figsize=(10,10))
plt.plot(x0,y0,'.',markersize=0.01)
# %%
plt.plot(y0,z0*pz*1e3,'.',markersize=0.01)
# %%
plt.plot(x0,z0*pz*1e3,'.',markersize=0.01)

# %% estimate tilt
X = np.vstack([y0,x0,np.ones(x0.shape)])
Y = z0*pz*1e3
beta=np.matmul(np.matmul(np.linalg.inv(np.matmul(X,X.transpose())),X),Y.transpose())
# %%
zcorrected = z0-np.matmul(beta,X)/pz/1e3
# %%
plt.plot(y0,zcorrected,'.',markersize=0.01)
# %%
plt.plot(x0,zcorrected,'.',markersize=0.01)
# %%
residue_total_tilt = 1.52/1.35*beta[0:2]*f.rois.image_size[-1] # residual stage tilt across FOV, unit: nm
print('residual total tilt: ',residue_total_tilt)
# %%
stagetilt = f.res.stagetilt.flatten()*1e3-1.52/1.35*beta[0:2] # corrected stage tilt, unit: nm/pixel
print('updated stage tilt: ',list(stagetilt))
# %%
f.res.stagetilt = stagetilt/1e3
# %% save results
res = dict(x=x,y=y,z=z,LL=LL,pz=pz,photon=photon,bg=bg,cors=cors,crlb=crlb)
savename = r'C:\Users\Sheng\Documents\git\python\uiPSF\dev_fig\loc_test_fulldata1.h5'
with h5.File(savename, "w") as f1:
    g1 = f1.create_group("res")
    for k, v in res.items():
        g1[k] = v
# %%
