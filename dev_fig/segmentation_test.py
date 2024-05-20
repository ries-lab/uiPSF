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
import scipy as sp
import scipy.signal as sig
import psflearning.learning.utilities as util
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
L.param = io.param.combine(basefile='config_base',psftype='insitu',channeltype='1ch',sysfile='SEQ')
#%%
#L.param.datapath = r'Y:\Projects\Super Critical Angle Localization Microscopy\Data\10-06-2023\Data4/'
#L.param.filelist = [r'Y:\Projects\Super Critical Angle Localization Microscopy\Data\10-06-2023\Data4\Data4-2023-10-7-17-36-53.h5']
L.param.datapath = r'Y:\Projects\Super Critical Angle Localization Microscopy\Data\01-12-24\Data4\0.35nM Lifeact\Cell_04\Label_01/'
L.param.filelist = [r'Y:\Projects\Super Critical Angle Localization Microscopy\Data\01-12-24\Data4\0.35nM Lifeact\Cell_04\Label_01\Data_2024-1-13-10-41-6.h5']
L.param.roi.peak_height = 0.4
L.param.insitu.frame_range = []
#%% load psf model
#resfile = r'Y:\Projects\Super Critical Angle Localization Microscopy\Data\10-06-2023\Data4\psf_kimm_insitu_zernike_single.h5'
resfile = r'Y:\Projects\Super Critical Angle Localization Microscopy\Data\01-12-24\Data4\0.35nM Lifeact\Cell_04\Label_01\tiltFD\psf_kimm4_insitu_zernike_single.h5'
f,p = io.h5.load(resfile) 
I_model = f.res.I_model
#%%
L.param.insitu.dataId = 0
images = L.load_data()

#%% build layers for segmentation
sigma = [2,2]
poolsize = [3,3]
roisize = [15,15]
conv2d,max_pool_2d,conv_uniform = util.gen_layers(images.shape,sigma,poolsize,roisize)

#%% localize
thresh = 0.3
x = []
y = []
z = []
LL = []
batchsize = 1000
for id in range(5,7):
    # load data
    L.param.insitu.dataId = id
    images = L.load_data()
    # batch process
    Nf = images.shape[0]
    ind = list(range(0,Nf,batchsize))+[Nf]
    for i in range(len(ind)-1):
        # segment
        rois,coords = util.crop_rois(images[ind[i]:ind[i+1]],conv2d,max_pool_2d,conv_uniform,thresh,roisize)
        #% remove negative values
        offset = np.min((np.quantile(rois,1e-3),0))
        rois = rois - offset
        #% localize
        dll = localizationlib(usecuda=True)
        pz = p.pixel_size.z
        locres = dll.loc_ast(rois,I_model,pz)
        # collect results
        x.append(coords[:,-1]+locres[-1]['x'].flatten())
        y.append(coords[:,-2]+locres[-1]['y'].flatten())
        z.append(locres[-1]['z'].flatten())
        LL.append(locres[2])
# combine results
x = np.hstack(x)
y = np.hstack(y)
z = np.hstack(z)
LL = np.hstack(LL)
# %% remove outliers
maskll = LL>-300
xf = x[maskll]
yf = y[maskll]
zf = z[maskll]
#%
plt.plot(xf,zf,'.',markersize=1)

# %%
plt.plot(xf,yf,'.',markersize=1)


#%% select background region
nbin_x = 6
nbin_y = 6
count_x, edge_x = np.histogram(xf, nbin_x)
count_y, edge_y = np.histogram(yf, nbin_y)
ind_x = np.digitize(xf, edge_x)
ind_y = np.digitize(yf, edge_y)
x0 = []
y0 = []
z0 = []
for xx in range(1,nbin_x+1):
    for yy in range(1,nbin_y+1):
        maskid = np.where((ind_x==xx) & (ind_y==yy))
        if maskid[0].size>0:
            zi = zf[maskid]                                        
            out = np.histogram(zi,bins=np.arange(0,25,0.1))
            g = out[0]
            bins = (out[1][1:]+out[1][:-1])/2
            ind, _ = sig.find_peaks(g/np.max(g),height=0.07,width=None)
            zcutoff = bins[ind[0]]+1
            mask = zi<zcutoff
            x0.append(xf[maskid][mask])
            y0.append(yf[maskid][mask])
            z0.append(zf[maskid][mask])
            #print(zcutoff)
            #plt.plot(bins,g)
            #plt.show()
x0 = np.hstack(x0)
y0 = np.hstack(y0)
z0 = np.hstack(z0)
plt.plot(x0,y0,'.',markersize=1)
# %%
plt.plot(y0,z0,'.',markersize=1)
# %%
plt.plot(x0,z0,'.',markersize=1)
# %%
plt.plot(x0,y0,'.',markersize=1)
# %%
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
