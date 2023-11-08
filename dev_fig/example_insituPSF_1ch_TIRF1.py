#%%
import sys
sys.path.append("..")
from psflearning.psflearninglib import psflearninglib
from psflearning import io
from psflearning.makeplots import *

#%% load config file
L = psflearninglib()
L.param = io.param.combine(basefile='config_base',psftype='insitu',channeltype='1ch',sysfile='TIRF')

#%% edit user defined parameters
L.param.datapath = r'Y:\Projects\Super Critical Angle Localization Microscopy\Data\05-10-2023\Cell 07 data/'
L.param.savename = 'psfmodel_iter'
L.param.keyword = 'Cell7' # keyword of the file name or the full file name without extension
L.param.gain = 0.45
L.param.ccd_offset = 100
L.param.option.insitu.stage_pos = 0.6 # micron, stage position relative to infocus at the coverslip
L.param.option.model.symmetric_mag = True
L.param.option.model.const_pupilmag = True
L.param.option.imaging.RI.med = 1.335
L.param.option.insitu.repeat = 3
L.param.roi.peak_height = 0.5
L.param.option.model.n_max = 5
L.param.option.insitu.z_range = 1.0
L.param.plotall = True
#%% load data and identify candidate emitter
images = L.load_data()
L.getpsfclass()
dataobj = L.prep_data(images)
#%% define background ROI
bgroi = [10,65,120,250]
L.param.option.insitu.backgroundROI = bgroi
plt.figure(figsize=[8,8])
plt.plot(dataobj.centers[:,-1],dataobj.centers[:,-2],'.')
plt.plot([bgroi[2],bgroi[3],bgroi[3],bgroi[2],bgroi[2]],[bgroi[0],bgroi[0],bgroi[1],bgroi[1],bgroi[0]])
plt.axis('equal')
#%% learn PSF
resfile = L.iterlearn_psf(dataobj,time=0)

#%% load result
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
#%%
showcoord(f,p)
print(f.res.stagepos)
#%% show scatter plot in x-y, size indicates the z position
cor = f.res.cor
plt.figure(figsize=[8,8])
maskcor = (cor[:,-1]>bgroi[2]) & (cor[:,-1]<bgroi[3]) & (cor[:,-2]>bgroi[0]) & (cor[:,-2]<bgroi[1]) 
plt.scatter(cor[:,-1],cor[:,-2],f.res.pos[:,0])
plt.scatter(cor[maskcor,-1],cor[maskcor,-2],f.res.pos[maskcor,0])

