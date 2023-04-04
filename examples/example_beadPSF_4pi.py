# %% imports
import sys
import matplotlib.pyplot as plt
import numpy as np


sys.path.append("..")
from psflearning.psflearninglib import psflearninglib
from psflearning import io
from psflearning.makeplots import *
# only for testing, easier to switch between win and linux
maindatadir = io.param.load('../config/config_path.yaml').main_data_dir 
#%% load parameters
L = psflearninglib()
L.param = io.param.load('../config/config_zernike_4pi.yaml').Params
L.param.datapath = maindatadir + L.param.datapath
L.param.savename = L.param.datapath + L.param.savename
images = L.load_data()

#%%
L.getpsfclass()

dataobj = L.prep_data(images)

psfobj,fitter = L.learn_psf(dataobj,time=0)

resfile = L.save_result(psfobj,dataobj,fitter)

# %% show results
f,p = io.h5.load(resfile)
#%%
showpsfvsdata(f,p,index=0)
#%%
showlocalization(f,p)
#%%
try:
    showpupil(f,p)
except:
    print('no learned pupil')
try:
    showzernike(f,p)
except:
    print('no learned Zernike coefficients')
#%%
showtransform(f)
np.set_printoptions(precision=4,suppress=True)
print(f.res.T)
#%%
showlearnedparam(f,p)


#%%
Nchannel = f.rois.psf_data.shape[0]
if hasattr(f.res.channel0,'sigma'):
    for i in range(Nchannel):
        print(f.res['channel'+str(i)].sigma)
    for i in range(Nchannel):
        print(f.res['channel'+str(i)].modulation_depth)
print(f.res.channel0.phase_dm)













