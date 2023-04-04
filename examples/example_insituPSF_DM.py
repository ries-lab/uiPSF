#%%
import sys
import matplotlib.pyplot as plt
import numpy as np
import glob

sys.path.append("..")
from psflearning.psflearninglib import psflearninglib
from psflearning import io
from psflearning.learning import utilities as util 
from psflearning.makeplots import *

maindatadir = io.param.load('../config/config_path.yaml').main_data_dir

#%% load parameters
folder = maindatadir+'insitu data/from Yiming/Tubulin/'
folderlist = glob.glob(folder+'/*'+'depth'+'*/')
#%%
for foldername in folderlist:
    nl = foldername.split('zernike_')[1].split('_')
    j = util.nl2noll(int(nl[0]),int(nl[1]))-1
    L = psflearninglib()
    L.param = io.param.load('../config/config_insitu_DM.yaml').Params

    L.param.option.insitu.zernike_index=[j]
    L.param.option.insitu.zernike_coeff=[int(nl[2])]

    L.param.datapath = foldername
    L.param.savename = L.param.datapath + 'test1'
    images = L.load_data()
    
    L.getpsfclass()
    dataobj = L.prep_data(images)
    resfile = L.iterlearn_psf(dataobj,time=0)

#%%
C = []
for foldername in folderlist:
    nl = foldername.split('zernike_')[1].split('_')
    j = util.nl2noll(int(nl[0]),int(nl[1]))-1
    resfile = foldername + 'test_insitu_single.h5'
    f,p = io.h5.load(resfile)
    C.append(f.res.zernike_coeff[1][j])
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




# %%
cor = f.rois.cor
frames = f.rois.fileID
I_model = f.res.I_model
fig = plt.figure(figsize=[6,6])
plt.plot(np.sum(I_model,axis=(-1,-2)).transpose())
plt.xlabel('z')
plt.ylabel('Inorm')