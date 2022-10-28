#%%
import sys
import matplotlib.pyplot as plt
import numpy as np
import glob
sys.path.append("..")
from psflearning.psflearninglib import psflearninglib
from psflearning import io


maindatadir = io.param.load('../config/config_path.yaml').main_data_dir

#%% load parameters

L = psflearninglib()
L.param = io.param.load('../config/config_insitu_DM.yaml').Params


#%%
#folder = maindatadir+r'insitu data/from Yiming/In-situ PSF learing data/high order/'
folder = maindatadir+r'insitu data\from Yiming\Tubulin\high order/'
folderlist = glob.glob(folder+'/*'+'depth'+'*/')
#%% load data
for foldername in folderlist:
    
    L.param.datapath = foldername

    L.param.keyword = 'Pos0.'
    images = L.load_data()

    L.getpsfclass()

    #% segmentation

    dataobj = L.prep_data(images)

    #% learning
    L.param.option.insitu.zernike_index=[]

    psfobj,fitter = L.learn_psf(dataobj,time=0)

    #% save file
    L.param.savename = L.param.datapath + 'psfmodel_test_z_[]_zs1_n8_seg200'
    resfile = L.save_result(psfobj,dataobj,fitter)


#%% load and show result
f,p = io.h5.load(resfile)

cor = f.rois.cor
frames = f.rois.fileID
I_model = f.res.I_model
fig = plt.figure(figsize=[16,6])
ax = fig.add_subplot(1,2,1)
plt.plot(np.sum(I_model,axis=(-1,-2)).transpose())
ax.set_xlabel('z')
ax.set_ylabel('Inorm')

pos = f.res.pos
fig = plt.figure(figsize=[16,8])
ax = fig.add_subplot(2,4,1)
plt.plot(pos[:,1]-cor[:,0],'.')
plt.title('y')
ax = fig.add_subplot(2,4,2)
plt.plot(pos[:,2]-cor[:,1],'.')
plt.title('x')
ax = fig.add_subplot(2,4,3)
plt.plot(pos[:,0],'.')
plt.title('z')
ax = fig.add_subplot(2,4,5)
plt.plot(f.res.intensity,'.')
plt.title('phton')
ax = fig.add_subplot(2,4,6)
plt.plot(f.res.bg,'.')
plt.title('background')
# %
I2 = I_model
cc = I2.shape[-1]//2
fig = plt.figure(figsize=[8,8])
ax = fig.add_subplot(2,2,1)
plt.imshow(I2[:,cc,:])
plt.title('psf')
aperture=np.float32(psfobj.aperture)
Zk = f.res.zernike_polynomial


pupil_mag = np.sum(Zk*f.res.zernike_coeff[0].reshape((-1,1,1)),axis=0)*aperture
pupil_phase = np.sum(Zk[4:]*f.res.zernike_coeff[1][4:].reshape((-1,1,1)),axis=0)*aperture

ax = fig.add_subplot(2,2,3)

plt.imshow(pupil_mag)
plt.title('pupil magnitude')
ax = fig.add_subplot(2,2,4)

plt.imshow(pupil_phase)
plt.title('pupil phase')
plt.show()
#

plt.plot(f.res.zernike_coeff.transpose(),'.-')
plt.legend(['magnitude','phase'])
plt.xlabel('zernike polynomial')
plt.ylabel('coefficient')
plt.show()


Nz = I_model.shape[0]
zind = range(0,Nz,4)
fig = plt.figure(figsize=[3*len(zind),3])
for i,id in enumerate(zind):
    ax = fig.add_subplot(1,len(zind),i+1)
    plt.imshow(I_model[id],cmap='twilight')
    plt.axis('off')

#%%
plt.imshow(Zk[14]*aperture,cmap='viridis')
#%%
zf = pos[:,0]*p.pixel_size.z*1e3
plt.plot(frames,zf,'.')
plt.xlabel('frames')
plt.ylabel('learned z (nm)')


