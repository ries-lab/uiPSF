#%%
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("..")
from psflearning.psflearninglib import psflearninglib
from psflearning import io

maindatadir = io.param.load('../config/config_path.yaml').main_data_dir

#%% load parameters

L = psflearninglib()
L.param = io.param.load('../config/config_insitu_tetrapod_yoav.yaml').Params
#%% load bead model
psffile = maindatadir + r'\insitu data\from Yoav\3D STORM TP\bead zstack/Tetra_psfmodel_pupil_vector_single.h5'
f0,p0 = io.h5.load(psffile)

I_init=f0.res.I_model
bead_data = f0.rois.psf_data

#%%

L.param.datapath = maindatadir+r'insitu data\from Yoav\3D STORM TP\converted/'
L.param.keyword = 'data*'
images = L.load_data()
L.getpsfclass()

#%%
L.param.option.model.init_pupil_file = []
L.param.roi.peak_height = 0.3
L.param.option.insitu.min_photon = 0.7
L.param.option.insitu.stage_pos = 3.5
for nn in range(0,5):
    
    dataobj = L.prep_data(images)

    #
    
    L.param.option.insitu.zernike_index=[12]
    L.param.option.insitu.zernike_coeff=[-2]
    L.param.loss_weight.smooth = 0.000
    psfobj,fitter = L.learn_psf(dataobj,time=0)

    #
    L.param.savename = L.param.datapath + 'psfmodel_loop'
    resfile = L.save_result(psfobj,dataobj,fitter)
    L.param.option.model.init_pupil_file = resfile
    L.param.option.insitu.min_photon = max([0.7-nn*0.1,0.4])
    
    #L.param.roi.peak_height = max([0.5-0.1*nn,0.3])
    #
    f,p = io.h5.load(resfile)
    I_model = f.res.I_model
    Nz = I_model.shape[0]
    zind = range(0,Nz,4)
    fig = plt.figure(figsize=[3*len(zind),3])
    for i,id in enumerate(zind):
        ax = fig.add_subplot(1,len(zind),i+1)
        plt.imshow(I_model[id],cmap='twilight')
        plt.axis('off')

    L.param.option.insitu.stage_pos = float(f.res.stagepos)
#%%
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
if hasattr(f.res,'zernike_coeff'):
    Zk = f.res.zernike_polynomial


    pupil_mag = np.sum(Zk*f.res.zernike_coeff[0].reshape((-1,1,1)),axis=0)*aperture
    pupil_phase = np.sum(Zk[4:]*f.res.zernike_coeff[1][4:].reshape((-1,1,1)),axis=0)*aperture

    

else:
    pupil_mag = np.abs(f.res.pupil)
    pupil_phase = np.angle(f.res.pupil)
    #pupil_phase = f.res.pupil_real[0]
    #pupil_phase = f.res.pupil_real[1]*aperture
ax = fig.add_subplot(2,2,3)

plt.imshow(pupil_mag)
plt.title('pupil magnitude')
ax = fig.add_subplot(2,2,4)

plt.imshow(pupil_phase)
plt.title('pupil phase')
plt.show()
if hasattr(f.res,'zernike_coeff'):
    plt.plot(f.res.zernike_coeff.transpose(),'.-')
    plt.legend(['magnitude','phase'])
    plt.xlabel('zernike polynomial')
    plt.ylabel('coefficient')

#%%
#plt.imshow(Zk[12]*aperture,cmap='viridis')
#%%
zf = pos[:,0]*p.pixel_size.z*1e3
plt.plot(frames,zf,'.')
plt.xlabel('frames')
plt.ylabel('learned z (nm)')

# %%
resfile1 = r'E:\EMBL files\data for PSF learning\insitu data\from Yiming\In-situ PSF learing data\DMO6umNPC\psfmodel__insitu_pupil_single.h5'
f1,p1 = io.h5.load(resfile1)
#I_model = f1.res.I_model

I_init1 = I_init[0:,0:,0:]
Nz = I_model.shape[0]
zind = range(0,Nz,4)
fig = plt.figure(figsize=[3*len(zind),6])
for i,id in enumerate(zind):
    ax = fig.add_subplot(2,len(zind),i+1)
    plt.imshow(I_init1[id],cmap='twilight')
    plt.axis('off')
    ax = fig.add_subplot(2,len(zind),i+1+len(zind))
    plt.imshow(I_model[id],cmap='twilight')
    plt.axis('off')





   








# %% issues
# 1. optimize zernike coefficient sequentially, e.g. optimize order from 1 to 5, then from 6 to 9
# 2. SMLM data has small z range, sampling in z is not uniform
# 3. large datasets, optimize in batches
# 4. pupil magnitude has to be cicular symmetric, otherwise, causing a lot fluctuation in learned coefficient
# 5. initial PSF model, it can be obtained by collecting one or two bead data, 
#    but for thick sample, the initial model might need to include index mismatch aberration
# 6. start with astigmatism PSF model