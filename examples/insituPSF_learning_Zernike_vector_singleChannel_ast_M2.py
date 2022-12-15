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
L.param = io.param.load('../config/config_insitu_M2.yaml').Params


#%%
images = L.load_data()
L.getpsfclass()

#%%
dataobj = L.prep_data(images)

#%%
iterN = L.param.option.insitu.repeat
resfile = L.iterlearn_psf(dataobj,iterationN=iterN,time=0)
#%%
#psfobj,fitter = L.learn_psf(dataobj,time=0)
#resfile = L.save_result(psfobj,dataobj,fitter)

#%%
f,p = io.h5.load(resfile)

cor = f.rois.cor
frames = f.rois.fileID
I_model = f.res.I_model
fig = plt.figure(figsize=[6,6])
plt.plot(np.sum(I_model,axis=(-1,-2)).transpose())
plt.xlabel('z')
plt.ylabel('Inorm')

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
aperture=np.float32(np.abs(f.res.pupil)>0)
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


if hasattr(f.res,'zernike_coeff'):
    fig = plt.figure(figsize=[10,6])
    plt.plot(f.res.zernike_coeff.transpose(),'.-')
    plt.legend(['magnitude','phase'])
    plt.xlabel('zernike polynomial')
    plt.ylabel('coefficient')



# %%
Nz = I_model.shape[0]
zind = range(0,Nz,4)
fig = plt.figure(figsize=[2*len(zind),2])
for i,id in enumerate(zind):
    ax = fig.add_subplot(1,len(zind),i+1)
    plt.imshow(I_model[id],cmap='twilight')
    plt.axis('off')

plt.show()



   







