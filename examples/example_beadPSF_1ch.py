# %% imports
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("..")
from psflearning.psflearninglib import psflearninglib
from psflearning import io

maindatadir = io.param.load('../config/config_path.yaml').main_data_dir

#%% load parameters
L = psflearninglib()
L.param = io.param.load('../config/config_zernike_M2.yaml').Params
#L.param.datapath = maindatadir + L.param.datapath
#L.param.savename = L.param.datapath + L.param.savename
L.param.datapath = L.param.datapath
images = L.load_data()

#%%
L.getpsfclass()
dataobj = L.prep_data(images)
psfobj,fitter = L.learn_psf(dataobj,time=0)
resfile = L.save_result(psfobj,dataobj,fitter)

# %% show results

f,p = io.h5.load(resfile)
cor = f.rois.cor
pos = f.res.pos
fig = plt.figure(figsize=[16,8])
ax = fig.add_subplot(2,4,1)
plt.plot(pos[:,1]-cor[:,0])
plt.title('y')
ax = fig.add_subplot(2,4,2)
plt.plot(pos[:,2]-cor[:,1])
plt.title('x')
ax = fig.add_subplot(2,4,3)
plt.plot(pos[:,0])
plt.title('z')
ax = fig.add_subplot(2,4,5)
plt.plot(f.res.intensity.transpose())
plt.title('photon')
ax = fig.add_subplot(2,4,6)
plt.plot(f.res.bg)
plt.title('background')
ax = fig.add_subplot(2,4,7)
plt.plot(f.res.drift_rate)
plt.title('drift')

#
if hasattr(f.res,'pupil'):
    fig = plt.figure(figsize=[12,6])
    pupil = f.res.pupil

    ax = fig.add_subplot(1,2,1)

    plt.imshow(np.abs(pupil))
    plt.title('pupil magnitude')
    ax = fig.add_subplot(1,2,2)

    plt.imshow(np.angle(pupil))
    plt.title('pupil phase')

#
if hasattr(f.res,'zernike_coeff'):
    fig = plt.figure(figsize=[12,6])

    plt.plot(f.res.zernike_coeff.transpose(),'.-')
    plt.xlabel('zernike polynomial')
    plt.ylabel('coefficient')
    plt.legend(['pupil magnitude','pupil phase'])
    

    aperture=np.float32(psfobj.aperture)
    Zk = f.res.zernike_polynomial

    pupil_mag = np.sum(Zk*f.res.zernike_coeff[0].reshape((-1,1,1)),axis=0)*aperture
    pupil_phase = np.sum(Zk[4:]*f.res.zernike_coeff[1][4:].reshape((-1,1,1)),axis=0)*aperture


    fig = plt.figure(figsize=[12,6])

    ax = fig.add_subplot(1,2,1)
    plt.imshow(pupil_mag)
    #plt.colorbar(orientation='vertical')
    ax = fig.add_subplot(1,2,2)
    plt.imshow(pupil_phase)
    #plt.colorbar(orientation='vertical')

# %%
psf_data = f.rois.psf_data
psf_fit = f.rois.psf_fit

ind1 = 1
im1 = psf_data[ind1]
im2 = psf_fit[ind1]
Nz = im1.shape[0]
zind = range(0,Nz,4)
fig = plt.figure(figsize=[3*len(zind),6])
for i,id in enumerate(zind):
    ax = fig.add_subplot(2,len(zind),i+1)
    plt.imshow(im1[id],cmap='twilight')
    plt.axis('off')
    ax = fig.add_subplot(2,len(zind),i+1+len(zind))
    plt.imshow(im2[id],cmap='twilight')
    plt.axis('off')
# %%
Nz = f.locres.loc.z.shape[1]
fig = plt.figure(figsize=[16,4])
ax = fig.add_subplot(1,3,1)
plt.plot(f.locres.loc.x.transpose()*p.pixel_size.x*1e3,'k',alpha=0.1)
plt.plot(f.locres.loc.x[0]*0.0,'r')
ax.set_ylabel('x bias (nm)')
ax = fig.add_subplot(1,3,2)
plt.plot(f.locres.loc.y.transpose()*p.pixel_size.y*1e3,'k',alpha=0.1)
plt.plot(f.locres.loc.y[0]*0.0,'r')
ax.set_ylabel('y bias (nm)')
ax = fig.add_subplot(1,3,3)
plt.plot(np.transpose(f.locres.loc.z-np.linspace(0,Nz-1,Nz))*p.pixel_size.z*1e3,'k',alpha=0.1)
plt.plot(f.locres.loc.z[0]*0.0,'r')
ax.set_ylabel('z bias (nm)')
ax.set_ylim([-40,40])


plt.show()



# %%
[0.7478614, 0.7868592] # radius=0.025, bin=3
[0.75469255, 0.7930746 ] # radius=0, bin=3
[0.8040463 , 0.83986473] # radius=0, bin=1
[0.5191372 , 0.51132196] # radius=0.025, bin=3, vector
[0.53423333, 0.5222833 ] # radius=0.025, bin=2, vector
[0.59332156, 0.5808556 ] # radius=0.025, bin=1, vector

[0.8877916 , 1.7708156 , 1.5449411 , 1.7708156 , 0.59193677, 1.8801146 ]
[2.2039149 , 0.9988498 , 1.0983567 , 0.9988498 , 1.3747756 ,0.72415626]

[0.9431401, 1.8442363, 1.6164677, 1.8442363, 0.6229577, 1.9458437] #sigma=0.05
[0.8978031, 1.7640057, 1.5396551, 1.7640057, 0.6069426, 1.869314 ] #sigma=0.1
[0.95390594, 1.7663572 , 1.5249628 , 1.7663572 , 0.649004  ,1.8163725 ]#sigma=0.2
[1.4368327, 2.4023893, 1.9870223, 2.4023893, 1.0608294, 2.3228884] #sigma=0.3
[1.5022764, 2.04701  , 1.601577 , 2.04701  , 1.1209229, 1.8646094] #sigma=0.4

# create genpsfmodel() for each class, then call in post processing