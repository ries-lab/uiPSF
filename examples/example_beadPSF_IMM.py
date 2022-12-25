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
L.param = io.param.load('../config/config_zernike_IMM.yaml').Params
L.param.datapath = maindatadir + L.param.datapath
L.param.savename = L.param.datapath + L.param.savename
L.param.option.model.init_pupil_file =  maindatadir + L.param.option.model.init_pupil_file
#L.param.option.model.init_pupil_file = ''
images = L.load_data()

#%%
L.getpsfclass()
dataobj = L.prep_data(images)
psfobj,fitter = L.learn_psf(dataobj,time=0)
loc_FD = L.localize_FD(fitter, initz = [-4,-2,0,2,4])
resfile = L.save_result(psfobj,dataobj,fitter)

# %% show results

f,p = io.h5.load(resfile)
cor = f.rois.cor
pos = f.res.pos
fig = plt.figure(figsize=[16,8])
ax = fig.add_subplot(2,4,1)
plt.plot(pos[:,2]-cor[:,0])
plt.title('y')
ax = fig.add_subplot(2,4,2)
plt.plot(pos[:,3]-cor[:,1])
plt.title('x')
ax = fig.add_subplot(2,4,3)
plt.plot(pos[:,0])
plt.title('z stage')
ax = fig.add_subplot(2,4,4)
plt.plot(pos[:,1])
plt.title('z med')
ax = fig.add_subplot(2,4,5)
plt.plot(f.res.intensity.transpose())
plt.title('photon')
ax = fig.add_subplot(2,4,6)
plt.plot(f.res.bg)
plt.title('background')
ax = fig.add_subplot(2,4,8)
ccz = (psfobj.Zrange.shape[-3] - 1) // 2
zs0 = cor[:,0]
zs1 = pos[:,0]
plt.plot(zs0-zs1)
#plt.plot(zs1)
plt.title('z stage diff')
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

ind1 = 4
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
loc = f.locres.loc
Nz = loc.z.shape[1]
fig = plt.figure(figsize=[16,4])
ax = fig.add_subplot(1,3,1)
plt.plot(loc.x.transpose()*p.pixel_size.x*1e3,'k',alpha=0.1)
plt.plot(loc.x[0]*0.0,'r')
ax.set_ylabel('x bias (nm)')
ax = fig.add_subplot(1,3,2)
plt.plot(loc.y.transpose()*p.pixel_size.y*1e3,'k',alpha=0.1)
plt.plot(loc.y[0]*0.0,'r')
ax.set_ylabel('y bias (nm)')
ax = fig.add_subplot(1,3,3)
plt.plot(np.transpose(loc.z-np.linspace(0,Nz-1,Nz))*p.pixel_size.z*1e3,'k',alpha=0.1)
plt.plot(loc.z[0]*0.0,'r')
ax.set_ylabel('z bias (nm)')
ax.set_ylim([-40,40])


loc = f.locres.loc_FD
Nz = loc.z.shape[1]
fig = plt.figure(figsize=[16,4])
ax = fig.add_subplot(1,3,1)
plt.plot(loc.x.transpose()*p.pixel_size.x*1e3,'k',alpha=0.1)
plt.plot(loc.x[0]*0.0,'r')
ax.set_ylabel('x bias (nm)')
ax = fig.add_subplot(1,3,2)
plt.plot(loc.y.transpose()*p.pixel_size.y*1e3,'k',alpha=0.1)
plt.plot(loc.y[0]*0.0,'r')
ax.set_ylabel('y bias (nm)')
ax = fig.add_subplot(1,3,3)
plt.plot(np.transpose(loc.z-np.linspace(0,Nz-1,Nz))*p.pixel_size.z*1e3,'k',alpha=0.1)
plt.plot(loc.z[0]*0.0,'r')
ax.set_ylabel('z bias (nm)')
ax.set_ylim([-40,40])

plt.show()



# %%
