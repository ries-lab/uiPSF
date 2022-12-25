# %% imports
import sys
import matplotlib.pyplot as plt
import numpy as np


sys.path.append("..")
from psflearning.psflearninglib import psflearninglib
from psflearning import io

# only for testing, easier to switch between win and linux
maindatadir = io.param.load('../config/config_path.yaml').main_data_dir 
#%% load parameters
L = psflearninglib()
L.param = io.param.load('../config/config_zernike_2ch.yaml').Params
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
np.set_printoptions(precision=4,suppress=True)
print(f.res.T)
print(f.res.channel0.offset)
cor = f.rois.cor[0]
pos = f.res.channel0.pos
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
plt.plot(f.res.channel0.intensity.transpose())
ax.set_title('intensity')
ax = fig.add_subplot(2,4,6)
plt.plot(f.res.channel0.bg)
ax.set_title('background')
ax = fig.add_subplot(2,4,7)
plt.plot(f.res.channel0.drift_rate)
ax.set_title('drift rate')

# %%
psf_data = f.rois.psf_data
psf_fit = f.rois.psf_fit
Nchannel = psf_data.shape[0]
ind2 = 1
for ch in range(0,Nchannel):
    im1 = psf_data[ch,ind2]
    im2 = psf_fit[ch,ind2]
    Nz = im1.shape[0]
    zind = range(1,Nz,4)
    fig = plt.figure(figsize=[3*len(zind),6])
    for i,id in enumerate(zind):
        ax = fig.add_subplot(2,len(zind),i+1)
        plt.imshow(im1[id],cmap='twilight')
        plt.axis('off')
        ax = fig.add_subplot(2,len(zind),i+1+len(zind))
        plt.imshow(im2[id],cmap='twilight')
        plt.axis('off')

#%%
ch=1
imavg = psf_data[ch,ind2]
imavg1 = psf_fit[ch,ind2]
cc = psf_data.shape[-1]//2
fig = plt.figure(figsize=[12,6])
ax = fig.add_subplot(1,2,1)
plt.plot(imavg1[:,cc,:])
plt.plot(imavg[:,cc,:],'.')

ax = fig.add_subplot(1,2,2)
plt.plot(imavg[:,cc,:],'.')
plt.plot(imavg1[:,cc,:])
plt.show()

#%%
if hasattr(f.res.channel0,'pupil'):
    fig = plt.figure(figsize=[4*Nchannel,4])
    for i in range(0,Nchannel):
        ax = fig.add_subplot(1,Nchannel,i+1)
        pupil_phase = np.angle(f.res['channel'+str(i)].pupil)
        plt.imshow(pupil_phase)
        plt.axis('off')
        plt.title('pupil phase' + str(i))
    

    fig = plt.figure(figsize=[4*Nchannel,4])
    for i in range(0,Nchannel):
        ax = fig.add_subplot(1,Nchannel,i+1)
        pupil_mag = np.abs(f.res['channel'+str(i)].pupil)
        plt.imshow(pupil_mag)
        plt.axis('off')
        plt.title('pupil magnitude' + str(i))

#
if hasattr(f.res.channel0,'zernike_coeff'):
    fig = plt.figure(figsize=[4*Nchannel,4])
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    for i in range(0,Nchannel):
    
        ax1.plot(f.res['channel'+str(i)].zernike_coeff[0],'.-')    
        ax2.plot(f.res['channel'+str(i)].zernike_coeff[1],'.-')

    aperture=np.float32(psfobj.sub_psfs[0].aperture)
    Zk = f.res.channel0.zernike_polynomial
    fig = plt.figure(figsize=[4*Nchannel,4])
    for i in range(0,Nchannel):
        ax = fig.add_subplot(1,Nchannel,i+1)
        pupil_phase = np.sum(Zk[4:]*f.res['channel'+str(i)].zernike_coeff[1][4:].reshape((-1,1,1)),axis=0)*aperture
        plt.imshow(pupil_phase)
        plt.axis('off')
    plt.show()

    fig = plt.figure(figsize=[4*Nchannel,4])
    for i in range(0,Nchannel):
        ax = fig.add_subplot(1,Nchannel,i+1)
        pupil_mag = np.sum(Zk*f.res['channel'+str(i)].zernike_coeff[0].reshape((-1,1,1)),axis=0)*aperture
        plt.imshow(pupil_mag,)
        plt.axis('off')

    plt.show()
    #plt.colorbar(orientation='vertical')


# %%
px = p.pixel_size.x*1e3
py = p.pixel_size.y*1e3
pz = p.pixel_size.z*1e3
Nz = f.locres.loc.z.shape[1]
fig = plt.figure(figsize=[16,4])
ax = fig.add_subplot(1,3,1)
plt.plot(f.locres.loc.x.transpose()*px,'k',alpha=0.1)
plt.plot(f.locres.loc.x[0]*0.0,'r')
ax.set_ylabel('x bias (nm)')
ax.set_ylim([-40,40])
ax = fig.add_subplot(1,3,2)
plt.plot(f.locres.loc.y.transpose()*py,'k',alpha=0.1)
plt.plot(f.locres.loc.y[0]*0.0,'r')
ax.set_ylabel('y bias (nm)')
ax.set_ylim([-40,40])
ax = fig.add_subplot(1,3,3)
plt.plot(np.transpose(f.locres.loc.z-np.linspace(0,Nz-1,Nz))*pz,'k',alpha=0.1)
plt.plot(f.locres.loc.z[0]*0.0,'r')
ax.set_ylabel('z bias (nm)')
ax.set_ylim([-40,40])


plt.show()


# %%
np.set_printoptions(precision=4,suppress=True)
print(f.res.T)
ref_pos = f.res.channel0.pos
dxy = f.res.xyshift 
fig = plt.figure(figsize=[4*Nchannel,4])

for i in range(0,Nchannel):
    pos = f.res['channel'+str(i)].pos
    ax = fig.add_subplot(1,Nchannel,i+1)
    plt.plot(ref_pos[:,1],ref_pos[:,2],'.')
    plt.plot(pos[:,1]-dxy[i][0],pos[:,2]-dxy[i][1],'x')
    plt.plot(fitter.psf.imgcenter[0],fitter.psf.imgcenter[1],'o')

if hasattr(f.res.channel0,'sigma'):
    for i in range(Nchannel):
        print(f.res['channel'+str(i)].sigma)

# %%
