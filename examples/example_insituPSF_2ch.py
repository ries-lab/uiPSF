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
L.param = io.param.load('../config/config_insitu_2ch.yaml').Params
L.param.datapath = maindatadir + L.param.datapath
L.param.savename = L.param.datapath + L.param.savename
images = L.load_data()

#%%
L.getpsfclass()
dataobj = L.prep_data(images)

#%%
resfile = L.iterlearn_psf(dataobj,time=0)

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

# %%
psf_data = f.rois.psf_data
psf_fit = f.rois.psf_fit
Nchannel = psf_data.shape[0]


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

    aperture=np.float32(np.abs(f.res.channel0.pupil)>0)
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
    plt.plot(f.res.imgcenter[0],f.res.imgcenter[1],'o')

if hasattr(f.res.channel0,'sigma'):
    for i in range(Nchannel):
        print(f.res['channel'+str(i)].sigma)

# %%
for k in range(0,Nchannel):
    I_model = f.res['channel'+str(k)].I_model
    Nz = I_model.shape[0]
    zind = range(0,Nz,4)
    fig = plt.figure(figsize=[2*len(zind),2])
    for i,id in enumerate(zind):
        ax = fig.add_subplot(1,len(zind),i+1)
        plt.imshow(I_model[id],cmap='twilight')
        plt.axis('off')

plt.show()

# %%
