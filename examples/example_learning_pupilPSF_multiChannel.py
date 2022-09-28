# %% imports
import sys
import matplotlib.pyplot as plt
import numpy as np
import json
sys.path.append("..")
from psflearning.psflearninglib import psflearninglib

with open('datadir.json','r') as file:
    maindatadir = json.load(file)['main_data_dir']

#%% load parameters
paramfile = 'params_default.json'
L = psflearninglib()
L.getparam(paramfile)

#%%
#L.param['datapath'] = maindatadir+r'bead data\01-04-2022 bead\40nm_top/'
#L.param['datapath'] = maindatadir+ r'bead data\12-17-2021 bead\40nm/'
L.param['datapath'] = maindatadir+r'bead data\220124_SL_bead_3D_2C_M2/bead4_tetraspec_50nmstep_triggerCam_10ms_SM/'


L.param['keyword'] = 'bead'
L.param['subfolder'] = 'bead'
L.param['format'] = '.tif'
L.param['channeltype'] = 'multi'
L.param['gain'] = 4.55
L.param['ccd_offset'] = 400
images = L.load_data()

#%%
L.param['PSFtype'] = 'pupil_vector'
L.getpsfclass()

#%%
L.param['pixelsize_x'] =  0.127
L.param['pixelsize_y'] = 0.127
L.param['pixelsize_z'] = 0.05
L.param['roi_size'] = [25,25]
L.param['plotall'] = True
L.param['max_bead_number'] = 30
L.param['FOV']['x_center'] = 0
L.param['FOV']['y_center'] = 0
L.param['FOV']['radius'] = 0
L.param['FOV']['z_start'] = 1

dataobj = L.prep_data(images)

#%%
L.param['iteration'] = 100
L.param['vary_photon'] = True
L.param['loss_weight']['smooth'] = 0.0
L.param['option_params']['emission_wavelength'] = 0.68
L.param['option_params']['RI_imm'] = 1.516
L.param['option_params']['RI_med'] = 1.335
L.param['option_params']['NA'] = 1.43
L.param['option_params']['n_max'] = 8
L.param['option_params']['const_pupilmag'] = False
L.param['option_params']['with_apoid'] = False
L.param['option_params']['symmetric_mag'] = False
L.param['rej_threshold']['photon'] = 0.5
psfobj,fitter = L.learn_psf(dataobj,time=0)

#%%
L.param['savename'] = L.param['datapath'] + 'psfmodel_test'
resfile, res_dict, loc_dict, rois_dict = L.save_result(psfobj,dataobj,fitter)

# %% show results

cor = rois_dict['cor'][0]
ref_pos1 = res_dict['channel0']['pos']
fig = plt.figure(figsize=[16,8])
ax = fig.add_subplot(2,4,1)
plt.plot(ref_pos1[:,1]-cor[:,0])
plt.title('y')
ax = fig.add_subplot(2,4,2)
plt.plot(ref_pos1[:,2]-cor[:,1])
plt.title('x')
ax = fig.add_subplot(2,4,3)
plt.plot(ref_pos1[:,0])
plt.title('z')
ax = fig.add_subplot(2,4,5)
plt.plot(res_dict['channel0']['intensity'].transpose())
plt.title('photon')
ax = fig.add_subplot(2,4,6)
plt.plot(res_dict['channel0']['bg'])
plt.title('background')

# %%
psf_data = rois_dict['psf_data']
psf_fit = rois_dict['psf_fit']

Nchannel = psf_data.shape[0]
ind = 0
for i in range(0,Nchannel):
    im1 = psf_data[i,ind]
    im2 = psf_fit[i,ind]
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
aperture=np.float32(psfobj.sub_psfs[0].aperture)
Zk = res_dict['channel0']['zernike_polynomial']
fig = plt.figure(figsize=[4*Nchannel,4])
for i in range(0,Nchannel):
    ax = fig.add_subplot(1,Nchannel,i+1)
    pupil_phase = np.sum(Zk[4:]*res_dict['channel'+str(i)]['zernike_coeff'][1][4:].reshape((-1,1,1)),axis=0)*aperture
    plt.imshow(pupil_phase)
    plt.axis('off')
plt.show()

fig = plt.figure(figsize=[4*Nchannel,4])
for i in range(0,Nchannel):
    ax = fig.add_subplot(1,Nchannel,i+1)
    pupil_mag = np.sum(Zk*res_dict['channel'+str(i)]['zernike_coeff'][0].reshape((-1,1,1)),axis=0)*aperture
    plt.imshow(pupil_mag)
    plt.axis('off')
plt.show()

#%%
aperture=np.float32(psfobj.sub_psfs[0].aperture)

fig = plt.figure(figsize=[4*Nchannel,4])
for i in range(0,Nchannel):
    ax = fig.add_subplot(1,Nchannel,i+1)
    pupil_phase = np.angle(res_dict['channel'+str(i)]['pupil'])*aperture
    plt.imshow(pupil_phase)
    plt.axis('off')
    plt.title('pupil phase' + str(i))
plt.show()

fig = plt.figure(figsize=[4*Nchannel,4])
for i in range(0,Nchannel):
    ax = fig.add_subplot(1,Nchannel,i+1)
    pupil_mag = np.abs(res_dict['channel'+str(i)]['pupil'])*aperture
    plt.imshow(pupil_mag)
    plt.axis('off')
    plt.title('pupil magnitude' + str(i))

plt.show()
# %%
fig = plt.figure(figsize=[16,4])
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
for i in range(0,Nchannel):
    
    ax1.plot(res_dict['channel'+str(i)]['zernike_coeff'][0],'.-')    
    ax2.plot(res_dict['channel'+str(i)]['zernike_coeff'][1],'.-')
    #ax4.plot([16,28],res_dict['channel'+str(i)]['zernike_coeff2'][1][[16,28]],'o')
    #ax2.set_ylim((-0.6,0.6))
# %%
Nz = loc_dict['loc']['z'].shape[1]
fig = plt.figure(figsize=[16,4])
ax = fig.add_subplot(1,3,1)
plt.plot(loc_dict['loc']['x'].transpose()*L.param['pixelsize_x']*1e3,'k',alpha=0.1)
plt.plot(loc_dict['loc']['x'][0]*0.0,'r')
ax.set_ylabel('x bias (nm)')
ax = fig.add_subplot(1,3,2)
plt.plot(loc_dict['loc']['y'].transpose()*L.param['pixelsize_y']*1e3,'k',alpha=0.1)
plt.plot(loc_dict['loc']['x'][0]*0.0,'r')
ax.set_ylabel('y bias (nm)')
ax = fig.add_subplot(1,3,3)
plt.plot(np.transpose(loc_dict['loc']['z']-np.linspace(0,Nz-1,Nz))*L.param['pixelsize_z']*1e3,'k',alpha=0.1)
plt.plot(loc_dict['loc']['x'][0]*0.0,'r')
ax.set_ylabel('z bias (nm)')
ax.set_ylim([-40,40])

#%%
np.set_printoptions(precision=4,suppress=True)
print(res_dict['T'])
ref_pos = res_dict['channel0']['pos']
dxy = res_dict['xyshift'] 
fig = plt.figure(figsize=[4*Nchannel,4])

for i in range(0,Nchannel):
    pos = res_dict['channel'+str(i)]['pos']
    ax = fig.add_subplot(1,Nchannel,i+1)
    plt.plot(ref_pos[:,1],ref_pos[:,2],'.')
    plt.plot(pos[:,1]-dxy[i][0],pos[:,2]-dxy[i][1],'x')
    plt.plot(fitter.psf.imgcenter[0],fitter.psf.imgcenter[1],'o')

# %%
for i in range(Nchannel):
    print(res_dict['channel'+str(i)]['sigma'])

# %%
ind1 = 2
ch = 0
cc = psf_data.shape[-1]//2
imavg = psf_data[ch,ind1]
imavg1 = psf_fit[ch,ind1]

fig = plt.figure(figsize=[12,6])
ax = fig.add_subplot(1,2,1)
plt.plot(imavg1[:,:,cc])
plt.plot(imavg[:,:,cc],'.')

ax = fig.add_subplot(1,2,2)
plt.plot(imavg[:,cc,:],'.')
plt.plot(imavg1[:,cc,:])
plt.show()

fig = plt.figure(figsize=[12,6])
ax = fig.add_subplot(1,3,1)
plt.imshow(imavg[:,cc],cmap='twilight')
plt.colorbar(orientation='vertical')
ax = fig.add_subplot(1,3,2)
plt.imshow(imavg1[:,cc],cmap='twilight')
plt.colorbar(orientation='vertical')
ax = fig.add_subplot(1,3,3)
plt.imshow(imavg1[:,:,cc]-imavg[:,:,cc],cmap='twilight')
plt.colorbar(orientation='vertical')
plt.show()

# %%
psf_data = rois_dict['psf_data']

fig = plt.figure(figsize=[3*8,3*3])
Nbead = psf_data.shape[1]
ind = 0
for i in range(0,Nbead):
    im1 = psf_data[1,i,0]
    ax = fig.add_subplot(3,8,i+1)
    plt.imshow(im1,cmap='twilight')
    plt.axis('off')

# %%
