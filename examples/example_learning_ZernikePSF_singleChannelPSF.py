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
paramfile = r'params_default.json'
L = psflearninglib()
L.getparam(paramfile)


#%%
#L.param['datapath']= maindatadir + r'insitu data\from Yiming\In-situ PSF learing data\DMO1.2umNPC\PSF_DMO1.2_+-1um_10nm_2/'
L.param['datapath']= maindatadir + r'insitu data\from Yiming\In-situ PSF learing data\DMO6umNPC\PSF_DMO6_alpha30_+-3umstep50nm_5/'
#L.param['datapath'] = maindatadir + 'bead data/211207_SL_bead_3D_M2/40nm_bead_50nm/'
#L.param['datapath'] = r'D:\Sheng\data\12-08-2021 bead bottom\40nm/'
#L.param['datapath'] = maindatadir + r'bead data\01-04-2022 bead\40nm_top/'
#L.param['datapath'] = maindatadir + r'\insitu data\210122_Ulf_1C3D_M2\beadstacks/'

L.param['keyword'] = 'DMO'
L.param['subfolder'] = ''
L.param['format'] = '.tif'
L.param['channeltype'] = 'single'
L.param['gain'] = 0.47
L.param['ccd_offset'] = 100
images = L.load_data()

#%%
L.param['PSFtype'] = 'zernike'
L.getpsfclass()

#%%
L.param['pixelsize_x'] =  0.108
L.param['pixelsize_y'] = 0.108
L.param['pixelsize_z'] = 0.05
L.param['roi_size'] = [41,41]
L.param['gaus_sigma'] = [8,8]
L.param['peak_height'] =  0.2
L.param['plotall'] = True
L.param['max_bead_number'] = 50
L.param['FOV']['z_step'] = 1

dataobj = L.prep_data(images)

#%%
L.param['iteration'] = 200
L.param['vary_photon'] = True
L.param['loss_weight']['smooth'] = 0.00
L.param['option_params']['emission_wavelength'] = 0.67
L.param['option_params']['RI_imm'] = 1.405
L.param['option_params']['RI_med'] = 1.405
L.param['option_params']['NA'] = 1.35
L.param['option_params']['n_max'] = 8
L.param['option_params']['const_pupilmag'] = False
L.param['option_params']['with_apoid'] = False
L.param['rej_threshold']['bias_z'] = 5
L.param['rej_threshold']['mse'] = 3
L.param['rej_threshold']['photon'] = 5

psfobj,fitter = L.learn_psf(dataobj,time=0)

#%%
L.param['savename'] = L.param['datapath'] + 'psfmodel_LL_test'
resfile, res_dict, loc_dict, rois_dict = L.save_result(psfobj,dataobj,fitter)

# %% show results

cor = rois_dict['cor']
ref_pos1 = res_dict['pos']
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
plt.plot(res_dict['intensity'].transpose())
plt.title('phton')
ax = fig.add_subplot(2,4,6)
plt.plot(res_dict['bg'])
plt.title('background')
# %%
psf_data = rois_dict['psf_data']
psf_fit = rois_dict['psf_fit']

ind1 = 1
im1 = psf_data[ind1]
im2 = psf_fit[ind1]
Nz = im1.shape[0]
zind = range(0,Nz,12)
fig = plt.figure(figsize=[3*len(zind),6])
for i,id in enumerate(zind):
    ax = fig.add_subplot(2,len(zind),i+1)
    plt.imshow(im1[id],cmap='twilight')
    plt.axis('off')
    ax = fig.add_subplot(2,len(zind),i+1+len(zind))
    plt.imshow(im2[id],cmap='twilight')
    plt.axis('off')
#%%
fig = plt.figure(figsize=[12,6])
pupil = res_dict['pupil']

ax = fig.add_subplot(1,2,1)

plt.imshow(np.abs(pupil))
plt.title('pupil magnitude')
ax = fig.add_subplot(1,2,2)

plt.imshow(np.angle(pupil))
plt.title('pupil phase')
plt.show()
#%%
fig = plt.figure(figsize=[12,6])

plt.plot(res_dict['zernike_coeff'].transpose(),'.-')
plt.xlabel('zernike polynomial')
plt.ylabel('coefficient')
plt.legend(['pupil magnitude','pupil phase'])
plt.show()
#%%
aperture=np.float32(psfobj.aperture)
Zk = res_dict['zernike_polynomial']

pupil_mag = np.sum(Zk*res_dict['zernike_coeff'][0].reshape((-1,1,1)),axis=0)*aperture
pupil_phase = np.sum(Zk[4:]*res_dict['zernike_coeff'][1][4:].reshape((-1,1,1)),axis=0)*aperture


fig = plt.figure(figsize=[12,6])

ax = fig.add_subplot(1,2,1)
plt.imshow(pupil_mag)
#plt.colorbar(orientation='vertical')
ax = fig.add_subplot(1,2,2)
plt.imshow(pupil_phase)
#plt.colorbar(orientation='vertical')
plt.show()
#

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
# %%
N1 = 21
fig = plt.figure(figsize=[3*N1,3])
for i,zk in enumerate(Zk[0:N1]):

    ax = fig.add_subplot(1,N1,i+1)
    plt.imshow(zk*aperture,cmap='viridis')
    #plt.colorbar(orientation='horizontal')
    plt.axis('off')






















