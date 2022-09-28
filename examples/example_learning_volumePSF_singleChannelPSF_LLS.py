# %% imports
from distutils import core
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
#L.param['datapath'] = r'D:\Sheng\data\LLS\YuShi/'
L.param['datapath'] = maindatadir + r'LLS_Data/'
L.param['keyword'] = 'scan'
L.param['subfolder'] = ''
L.param['format'] = '.tiff'
L.param['channeltype'] = 'single'
L.param['gain'] = 1/2.27
L.param['ccd_offset'] = 100
images = L.load_data()

#%%
L.param['PSFtype'] = 'voxel'
L.getpsfclass()

#%%
L.param['pixelsize_x'] =  0.104
L.param['pixelsize_y'] = 0.104
L.param['pixelsize_z'] = 0.0578
L.param['roi_size'] = [37,27,27]
L.param['gaus_sigma'] = [6,2,2]
L.param['max_kernel'] = [9,3,3]
L.param['plotall'] = True
L.param['max_bead_number'] = 200
L.param['FOV']['x_center'] = 0
L.param['FOV']['y_center'] = 0
L.param['FOV']['radius'] = 0
L.param['FOV']['z_step'] = 2
L.param['peak_height'] = 0.2

L.param['skew_const'] = [0,-0.7845]

dataobj = L.prep_data(images)

#%%
L.param['iteration'] = 100
L.param['vary_photon'] = False
L.param['estimate_drift'] = True
L.param['loss_weight']['smooth'] = 1
L.param['loss_weight']['mse1'] = 1
L.param['loss_weight']['mse2'] = 1
L.param['option_params']['emission_wavelength'] = 0.6
L.param['option_params']['RI_imm'] = 1.334
L.param['option_params']['RI_med'] = 1.334
L.param['option_params']['RI_cov'] = 1.334
L.param['option_params']['NA'] = 1.1
L.param['option_params']['n_max'] = 10
L.param['option_params']['const_pupilmag'] = False
L.param['option_params']['with_apoid'] = False
L.param['rej_threshold']['bias_z'] = 5
L.param['rej_threshold']['mse'] = 3
L.param['rej_threshold']['photon'] = 1.5

psfobj,fitter = L.learn_psf(dataobj,time=0)

#%%
loc_FD = L.localize_FD(fitter)
#locres = fitter.localize(L.learning_result,'single',initz=[-2,-1,0,1,2],start_time=0)


#%%
L.param['savename'] = L.param['datapath'] + 'psfmodel_constI1'
resfile, res_dict, loc_dict, rois_dict = L.save_result(psfobj,dataobj,fitter)

#%%
psf_data = rois_dict['psf_data']
psf_fit = rois_dict['psf_fit']

I_model = res_dict['I_model']

# %%
Nz = psf_data.shape[-3]
cor = rois_dict['cor']
ref_pos1 = res_dict['pos']
fig = plt.figure(figsize=[16,8])
ax = fig.add_subplot(2,4,1)
plt.plot(ref_pos1[:,1]-cor[:,1])
plt.title('y')
ax = fig.add_subplot(2,4,2)
plt.plot(ref_pos1[:,2]-cor[:,2])
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
ax = fig.add_subplot(2,4,7)
plt.plot(res_dict['drift_rate'])
plt.title('drift')

#%%
ind1= 36
im1 = psf_data[ind1]
im2 = psf_fit[ind1]
#im2 = I_model
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

#%%
px = L.param['pixelsize_x']*1e3
py = L.param['pixelsize_y']*1e3
pz = L.param['pixelsize_z']*1e3
Nz = loc_dict['loc']['z'].shape[1]
fig = plt.figure(figsize=[16,4])
ax = fig.add_subplot(1,3,1)
plt.plot(loc_dict['loc']['x'].transpose()*px,'k',alpha=0.1)
plt.plot(loc_dict['loc']['x'][0]*0.0,'r')
ax.set_ylabel('x bias (nm)')
ax.set_ylim([-100,100])
ax = fig.add_subplot(1,3,2)
plt.plot(loc_dict['loc']['y'].transpose()*py,'k',alpha=0.1)
plt.plot(loc_dict['loc']['x'][0]*0.0,'r')
ax.set_ylabel('y bias (nm)')
ax.set_ylim([-100,100])
ax = fig.add_subplot(1,3,3)
plt.plot(np.transpose(loc_dict['loc']['z']-np.linspace(0,Nz-1,Nz))*pz,'k',alpha=0.1)
plt.plot(loc_dict['loc']['x'][0]*0.0,'r')
ax.set_ylabel('z bias (nm)')
ax.set_ylim([-100,100])
#%%
pupil = res_dict['pupil']
aperture = np.float32(np.abs(pupil)>0)
fig = plt.figure(figsize=[12,6])

ax = fig.add_subplot(1,2,1)

plt.imshow(np.abs(pupil))
plt.title('pupil magnitude')
ax = fig.add_subplot(1,2,2)

plt.imshow(np.angle(pupil)*aperture)
plt.title('pupil phase')
plt.show()
#%%
fig = plt.figure(figsize=[12,6])
cind = np.array([4,5,6,9,10,11,21])

plt.plot(res_dict['zernike_coeff'][1].transpose(),'k',alpha=0.1)
#plt.plot(cind,res_dict['zernike_coeff'][1,0,cind],'ro')
plt.xlabel('zernike polynomial')
plt.ylabel('coefficient')
plt.show()


# %%
imsz = np.array(rois_dict['image_size'])
Zmap = res_dict['zernike_map']
Zk = res_dict['zernike_polynomial']
scale = (imsz[-2:]-1)/(np.array(Zmap.shape[-2:])-1)

fig = plt.figure(figsize=[3*len(cind),6])
for i,id in enumerate(cind):
    ax = fig.add_subplot(2,len(cind),i+1)
    #plt.imshow(Zmap[1,id],cmap='twilight',vmin=-0.05,vmax=0.5)
    plt.imshow(Zmap[0,id],cmap='twilight')
    #plt.plot(cor[:,-1]/scale[-1],cor[:,-2]/scale[-2],'ro',markersize=5)
    plt.axis('off')
    ax = fig.add_subplot(2,len(cind),i+1+len(cind))
    plt.imshow(Zk[id]*aperture,cmap='viridis')
    plt.axis('off')

# %%
I2 = I_model
cc = I2.shape[-1]//2+0
fig = plt.figure(figsize=[12,4])
ax = fig.add_subplot(1,3,1)
plt.plot(I2[:,cc,:])
ax = fig.add_subplot(1,3,2)
plt.plot(I2[:,:,cc])
ax = fig.add_subplot(1,3,3)
plt.plot(np.sum(I2,axis=(-1,-2)))

plt.show()
#
#%%
ind1 = 14
cc = psf_data.shape[-1]//2
imavg = psf_data[ind1]
imavg1 = psf_fit[ind1]

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
import h5py as h5
modelfile = r'Z:\projects\decode\LLS_Data\psfmodel_voxel_single.h5'
f = h5.File(modelfile, 'r')
psf_model = f['res']['I_model']
spline_coeff = f['locres']['coeff']
# %%
