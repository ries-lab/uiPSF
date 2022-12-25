# %% imports
import sys
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
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
L.param['datapath'] = maindatadir + r'bead data\170608_YL_beads_glass4_M2/'

#L.param['datapath'] = r'D:\Sheng\data\12-08-2021 bead bottom\40nm/'
L.param['keyword'] = 'bead'
L.param['subfolder'] = 'bead'
L.param['format'] = '.tif'
L.param['channeltype'] = 'single'
L.param['gain'] = 4.55
L.param['ccd_offset'] = 400
images = L.load_data()

#%%
L.param['PSFtype'] = 'pupil_vector'
L.getpsfclass()

#%%
L.param['pixelsize_x'] =  0.117
L.param['pixelsize_y'] = 0.127
L.param['pixelsize_z'] = 0.05
L.param['roi_size'] = [41,41]
L.param['plotall'] = True
L.param['max_bead_number'] = 10
L.param['FOV']['x_center'] = 152
L.param['FOV']['y_center'] = 141
L.param['FOV']['radius'] = 80
L.param['FOV']['z_start'] = 10
L.param['FOV']['z_end'] = -10
L.param['FOV']['z_step'] = 5
L.param['peak_height'] = 0.1

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
L.param['rej_threshold']['bias_z'] = 3
L.param['rej_threshold']['mse'] = 3
L.param['rej_threshold']['photon'] = 0.3

psfobj,fitter = L.learn_psf(dataobj,time=0)

#%%
#locres = fitter.localize(L.learning_result,'single',initz=[-6,-3,0,3,6],start_time=0)

#%%
L.param['savename'] = L.param['datapath'] + 'psfmodel_test'
resfile, res_dict, loc_dict, rois_dict = L.save_result(psfobj,dataobj,fitter)



#%%
psf_data = rois_dict['psf_data']
psf_fit = rois_dict['psf_fit']
I_model = res_dict['I_model']


np.set_printoptions(precision=4,suppress=True)
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
ind1 = 5

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
#%%
ind1 = 9
fig = plt.figure(figsize=[12,6])
pupil = res_dict['pupil']/res_dict['apodization']


ax = fig.add_subplot(1,2,1)

plt.imshow(np.abs(pupil))
plt.title('pupil magnitude')
ax = fig.add_subplot(1,2,2)

plt.imshow(np.angle(pupil))
plt.title('pupil phase')
plt.show()
#%%
fig = plt.figure(figsize=[12,6])
cind = [4,5,6,7,10,11,12,15,16,28,29]

plt.plot(res_dict['zernike_coeff'][1].transpose(),'k',alpha=1)
#plt.plot(cind,res_dict['zernike_coeff'][1,0,cind],'ro')
plt.xlabel('zernike polynomial')
plt.ylabel('coefficient')
plt.show()

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
ax.set_ylim([-40,40])
ax = fig.add_subplot(1,3,2)
plt.plot(loc_dict['loc']['y'].transpose()*py,'k',alpha=0.1)
plt.plot(loc_dict['loc']['x'][0]*0.0,'r')
ax.set_ylabel('y bias (nm)')
ax.set_ylim([-40,40])
ax = fig.add_subplot(1,3,3)
plt.plot(np.transpose(loc_dict['loc']['z']-np.linspace(0,Nz-1,Nz))*pz,'k',alpha=0.1)
plt.plot(loc_dict['loc']['x'][0]*0.0,'r')
ax.set_ylabel('z bias (nm)')
ax.set_ylim([-40,40])

#%%
ind1 = 4
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
modelfile = r'Z:\2017\17_06\170608_YL_beads_glass4_M2\psfmodel_LL1_pupil_vector_single.h5'
f = h5.File(modelfile, 'r')
psf_model = f['res']['I_model']
loc = f['locres']['loc']

# %%
