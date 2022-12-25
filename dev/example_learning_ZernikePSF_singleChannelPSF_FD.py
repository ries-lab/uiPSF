
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
#L.param['datapath'] = r'D:\Sheng\data\12-08-2021 bead bottom\40nm/'
L.param['datapath'] = maindatadir + r'bead data\220524_SL_Beads_SilOilObj_M4\bead_100nm_1/'
L.param['keyword'] = 'bead'
L.param['subfolder'] = 'bead'
L.param['format'] = '.tif'
L.param['channeltype'] = 'single'
L.param['gain'] = 1/2.27
L.param['ccd_offset'] = 100
images = L.load_data()

#%%
L.param['PSFtype'] = 'zernike_FD'
L.getpsfclass()

#%%
L.param['pixelsize_x'] =  0.123
L.param['pixelsize_y'] = 0.123
L.param['pixelsize_z'] = 0.05
L.param['roi_size'] = [31,31]
L.param['plotall'] = True
L.param['max_bead_number'] = 20
L.param['FOV']['z_start'] = 0

dataobj = L.prep_data(images)

#%%
L.param['iteration'] = 200
L.param['vary_photon'] = True
L.param['loss_weight']['smooth'] = 0.001
L.param['loss_weight']['mse2'] = 0.0
L.param['option_params']['emission_wavelength'] = 0.6
L.param['option_params']['RI_imm'] = 1.406
L.param['option_params']['RI_med'] = 1.333
L.param['option_params']['NA'] = 1.35
L.param['option_params']['n_max'] = 8
L.param['option_params']['const_pupilmag'] = False
L.param['option_params']['with_apoid'] = False
L.param['rej_threshold']['bias_z'] = 10
L.param['rej_threshold']['mse'] = 3
L.param['rej_threshold']['photon'] = 3

psfobj,fitter = L.learn_psf(dataobj,time=0)

#%%
loc_FD = L.localize_FD(fitter)
#%%
L.param['savename'] = L.param['datapath'] + 'psfmodel_LL1'
resfile, res_dict, loc_dict, rois_dict = L.save_result(psfobj,dataobj,fitter)

#%%
psf_data = rois_dict['psf_data']
psf_fit = rois_dict['psf_fit']
I_model = res_dict['I_model']
I_model_all = res_dict['I_model_all']

# %%
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
ind1 = 0

im1 = psf_data[ind1]
im2 = psf_fit[ind1]
Nz = im1.shape[0]
zind = range(0,Nz,6)
fig = plt.figure(figsize=[3*len(zind),6])
for i,id in enumerate(zind):
    ax = fig.add_subplot(2,len(zind),i+1)
    plt.imshow(im1[id],cmap='twilight')
    plt.axis('off')
    ax = fig.add_subplot(2,len(zind),i+1+len(zind))
    plt.imshow(im2[id],cmap='twilight')
    plt.axis('off')
#%%
ind1 = 1
fig = plt.figure(figsize=[12,6])
pupil = res_dict['pupil'][ind1]


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

plt.plot(res_dict['zernike_coeff'][1].transpose(),'k',alpha=0.1)
plt.plot(cind,res_dict['zernike_coeff'][1,0,cind],'ro')
plt.xlabel('zernike polynomial')
plt.ylabel('coefficient')
plt.show()


# %%
aperture = np.float32(np.abs(pupil)>0)
imsz = np.array(rois_dict['image_size'])
Zmap = res_dict['zernike_map']
Zk = res_dict['zernike_polynomial']
scale = (imsz[-2:]-1)/(np.array(Zmap.shape[-2:])-1)

fig = plt.figure(figsize=[3*len(cind),6])
for i,id in enumerate(cind):
    ax = fig.add_subplot(2,len(cind),i+1)
    #plt.imshow(Zmap[1,id],cmap='twilight',vmin=-0.05,vmax=0.5)
    plt.imshow(Zmap[1,id],cmap='twilight')
    #plt.plot(cor[:,-1]/scale[-1],cor[:,-2]/scale[-2],'ro',markersize=5)
    plt.axis('off')
    ax = fig.add_subplot(2,len(cind),i+1+len(cind))
    plt.imshow(Zk[id]*aperture,cmap='viridis')
    plt.axis('off')


# %%
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

# %%
ind = 3
loci = loc_dict['loc_FD']
locavg = loc_dict['loc']
Nz = loci['z'].shape[1]
fig = plt.figure(figsize=[16,4])
ax = fig.add_subplot(1,3,1)
plt.plot((loci['x'][ind])*px)
plt.plot((locavg['x'][ind])*px)
ax.set_ylim([-100,100])
ax.set_ylabel('x bias (nm)')
ax = fig.add_subplot(1,3,2)
plt.plot((loci['y'][ind])*py)
plt.plot((locavg['y'][ind])*py)
ax.set_ylim([-100,100])
ax.set_ylabel('y bias (nm)')
ax = fig.add_subplot(1,3,3)
plt.plot((np.squeeze(loci['z'][ind])-np.linspace(0,Nz-1,Nz))*pz)
plt.plot((np.squeeze(locavg['z'][ind])-np.linspace(0,Nz-1,Nz))*pz)
ax.set_ylabel('z bias (nm)')
plt.legend(('FD PSF','average PSF'))
ax.set_ylim([-100,100])
# %%
