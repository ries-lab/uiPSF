#%%
import sys
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import json
sys.path.append("..")
from psflearning.psflearninglib import psflearninglib
from psflearning.learning import localizationlib
from psflearning import io

cfg = io.param.load('../config/config_insitu_embl.yaml')

#%% load parameters
#paramfile = r'params_default.json'
L = psflearninglib()
L.param = cfg.Params
#psffile = maindatadir+'211207_SL_bead_3D_M2/40nm_bead_50nm/psfmodel_voxel_single.h5'
#psffile = maindatadir + r'\insitu data\210122_Ulf_1C3D_M2\beadstacks/psfmodel_voxel_single.h5'

#f = h5.File(psffile, 'r')
#I_init=np.array(f.get('res/I_model')).astype(np.float32)
#bead_data = np.array(f.get('rois/psf_data')).astype(np.float32)
#params = json.loads(f.attrs['params'])


#%%
#L.param.datapath = maindatadir+'190910_u2os_course_96_WGA_3D_M2/01_191009_u2os_course_96_WGA_3D_ritu_1/'

L.param.datapath = 'E:\Lucas\Bottom_cellMem24h_Pos0_Bottom_cellMem24h_Loc_1\Pos0_Bottom_cellMem24h_Loc_1/'

L.param['keyword'] = 'Bottom'
L.param['subfolder'] = ''
L.param['format'] = '.tif'
L.param['channeltype'] = 'single'
L.param['datatype'] = 'smlm'
L.param['gain'] = 1/5
L.param['ccd_offset'] = 400
L.param['frame_range'] = [1000,4000]# start and end frames in SMLM data
L.param['PSFtype'] = 'insitu'
images = L.load_data()

L.getpsfclass()

#%%
L.param['pixelsize_x'] =  0.127
L.param['pixelsize_y'] = 0.127
L.param['pixelsize_z'] = 0.05 # this value can be arbiturary, currently I'm setting it the same as in the initial PSF model
L.param['roi_size'] = [15,15]
L.param['peak_height'] =  0.2
L.param['plotall'] = True
L.param['FOV']['x_center'] = 100
L.param['FOV']['y_center'] = 110
L.param['FOV']['radius'] = 80
dataobj = L.prep_data(images)

#%%
L.param['iteration'] = 200
L.param['option_params']['emission_wavelength'] = 0.68
L.param['option_params']['RI_imm'] = 1.516
L.param['option_params']['RI_med'] = 1.335
L.param['option_params']['NA'] = 1.43
L.param['option_params']['n_max'] = 5
L.param['option_params']['const_pupilmag'] = True
L.param['option_params']['symmetric_mag'] = True
L.param['option_params']['with_apoid'] = True
L.param['option_params']['stage_pos'] =  0.4 # um
L.param['rej_threshold']['bias_z'] = 1 # here is the quantile in x and y
L.param['rej_threshold']['mse'] = 0.8 # quantile in mse
L.param['option_params']['init_pupil_file'] = ''
L.param['option_params']['min_photon'] = 0.4
L.param['option_params']['partition_data'] = True
L.param['option_params']['partition_size'] = [11,200] # [segment #, # of psf per segment]
L.param['batch_size'] = 2000
L.param['option_params']['zernike_index'] = [5,8]
L.param['option_params']['zernike_coeff'] = [-0.5,0.0]
psfobj,fitter = L.learn_psf(dataobj,time=0)

#%%
L.param['savename'] = L.param['datapath'] + 'psfmodel_test'
resfile, res_dict, loc_dict, rois_dict = L.save_result(psfobj,dataobj,fitter)


#%% localize bead data with insitu PSF
#dll = localizationlib(usecuda=True)
#data = bead_data
#psf_model = res_dict['I_model']
#locres = dll.loc_ast(np.flip(data,axis=1),psf_model,params['pixelsize_z'])
#%% localize bead data with init PSF 
#dll = localizationlib(usecuda=True)
#data = bead_data
#psf_model = I_init
#locres = dll.loc_ast(data,psf_model,params['pixelsize_z'])
#%%
cor = rois_dict['cor']
frames = rois_dict['fileID']

I_model = res_dict['I_model']
fig = plt.figure(figsize=[16,6])
ax = fig.add_subplot(1,2,1)
plt.plot(np.sum(I_model,axis=(-1,-2)).transpose())
ax.set_xlabel('z')
ax.set_ylabel('Inorm')

ref_cor = cor
ref_pos1 = res_dict['pos']
fig = plt.figure(figsize=[16,8])
ax = fig.add_subplot(2,4,1)
plt.plot(ref_pos1[:,1]-ref_cor[:,0],'.')
plt.title('y')
ax = fig.add_subplot(2,4,2)
plt.plot(ref_pos1[:,2]-ref_cor[:,1],'.')
plt.title('x')
ax = fig.add_subplot(2,4,3)
plt.plot(ref_pos1[:,0],'.')
plt.title('z')
ax = fig.add_subplot(2,4,5)
plt.plot(res_dict['intensity'],'.')
plt.title('phton')
ax = fig.add_subplot(2,4,6)
plt.plot(res_dict['bg'],'.')
plt.title('background')
# %
I2 = I_model
cc = I2.shape[-1]//2
fig = plt.figure(figsize=[8,8])
ax = fig.add_subplot(2,2,1)
plt.imshow(I2[:,cc,:])
plt.title('psf')
aperture=np.float32(psfobj.aperture)
Zk = res_dict['zernike_polynomial']
#pupil_mag = np.abs(np.sum(Zk*res_dict['zernike_coeff'][0],axis=0))*np.real(aperture)
#pupil_phase = np.sum(Zk[4:]*res_dict['zernike_coeff'][1][4:],axis=0)*np.real(aperture)

pupil_mag = np.sum(Zk*res_dict['zernike_coeff'][0].reshape((-1,1,1)),axis=0)*aperture
pupil_phase = np.sum(Zk[4:]*res_dict['zernike_coeff'][1][4:].reshape((-1,1,1)),axis=0)*aperture

ax = fig.add_subplot(2,2,3)

plt.imshow(pupil_mag)
plt.title('pupil magnitude')
ax = fig.add_subplot(2,2,4)

plt.imshow(pupil_phase)
plt.title('pupil phase')
plt.show()
#

plt.plot(res_dict['zernike_coeff'].transpose())
plt.legend(['magnitude','phase'])
plt.xlabel('zernike polynomial')
plt.ylabel('coefficient')
plt.show()

#%%
zf = res_dict['pos'][:,0]*params['pixelsize_z']*1e3
plt.plot(frames,zf,'.')
plt.xlabel('frames')
plt.ylabel('learned z (nm)')
#%% compare bead and insitu PSF

I_init1 = np.flip(I_init[:,8:-8,8:-8],axis=0)
Nz = I_model.shape[0]
zind = range(0,Nz-1,4)
fig = plt.figure(figsize=[3*len(zind),6])
for i,id in enumerate(zind):
    ax = fig.add_subplot(2,len(zind),i+1)
    plt.imshow(I_init1[id],cmap='twilight')
    plt.axis('off')
    ax = fig.add_subplot(2,len(zind),i+1+len(zind))
    plt.imshow(I_model[id],cmap='twilight')
    plt.axis('off')





   








# %% issues
# 1. optimize zernike coefficient sequentially, e.g. optimize order from 1 to 5, then from 6 to 9
# 2. SMLM data has small z range, sampling in z is not uniform
# 3. large datasets, optimize in batches
# 4. pupil magnitude has to be cicular symmetric, otherwise, causing a lot fluctuation in learned coefficient
# 5. initial PSF model, it can be obtained by collecting one or two bead data, 
#    but for thick sample, the initial model might need to include index mismatch aberration
# 6. start with astigmatism PSF model