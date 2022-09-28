#%%
import sys
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import json
sys.path.append("..")
from psflearning.psflearninglib import psflearninglib
from psflearning.learning import (localizationlib,PSFZernikeBased_vector_smlm)


with open('datadir.json','r') as file:
    maindatadir = json.load(file)['main_data_dir']

#%% load parameters
paramfile = r'params_default.json'
L = psflearninglib()
L.getparam(paramfile)
#psffile = maindatadir+'211207_SL_bead_3D_M2/40nm_bead_50nm/psfmodel_voxel_single.h5'
psffile = maindatadir + r'\insitu data\from Yiming\In-situ PSF learing data\Single molecule data for different DM models/U20S_MT_tublin_AF647_depth_200nm_6_2_zerniek_1_1/psfmodel_test_insitu_single.h5'
f = h5.File(psffile, 'r')
I_init=np.array(f.get('res/I_model')).astype(np.float32)
bead_data = np.array(f.get('rois/psf_data')).astype(np.float32)
params = json.loads(f.attrs['params'])


#%%
#L.param['datapath'] = maindatadir+'190910_u2os_course_96_WGA_3D_M2/01_191009_u2os_course_96_WGA_3D_ritu_1/'
#L.param['datapath'] = maindatadir+'insitu data/'
L.param['datapath'] = maindatadir+r'insitu data\from Yiming\In-situ PSF learing data\Single molecule data for different DM models/U20S_MT_tublin_AF647_depth_200nm_2_2_zerniek_-1_1/'

#L.param['datapath'] = r'D:\Sheng\data\12-08-2021 bead bottom\40nm/'
#L.param['datapath'] = maindatadir+ r'bead data\01-04-2022 bead\40nm_top/'

L.param['keyword'] = 'Default.'
L.param['subfolder'] = ''
L.param['format'] = '.tif'
L.param['channeltype'] = 'single'
L.param['datatype'] = 'smlm'
L.param['gain'] = 0.47
L.param['ccd_offset'] = 100
L.param['frame_range'] = [1000,3000]# start and end frames in SMLM data
L.param['PSFtype'] = 'insitu'
images = L.load_data()

L.getpsfclass()

#%%
L.param['pixelsize_x'] =  0.108
L.param['pixelsize_y'] = 0.108
L.param['pixelsize_z'] = 0.05 # this value can be arbiturary, currently I'm setting it the same as in the initial PSF model
L.param['roi_size'] = [21,21]
L.param['gaus_sigma'] = [2,2]
L.param['peak_height'] =  0.2
L.param['plotall'] = True
L.param['FOV']['x_center'] = 0
L.param['FOV']['y_center'] = 0
L.param['FOV']['radius'] = 0
dataobj = L.prep_data(images)

#%%
L.param['iteration'] = 200
L.param['option_params']['emission_wavelength'] = 0.67
L.param['option_params']['RI_imm'] = 1.406
L.param['option_params']['RI_med'] = 1.33
L.param['option_params']['RI_cov'] = 1.524
L.param['loss_weight']['smooth'] = 0.0
L.param['option_params']['NA'] = 1.35
L.param['option_params']['n_max'] = 6
L.param['option_params']['const_pupilmag'] = True
L.param['option_params']['symmetric_mag'] = True
L.param['option_params']['with_apoid'] = True
L.param['option_params']['stage_pos'] =  0.8 # um
L.param['rej_threshold']['bias_z'] = 0.99 # here is the quantile in x and y
L.param['rej_threshold']['mse'] = 0.8 # quantile in mse
L.param['option_params']['init_pupil_file'] = ''
L.param['option_params']['min_photon'] = 0.2
L.param['option_params']['partition_data'] = True
L.param['option_params']['partition_size'] = [11,100] # [segment #, # of psf per segment]
L.param['batch_size'] = 2000
L.param['option_params']['zernike_index'] = [5] # if set to empty, it will search between 4 to 20
L.param['option_params']['zernike_coeff'] = [0.5]
L.param['option_params']['z_range'] = 2.0
psfobj,fitter = L.learn_psf(dataobj,time=0)

#%%
L.param['savename'] = L.param['datapath'] + 'psfmodel_test1'
resfile, res_dict, loc_dict, rois_dict = L.save_result(psfobj,dataobj,fitter)

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

plt.plot(res_dict['zernike_coeff'].transpose(),'.-')
plt.legend(['magnitude','phase'])
plt.xlabel('zernike polynomial')
plt.ylabel('coefficient')
plt.show()

#%%
plt.imshow(Zk[23]*aperture,cmap='viridis')
#%%
zf = res_dict['pos'][:,0]*params['pixelsize_z']*1e3
plt.plot(frames,zf,'.')
plt.xlabel('frames')
plt.ylabel('learned z (nm)')
#%% compare bead and insitu PSF

I_init1 = I_init
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





   



#%%
Nz = 41
psfobj = PSFZernikeBased_vector_smlm(options=L.param['option_params'])
psfobj.data = dataobj
pixelsize_z = np.array(psfobj.data.pixelsize_z)
psfobj.stagepos = psfobj.options['stage_pos']/psfobj.data.pixelsize_z
psfobj.calpupilfield('vector', Nz)
init_sigma = np.ones((2,),dtype=np.float32)*psfobj.options['gauss_filter_sigma']*np.pi

LL = []
LLavg = []
for k in range(4,21):
    for val in [-0.5,0.5]:
        init_Zcoeff = np.zeros((2,psfobj.Zk.shape[0],1,1),dtype=np.float32)
        init_Zcoeff[0,0,0,0] = 1
        init_Zcoeff[1,k,0,0] = val
        I_init = psfobj.genpsfmodel(init_Zcoeff,init_sigma)

        zind = range(0,Nz-1,4)
        fig = plt.figure(figsize=[3*len(zind),3])
        for i,id in enumerate(zind):
            ax = fig.add_subplot(1,len(zind),i+1)
            plt.imshow(I_init[id],cmap='twilight')
            plt.axis('off')
        
        dll = localizationlib(usecuda=True)
        locres = dll.loc_ast(dataobj.rois,I_init,pixelsize_z,start_time=0.0)
        LL.append(locres[2])
        LLavg.append(np.median(locres[2]))


#%%
LLavg = np.stack(LLavg).reshape(-1,2)
plt.plot(LLavg)
#%%
llmax = LL[0]
for i,ll in enumerate(LL):
    if np.median(ll-llmax)>0.0:
        llmax = ll
        zernike_index = i//2+4
        zernike_coeff = np.mod(i,2)-0.5

print([zernike_index,zernike_coeff])
# %% issues
# 1. optimize zernike coefficient sequentially, e.g. optimize order from 1 to 5, then from 6 to 9
# 2. SMLM data has small z range, sampling in z is not uniform
# 3. large datasets, optimize in batches
# 4. pupil magnitude has to be cicular symmetric, otherwise, causing a lot fluctuation in learned coefficient
# 5. initial PSF model, it can be obtained by collecting one or two bead data, 
#    but for thick sample, the initial model might need to include index mismatch aberration
# 6. start with astigmatism PSF model