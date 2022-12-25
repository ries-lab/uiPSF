#%%
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
#L.param['datapath'] = maindatadir + r'\insitu data\210122_Ulf_1C3D_M2\beadstacks/'

#L.param['datapath']= maindatadir + r'bead data\211207_SL_bead_3D_M2\40nm_bead_50nm/'
#L.param['datapath'] = maindatadir + r'\sim data\DH_single\img_bsz50nm/'
#L.param['datapath'] = maindatadir + r'sim data\ast_single\img3_bsz50nm/'
L.param['datapath'] = maindatadir+r'\bead data\211207_SL_bead_3D_M2/40nm_bead_50nm/'
#L.param['datapath'] = r'D:\Sheng\data\12-08-2021 bead bottom\40nm/'
#L.param['datapath'] = maindatadir+r'bead data\01-03-2022 bead\100nm_top/'

L.param['keyword'] = 'bead'
L.param['subfolder'] = 'bead'
L.param['format'] = '.tif'
L.param['channeltype'] = 'single'
L.param['varname'] = ''
L.param['gain'] = 4.55
L.param['ccd_offset'] = 400
L.param['stage_mov_dir'] = 'normal'
images = L.load_data()

#%%
L.param['PSFtype'] = 'voxel'
L.getpsfclass()

#%%
L.param['pixelsize_x'] =  0.127
L.param['pixelsize_y'] = 0.127
L.param['pixelsize_z'] = 0.05
L.param['roi_size'] = [31,31]
L.param['gaus_sigma'] = [2,2]
L.param['peak_height'] = 0.2
L.param['plotall'] = True
L.param['max_bead_number'] = 100
L.param['bead_radius'] = 0.0
L.param['FOV']['z_start'] = 1
L.param['FOV']['z_end'] = -0
L.param['FOV']['z_step'] = 1
dataobj = L.prep_data(images)

#%%
L.param['iteration'] = 100
L.param['vary_photon'] = False
L.param['rej_threshold']['bias_z'] = 5
L.param['rej_threshold']['mse'] = 3
L.param['rej_threshold']['photon'] = 1.5
psfobj,fitter = L.learn_psf(dataobj,time=0)

#%%
L.param['savename'] = L.param['datapath'] + 'psfmodel_test'
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

ind1 = 0
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
N1 = 10
fig = plt.figure(figsize=[3*N1,3])
for i,zk in enumerate(Zk[0:N1]):

    ax = fig.add_subplot(1,N1,i+1)
    plt.imshow(zk*aperture,cmap='viridis')
    #plt.colorbar(orientation='horizontal')
    plt.axis('off')





