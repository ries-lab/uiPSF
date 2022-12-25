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
#L.param['datapath'] = maindatadir+r'sim data/ast/img2_bsz200nm/'
#L.param['datapath'] = r'D:\Sheng\data\12-08-2021 bead bottom\40nm/'
#L.param['datapath'] = maindatadir+r'bead data\01-04-2022 bead\40nm_top/'
L.param['datapath'] = maindatadir+r'bead data\190910_beads3Ddualcolor_M2/'
#L.param['datapath'] = maindatadir+r'bead data\220124_SL_bead_3D_2C_M2/bead4_tetraspec_50nmstep_triggerCam_10ms_SM/'

L.param['keyword'] = '3d'
L.param['subfolder'] = '3d'
L.param['format'] = '.tif'
L.param['channeltype'] = 'multi'
#L.param['varname'] = 'data_blur'
L.param['gain'] = 2.9
L.param['ccd_offset'] = 500
images = L.load_data()

#%%
L.param['PSFtype'] = 'voxel'
L.getpsfclass()

#%%
L.param['pixelsize_x'] =  0.127
L.param['pixelsize_y'] = 0.127
L.param['pixelsize_z'] = 0.04
L.param['roi_size'] = [25,25]
L.param['plotall'] = True
L.param['max_bead_number'] = 100
L.param['bead_radius'] = 0.0
L.param['FOV']['z_step'] = 2
L.param['FOV']['z_start'] = 1
dataobj = L.prep_data(images)

#%%
L.param['iteration'] = 100
L.param['vary_photon'] = False
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







# %% load example data
#folder = 'Z:/2021/21_10/211011_PH_Predoc_course_test_M2/00_beads_2C_3D/beads_2C_3D/'
#folder = 'Z:/2021/21_11/211124_AT_Beads_3D_2C_M2/beads_calib/'
#folder = 'Z:/2022/22_01/220124_SL_bead_3D_2C_M2/bead4_tetraspec_50nmstep_triggerCam_10ms_SM/'
folder = r'E:\Yiming\190910_beads3Ddualcolor_M2/'
#%% load data .mat
folder = 'D:/Sheng/data/01-04-2022 bead/40nm_top/'
offsetfile = 'C:/Users/Ries Lab/Documents/sheng-gitlab/4Pi-analysis-SL/ccdoffset.mat'



# %%

for i in range(0,len(imgs)):
    posi = pos[i]
    img1 = imgs[i]
    cori = cor[i]
    plt.figure(figsize=[10,10])
    plt.imshow(np.sum(np.max(img1,axis = 1),axis=0))   
    plt.plot(posi[:,2],posi[:,1],'ro',markersize = 8,markerfacecolor='none')
    plt.show()
# %% show selected beads
img1 = imgs[0]
cor1 = centers[0]
fid = file_idxs[0]
for i in range(0,img1.shape[0]):
    plt.figure(figsize=[12,12])
    plt.imshow(np.max(img1[i],axis = 0))
    mask = fid==i
    plt.plot(cor1[mask,1]-1,cor1[mask,0]-1,'ro',markersize = 8,markerfacecolor='none')
    plt.show()

# %%
