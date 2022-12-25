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
#L.param['datapath']  = maindatadir + r'sim data\4pi\img1_bsz50nm/'
#L.param['datapath'] =  maindatadir + r'bead data\04-28-2022 bead/'
L.param['datapath'] = maindatadir + r'bead data\01-08-2022 bead\40nm_600/'
#L.param['datapath'] = r'D:\Sheng\data\02-10-2022 yeast mMaple\100nm_600/'
#L.param['datapath'] = r'D:\Sheng\data\06-15-2022 seipin AF647\100nm_676/'

L.param['keyword'] = 'bead'
L.param['subfolder'] = 'bead'
L.param['format'] = '.mat'
L.param['varname'] = ''
L.param['channeltype'] = '4pi'
L.param['gain'] = 1/2.27
L.param['ccd_offset'] = 100
L.param['stage_mov_dir'] = 'reverse'
images = L.load_data()

#%%
L.param['PSFtype'] = 'voxel'
L.getpsfclass()

#%%
L.param['pixelsize_x'] =  0.129
L.param['pixelsize_y'] = 0.129
L.param['pixelsize_z'] = 0.05
L.param['roi_size'] = [21,21]
L.param['plotall'] = True
L.param['max_bead_number'] = 50
L.param['peak_height'] = 0.2
L.param['modulation_period'] = 0.29
L.param['bead_radius'] = 0.05
L.param['FOV']['z_start'] = 0
L.param['FOV']['z_end'] = 0
L.param['FOV']['z_step'] = 1
dataobj = L.prep_data(images)

#%%
L.param['iteration'] = 100
L.param['estimate_drift'] = False
L.param['option_params']['phase_dm'] = [2,0,-2]
L.param['loss_weight']['mse1'] = 1
L.param['rej_threshold']['bias_z'] = 5
L.param['rej_threshold']['mse'] = 3
L.param['rej_threshold']['photon'] = 1.5
psfobj,fitter = L.learn_psf(dataobj,time=0)

#%%
L.param['savename'] = L.param['datapath'] + 'psfmodel_test'
resfile, res_dict, loc_dict, rois_dict = L.save_result(psfobj,dataobj,fitter)


#%%
psf_data = rois_dict['psf_data']
psf_fit = rois_dict['psf_fit']
Nchannel = psf_data.shape[0]

# %% show results
np.set_printoptions(precision=4,suppress=True)
print(res_dict['T'])
print(res_dict['channel0']['offset'])
cor = rois_dict['cor']
ref_cor = cor[0]
ref_pos1 = res_dict['channel0']['pos']
fig = plt.figure(figsize=[16,8])
ax = fig.add_subplot(2,4,1)
plt.plot(ref_pos1[:,1]-ref_cor[:,0])
ax.set_title('x')
ax = fig.add_subplot(2,4,2)
plt.plot(ref_pos1[:,2]-ref_cor[:,1])
ax.set_title('y')
ax = fig.add_subplot(2,4,3)
plt.plot(ref_pos1[:,0])
ax.set_title('z')
ax = fig.add_subplot(2,4,4)
plt.plot(np.angle(res_dict['channel0']['intensity']))
ax.set_title('phi')
ax = fig.add_subplot(2,4,5)
plt.plot(np.abs(res_dict['channel0']['intensity']))
ax.set_title('intensity')
ax = fig.add_subplot(2,4,6)
plt.plot(res_dict['channel0']['bg'])
ax.set_title('background')
ax = fig.add_subplot(2,4,7)
plt.plot(res_dict['channel0']['drift_rate'])
ax.set_title('drift rate')

# %%
ch = 1
ind2 = 10
cc = psf_data.shape[-1]//2
imavg = psf_data[ch,ind2,0]
imavg1 = psf_fit[ch,ind2,0]

fig = plt.figure(figsize=[12,6])
ax = fig.add_subplot(1,2,1)
plt.plot(imavg1[:,cc,:])
plt.plot(imavg[:,cc,:],'.')

ax = fig.add_subplot(1,2,2)
plt.plot(imavg[:,cc,:],'.')
plt.plot(imavg1[:,cc,:])
plt.show()

fig = plt.figure(figsize=[12,6])
ax = fig.add_subplot(1,3,1)
plt.imshow(imavg[:,cc,:],cmap='twilight')
plt.colorbar(orientation='vertical')
ax = fig.add_subplot(1,3,2)
plt.imshow(imavg1[:,cc,:],cmap='twilight')
plt.colorbar(orientation='vertical')
ax = fig.add_subplot(1,3,3)
plt.imshow(imavg1[:,cc,:]-imavg[:,cc,:],cmap='twilight')
plt.colorbar(orientation='vertical')
plt.show()

#%%
ch = 0
ind2 = 12
im1 = psf_data[ch,ind2,0]
im2 = psf_fit[ch,ind2,0]
Nz = im1.shape[0]
zind = range(0,Nz,Nchannel)
fig = plt.figure(figsize=[3*len(zind),6])
for i,id in enumerate(zind):
    ax = fig.add_subplot(2,len(zind),i+1)
    plt.imshow(im1[id],cmap='twilight')
    plt.axis('off')
    ax = fig.add_subplot(2,len(zind),i+1+len(zind))
    plt.imshow(im2[id],cmap='twilight')
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
ax.set_ylim([-20,20])
ax = fig.add_subplot(1,3,2)
plt.plot(loc_dict['loc']['y'].transpose()*py,'k',alpha=0.1)
plt.plot(loc_dict['loc']['x'][0]*0.0,'r')
ax.set_ylabel('y bias (nm)')
ax.set_ylim([-20,20])
ax = fig.add_subplot(1,3,3)
plt.plot(np.transpose(loc_dict['loc']['z']-np.linspace(0,Nz-1,Nz))*pz,'k',alpha=0.1)
plt.plot(loc_dict['loc']['x'][0]*0.0,'r')
ax.set_ylabel('z bias (nm)')
ax.set_ylim([-10,10])
# %%
print(res_dict['channel0']['phase_dm'])
# %%
zT = np.array([dataobj.channels[0].zT])
ind1 =0
I2 = res_dict['channel'+str(ind1)]['I_model']
A2 = res_dict['channel'+str(ind1)]['A_model']
cc = I2.shape[-1]//2+0
fig = plt.figure(figsize=[8,4])
ax = fig.add_subplot(1,2,1)
plt.plot(I2[:,cc,:])

ax = fig.add_subplot(1,2,2)
plt.plot(np.imag(A2[:,cc,:]))
plt.show()
# %%
ref_pos = res_dict['channel0']['pos']
dxy = res_dict['xyshift'] 
fig = plt.figure(figsize=[4*Nchannel,4])

for i in range(0,Nchannel):
    pos = res_dict['channel'+str(i)]['pos']
    ax = fig.add_subplot(1,Nchannel,i+1)
    plt.plot(ref_pos[:,1],ref_pos[:,2],'.')
    plt.plot(pos[:,1]-dxy[i][0],pos[:,2]-dxy[i][1],'x')
    plt.plot(fitter.psf.imgcenter[0],fitter.psf.imgcenter[1],'o')






