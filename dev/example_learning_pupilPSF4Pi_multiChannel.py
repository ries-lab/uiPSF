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
#L.param['datapath'] = r'D:\Sheng\data\06-15-2022 seipin AF647\100nm_676/'
#L.param['datapath'] = maindatadir +  r'bead data\04-28-2022 bead/'
L.param['datapath'] = maindatadir + r'bead data\01-08-2022 bead\40nm_600/'
L.param['keyword'] = 'bead'
L.param['subfolder'] = 'bead'
L.param['format'] = '.mat'
L.param['channeltype'] = '4pi'
L.param['gain'] = 1/2.27
L.param['ccd_offset'] = 100
L.param['stage_mov_dir'] = 'reverse'

images = L.load_data()

#%%
L.param['PSFtype'] = 'pupil'
L.getpsfclass()

#%%
L.param['pixelsize_x'] =  0.129
L.param['pixelsize_y'] = 0.129
L.param['pixelsize_z'] = 0.05
L.param['roi_size'] = [21,21]
L.param['plotall'] = True
L.param['max_bead_number'] = 20
L.param['modulation_period'] = 0.29
L.param['bead_radius'] = 0.05
dataobj = L.prep_data(images)

#%%
L.param['iteration'] = 100
L.param['estimate_drift'] = False
L.param['loss_weight']['smooth'] = 0
L.param['option_params']['emission_wavelength'] = 0.68
L.param['option_params']['RI_imm'] = 1.406
L.param['option_params']['RI_med'] = 1.406
L.param['option_params']['NA'] = 1.35
L.param['option_params']['n_max'] = 8
L.param['option_params']['const_pupilmag'] = False
L.param['option_params']['with_apoid'] = False
L.param['option_params']['link_zernikecoeff'] = False
L.param['option_params']['symmetric_mag'] = False
L.param['option_params']['phase_dm'] = [2,0,-2]

psfobj,fitter = L.learn_psf(dataobj,time=0)

#%%
L.param['savename'] = L.param['datapath'] + 'psfmodel'
resfile, res_dict, loc_dict, rois_dict = L.save_result(psfobj,dataobj,fitter)


#%%
psf_data = rois_dict['psf_data']
psf_fit = rois_dict['psf_fit']
Nchannel = psf_data.shape[0]

# %% show results
np.set_printoptions(precision=4,suppress=True)
print(f.res.T)
print(f.channel0.offset)
cor = f.rois.cor[0]
pos = f.res.channel0.pos
fig = plt.figure(figsize=[16,8])
ax = fig.add_subplot(2,4,1)
plt.plot(pos[:,1]-cor[:,0])
plt.title('y')
ax = fig.add_subplot(2,4,2)
plt.plot(pos[:,2]-cor[:,1])
plt.title('x')
ax = fig.add_subplot(2,4,3)
plt.plot(pos[:,0])
plt.title('z')
ax = fig.add_subplot(2,4,4)
plt.plot(np.angle(f.res.channel0.intensity))
ax.set_title('phi')
ax = fig.add_subplot(2,4,5)
plt.plot(np.abs(f.res.channel0.intensity))
ax.set_title('intensity')
ax = fig.add_subplot(2,4,6)
plt.plot(f.res.channel0.bg)
ax.set_title('background')
ax = fig.add_subplot(2,4,7)
plt.plot(f.res.channel0.drift_rate)
ax.set_title('drift rate')

# %%
ch = 0
ind2 = 2
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
ind2 = 10
for ch in range(0,Nchannel):
    im1 = psf_data[ch,ind2,0]
    im2 = psf_fit[ch,ind2,0]
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
Zk = f.res.channel0.zernike_polynomial
fig = plt.figure(figsize=[16,8])
for i in range(0,Nchannel):
    ax = fig.add_subplot(2,4,i+1)
    pupil_phase = np.sum(Zk[4:]*f.res['channel'+str(i)].zernike_coeff_phase[0][4:].reshape((-1,1,1)),axis=0)*aperture
    plt.imshow(pupil_phase,cmap='bwr')
    plt.axis('off')
    ax = fig.add_subplot(2,4,i+5)
    pupil_phase = np.sum(Zk[4:]*f.res['channel'+str(i)].zernike_coeff_phase[1][4:].reshape((-1,1,1)),axis=0)*aperture
    plt.imshow(pupil_phase,cmap='bwr')
    plt.axis('off')
plt.show()

fig = plt.figure(figsize=[16,8])
for i in range(0,Nchannel):
    ax = fig.add_subplot(2,4,i+1)
    pupil_mag = np.sum(Zk*f.res['channel'+str(i)].zernike_coeff_mag[0].reshape((-1,1,1)),axis=0)*aperture
    plt.imshow(pupil_mag,cmap='bwr')
    plt.axis('off')
    ax = fig.add_subplot(2,4,i+5)
    pupil_mag = np.sum(Zk*f.res['channel'+str(i)].zernike_coeff_mag[1].reshape((-1,1,1)),axis=0)*aperture
    plt.imshow(pupil_mag,cmap='bwr')
    plt.axis('off')
plt.show()

#%%
aperture=np.float32(psfobj.sub_psfs[0].aperture)

fig = plt.figure(figsize=[16,8])
for i in range(0,Nchannel):
    ax = fig.add_subplot(2,4,i+1)
    pupil_phase = np.angle(f.res['channel'+str(i)].pupil1)
    plt.imshow(pupil_phase,cmap='bwr')
    plt.axis('off')
    ax = fig.add_subplot(2,4,i+5)
    pupil_phase = np.angle(f.res['channel'+str(i)].pupil2)
    plt.imshow(pupil_phase,cmap='bwr')
    plt.axis('off')
plt.show()

fig = plt.figure(figsize=[16,8])
for i in range(0,Nchannel):
    ax = fig.add_subplot(2,4,i+1)
    pupil_mag = np.abs(f.res['channel'+str(i)].pupil1)
    plt.imshow(pupil_mag,cmap='bwr',vmax=1.3,vmin=0.0)
    plt.axis('off')
    ax = fig.add_subplot(2,4,i+5)
    pupil_mag = np.abs(f.res['channel'+str(i)].pupil2)
    plt.imshow(pupil_mag,cmap='bwr',vmax=1.3,vmin=0.0)
    plt.axis('off')
plt.show()
# %%
fig = plt.figure(figsize=[16,8])
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
for i in range(0,Nchannel):
    
    ax1.plot(f.res['channel'+str(i)].zernike_coeff_mag[0],'.-')    
    ax2.plot(f.res['channel'+str(i)].zernike_coeff_phase[0],'.-')
    ax2.set_ylim((-0.6,0.6))
    ax3.plot(f.res['channel'+str(i)].zernike_coeff_mag[1],'.-')
    ax4.plot(f.res['channel'+str(i)]zernike_coeff_phase[1],'.-')
    #ax4.plot([16,28],res_dict['channel'+str(i)]['zernike_coeff2'][1][[16,28]],'o')
    ax4.set_ylim((-0.6,0.6))
# %%
px = p.pixel_size.x*1e3
py = p.pixel_size.y*1e3
pz = p.pixel_size.z*1e3
Nz = f.locres.loc.z.shape[1]
fig = plt.figure(figsize=[16,4])
ax = fig.add_subplot(1,3,1)
plt.plot(f.locres.loc.x.transpose()*px,'k',alpha=0.1)
plt.plot(f.locres.loc.x[0]*0.0,'r')
ax.set_ylabel('x bias (nm)')
ax.set_ylim([-40,40])
ax = fig.add_subplot(1,3,2)
plt.plot(f.locres.loc.y.transpose()*py,'k',alpha=0.1)
plt.plot(f.locres.loc.y[0]*0.0,'r')
ax.set_ylabel('y bias (nm)')
ax.set_ylim([-40,40])
ax = fig.add_subplot(1,3,3)
plt.plot(np.transpose(f.locres.loc.z-np.linspace(0,Nz-1,Nz))*pz,'k',alpha=0.1)
plt.plot(f.locres.loc.z[0]*0.0,'r')
ax.set_ylabel('z bias (nm)')
ax.set_ylim([-10,10])
# %%
for i in range(Nchannel):
    print(res_dict['channel'+str(i)]['sigma'])
for i in range(Nchannel):
    print(res_dict['channel'+str(i)]['modulation_depth'])
print(res_dict['channel0']['phase_dm'])
# %%
zT = np.array([dataobj.channels[0].zT])
ind1 = 1
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

# %%
plt.plot(res_dict['channel0']['obj_misalign']*129)
plt.ylabel('obj misalign (nm)')
