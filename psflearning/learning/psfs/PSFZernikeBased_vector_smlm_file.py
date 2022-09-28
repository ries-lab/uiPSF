import numpy as np
import scipy as sp
import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter
from .PSFInterface_file import PSFInterface
from ..data_representation.PreprocessedImageDataInterface_file import PreprocessedImageDataInterface
from ..loss_functions import mse_real_zernike
from .. import utilities as im
from .. import imagetools as nip
from ..loclib import localizationlib

class PSFZernikeBased_vector_smlm(PSFInterface):
    """
    PSF class that uses a 3D volume to describe the PSF.
    Should only be used with single-channel data.
    """
    def __init__(self, options=None) -> None:
        self.parameters = None
        self.data = None

        self.Zphase = None
        self.zT = None
        self.bead_kernel = None
        self.options = options
        self.default_loss_func = mse_real_zernike
        return

    def calc_initials(self, data: PreprocessedImageDataInterface, start_time=None):
        """
        Provides initial values for the optimizable varibales for the fitter class.
        """
        self.data = data
        _, rois, centers, frames = self.data.get_image_data()
        pixelsize_z = np.array(self.data.pixelsize_z)
        self.stagepos = self.options['stage_pos']/self.data.pixelsize_z
        if hasattr(self,'initpsf'):
            I_init = self.initpsf
            Nz = I_init.shape[0]
            self.calpupilfield('vector', Nz)
        elif not self.options['zernike_index']:
            I_init = self.estzernike(start_time=start_time)
        else:
            Nz = np.int32(self.options['z_range']/self.data.pixelsize_z+1)
            self.calpupilfield('vector', Nz)
            init_sigma = np.ones((2,),dtype=np.float32)*self.options['gauss_filter_sigma']*np.pi
            init_Zcoeff = np.zeros((2,self.Zk.shape[0],1,1),dtype=np.float32)
            init_Zcoeff[0,0,0,0] = 1
            init_Zcoeff[1,self.options['zernike_index'],0,0] = self.options['zernike_coeff']
            I_init = self.genpsfmodel(init_Zcoeff,init_sigma)
        
        dll = localizationlib(usecuda=True)
        locres = dll.loc_ast(rois,I_init,pixelsize_z,start_time=start_time)

        xp = locres[-1]['x']
        yp = locres[-1]['y']
        zp = locres[-1]['z']     
        photon = locres[0][2]  
        bg = locres[0][3]
        a = 0.99
        a1 = self.options['min_photon']
        mask = (xp>np.quantile(xp,1-a)) & (xp<np.quantile(xp,a)) & (yp>np.quantile(yp,1-a)) & (yp<np.quantile(yp,a)) & (zp>np.quantile(zp,1-a)) & (zp<np.quantile(zp,a))
        mask = mask.flatten() & (locres[2]>np.quantile(locres[2],0.1)) & (photon>np.quantile(photon,a1))
        

        self.data.rois = rois[mask]
        self.data.centers = centers[mask,:]
        self.data.frames = frames[mask]
        initz = zp.flatten()[mask]
        LL = locres[2][mask]
        
        if self.options['partition_data']:
            initz, roisavg = self.partitiondata(initz,LL)
            
        _, rois, _, _ = self.data.get_image_data()

        init_positions = np.zeros((rois.shape[0], 3))
      
        
        z_center = self.stagepos*self.nmed/self.nimm
        init_positions[:,0] = (initz-z_center)

        init_backgrounds = np.array(np.min(gaussian_filter(rois, [0, 2, 2]), axis=(-2, -1), keepdims=True))
        init_intensities = np.sum(rois - init_backgrounds, axis=(-2, -1), keepdims=True)
       
     
                                
        if self.options['const_pupilmag']:
            self.n_max_mag = 0
        else:
            self.n_max_mag = 100

        
        self.bead_kernel = tf.complex(self.data.bead_kernel,0.0)
        self.weight = np.array([np.median(init_intensities)*10, 10, 30, 0.2, 0.2],dtype=np.float32) # [I, bg, pos, coeff]
        sigma = np.ones((2,))*self.options['gauss_filter_sigma']*np.pi
        
        init_Zcoeff = np.zeros((2,self.Zk.shape[0],1,1))
        init_Zcoeff[:,0,0,0] = [1,0]/self.weight[4]
        init_backgrounds[init_backgrounds<0.1] = 0.1
        init_backgrounds = init_backgrounds / self.weight[1]
        init_Intensity = init_intensities / self.weight[0]
        init_positions = init_positions / self.weight[2]

        return [init_positions.astype(np.float32),
                init_backgrounds.astype(np.float32),
                init_Intensity.astype(np.float32),
                init_Zcoeff.astype(np.float32),
                sigma.astype(np.float32)], start_time
        
    def calc_forward_images(self, variables):
        """
        Calculate forward images from the current guess of the variables.
        Shifting is done by Fourier transform and applying a phase ramp.
        """
        pos, backgrounds, intensities, Zcoeff, sigma = variables
        c1 = self.spherical_terms
        n_max = self.n_max_mag
        Nk = np.min(((n_max+1)*(n_max+2)//2,self.Zk.shape[0]))
        mask = c1<Nk
        c1 = c1[mask]
        if self.options['symmetric_mag']:
            pupil_mag = tf.abs(tf.reduce_sum(self.Zk[c1]*tf.gather(Zcoeff[0],indices=c1)*self.weight[4],axis=0))
        else:
            pupil_mag = tf.abs(tf.reduce_sum(self.Zk[0:Nk]*Zcoeff[0][0:Nk]*self.weight[4],axis=0))
        pupil_phase = tf.reduce_sum(self.Zk[4:]*Zcoeff[1][4:]*self.weight[3],axis=0)
        pupil = tf.complex(pupil_mag*tf.math.cos(pupil_phase),pupil_mag*tf.math.sin(pupil_phase))*self.aperture*self.apoid
                
        pos = tf.complex(tf.reshape(pos*self.weight[2],pos.shape+(1,1)),0.0)

        #phiz = 1j*2*np.pi*self.kz*(pos[:,0])
        
        IMMphase = 1j*2*np.pi*(self.kz_med*self.stagepos*self.nmed/self.nimm-self.kz*self.stagepos)
        phiz = 1j*2*np.pi*self.kz_med*pos[:,0] + IMMphase
        phixy = 1j*2*np.pi*self.ky*pos[:,1]+1j*2*np.pi*self.kx*pos[:,2]
        I_res = 0.0
        for h in self.dipole_field:
            PupilFunction = pupil*tf.exp(phiz+phixy)*h
            IntermediateImage = tf.transpose(im.cztfunc(PupilFunction,self.paramx),perm=(0,2,1))
            psfA = tf.transpose(im.cztfunc(IntermediateImage,self.paramy),perm=(0,2,1))        
            I_res += psfA*tf.math.conj(psfA)*self.normf

        #filter2 = tf.exp(-2*sigma*sigma*self.kspace)
        filter2 = tf.exp(-2*sigma[1]*sigma[1]*self.kspace_x-2*sigma[0]*sigma[0]*self.kspace_y)

        filter2 = filter2/tf.reduce_max(filter2)
        filter2 = tf.complex(filter2,0.0)

        I_blur = im.ift(im.ft(I_res,axes=[-1,-2])*filter2,axes=[-1,-2])
        psf_fit = tf.math.real(I_blur)*intensities*self.weight[0]
        
        forward_images = psf_fit + backgrounds*self.weight[1]

        return forward_images


    def estzernike(self,start_time = None):
        Nz = np.int32(self.options['z_range']/self.data.pixelsize_z+1)
        self.calpupilfield('vector', Nz)
        pixelsize_z = np.array(self.data.pixelsize_z)
        init_sigma = np.ones((2,),dtype=np.float32)*self.options['gauss_filter_sigma']*np.pi

        llmax = -1e6
        LLavg = []
        for k in range(4,21):
            for val in [-0.5,0.5]:
                init_Zcoeff = np.zeros((2,self.Zk.shape[0],1,1),dtype=np.float32)
                init_Zcoeff[0,0,0,0] = 1
                init_Zcoeff[1,k,0,0] = val
                I_init = self.genpsfmodel(init_Zcoeff,init_sigma)
                
                dll = localizationlib(usecuda=True)
                locres = dll.loc_ast(self.data.rois,I_init,pixelsize_z,start_time=start_time)
                start_time = locres[-2]
                ll = locres[2]
                LLavg.append(np.median(ll))
                if np.median(ll-llmax)>0.0:
                    llmax = ll
                    zernike_index = k
                    zernike_coeff = val
                    I_init_optim = I_init
        return I_init_optim

    def genpsfmodel(self,Zcoeff,sigma):
        pupil_mag = tf.abs(tf.reduce_sum(self.Zk*Zcoeff[0],axis=0))
        pupil_phase = tf.reduce_sum(self.Zk*Zcoeff[1],axis=0)
        pupil = tf.complex(pupil_mag*tf.math.cos(pupil_phase),pupil_mag*tf.math.sin(pupil_phase))*self.aperture*self.apoid

        z_center = self.stagepos*self.nmed/self.nimm
        zrange = -self.Zrange+self.Zrange[0]-z_center
        IMMphase = 1j*2*np.pi*(self.kz_med*self.stagepos*self.nmed/self.nimm-self.kz*self.stagepos)  
        phiz = 1j*2*np.pi*self.kz_med*zrange + IMMphase
        #phiz = 1j*2*np.pi*self.kz*self.Zrange
        phixy = 1j*2*np.pi*self.ky*0.0+1j*2*np.pi*self.kx*0.0
        I_res = 0.0
        for h in self.dipole_field:
            PupilFunction = pupil*tf.exp(phiz+phixy)*h
            IntermediateImage = tf.transpose(im.cztfunc(PupilFunction,self.paramx),perm=(0,2,1))
            psfA = tf.transpose(im.cztfunc(IntermediateImage,self.paramy),perm=(0,2,1))        
            I_res += psfA*tf.math.conj(psfA)*self.normf

        I_res = np.real(I_res)
        
        filter2 = tf.exp(-2*sigma[1]*sigma[1]*self.kspace_x-2*sigma[0]*sigma[0]*self.kspace_y)

        filter2 = filter2/tf.reduce_max(filter2)
        filter2 = tf.complex(filter2,0.0)
        I_model = np.real(im.ift3d(im.ft3d(I_res)*filter2))
        
        return I_model

    def partitiondata(self,zf,LL):
        _, rois, centers, frames = self.data.get_image_data()
        
        nbin = self.options['partition_size'][0]
        count,edge = np.histogram(zf,nbin)
        ind = np.digitize(zf,edge)
        Npsf = self.options['partition_size'][1]
        rois1 = []
        rois1_avg = []
        zf1 = []
        cor = []
        fid = []
        for ii in range(1,nbin+1):
            mask = (ind==ii)
            im1 = rois[mask]
            Nslice = np.min((Npsf,im1.shape[0]))            
            #indsample = list(np.random.choice(im1.shape[0],Nslice,replace=False))
            indsample = np.argsort(-LL[mask])[0:Nslice]
            rois1.append(im1[indsample])
            rois1_avg.append(np.mean(rois1[-1],axis=0))
            cor.append(centers[mask,:][indsample])
            fid.append(frames[mask][indsample])
            zf1.append(zf[mask][indsample])
        rois1 = np.concatenate(rois1,axis=0)
        rois1_avg = np.stack(rois1_avg)
        zf1 = np.concatenate(zf1,axis=0)
        
        self.data.rois = rois1
        self.data.centers = np.concatenate(cor,axis=0)
        self.data.frames = np.concatenate(fid,axis=0)
        return zf1, rois1_avg

    def postprocess(self, variables):
        """
        Applies postprocessing to the optimized variables. In this case calculates
        real positions in the image from the positions in the roi. Also, normalizes
        psf and adapts intensities and background accordingly.
        """
        positions, backgrounds, intensities, Zcoeff,sigma = variables
        #z_center = (self.Zrange.shape[-3] - 1) // 2
        positions = positions*self.weight[2]
        pupil_mag = tf.abs(tf.reduce_sum(self.Zk*Zcoeff[0]*self.weight[4],axis=0))
        pupil_phase = tf.reduce_sum(self.Zk*Zcoeff[1]*self.weight[3],axis=0)
        pupil = tf.complex(pupil_mag*tf.math.cos(pupil_phase),pupil_mag*tf.math.sin(pupil_phase))*self.aperture*self.apoid

        z_center = self.stagepos*self.nmed/self.nimm
        zrange = -self.Zrange+self.Zrange[0]-z_center
        #phiz = 1j*2*np.pi*self.kz*self.Zrange
        IMMphase = 1j*2*np.pi*(self.kz_med*z_center-self.kz*self.stagepos)
        phiz = 1j*2*np.pi*self.kz_med*zrange + IMMphase
        phixy = 1j*2*np.pi*self.ky*0.0+1j*2*np.pi*self.kx*0.0
        I_res = 0.0
        for h in self.dipole_field:
            PupilFunction = pupil*tf.exp(phiz+phixy)*h
            IntermediateImage = tf.transpose(im.cztfunc(PupilFunction,self.paramx),perm=(0,2,1))
            psfA = tf.transpose(im.cztfunc(IntermediateImage,self.paramy),perm=(0,2,1))        
            I_res += psfA*tf.math.conj(psfA)*self.normf

        I_model = np.real(I_res)
        #filter2 = tf.exp(-2*sigma*sigma*self.kspace)
        filter2 = tf.exp(-2*sigma[1]*sigma[1]*self.kspace_x-2*sigma[0]*sigma[0]*self.kspace_y)

        filter2 = filter2/tf.reduce_max(filter2)
        filter2 = tf.complex(filter2,0.0)
        I_model = np.real(im.ift3d(im.ft3d(I_model)*filter2))
        # calculate global positions in images since positions variable just represents the positions in the rois
        images, _, centers, _ = self.data.get_image_data()
        original_shape = images.shape[-3:]
        #centers_with_z = np.concatenate((np.full((centers.shape[0], 1), z_center), centers[:,-2:]), axis=1)
        
        global_positions = np.swapaxes(np.vstack((positions[:,0]+z_center,centers[:,-2]-positions[:,-2],centers[:,-1]-positions[:,-1])),1,0)

        # use modulo operator to get rid of periodicity from FFT shifting
        #global_positions = centers_with_z - positions
            
        # make sure everything has correct dtype
        # this is probably not needed anymore (see Fitter)
        # but just left since it does no harm
        return [global_positions.astype(np.float32),
                backgrounds*self.weight[1], # already correct
                intensities*self.weight[0], # already correct
                I_model,
                np.complex64(pupil),
                Zcoeff*np.reshape(self.weight[[4,3]],(2,1,1,1)),     
                sigma,
                variables] # already correct
    

    def res2dict(self,res):
        res_dict = dict(pos=res[0],
                        bg=np.squeeze(res[1]),
                        intensity=np.squeeze(res[2]),
                        I_model = res[3],
                        pupil = res[4],
                        zernike_coeff = np.squeeze(res[5]),
                        sigma = res[6]/np.pi,
                        offset=np.min(res[3]),
                        zernike_polynomial = self.Zk,
                        apodization = self.apoid,
                        cor_all = self.data.centers_all,
                        cor = self.data.centers)

        return res_dict