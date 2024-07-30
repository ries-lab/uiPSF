import numpy as np
import scipy as sp
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.ndimage.filters import gaussian_filter
from .PSFInterface_file import PSFInterface
from ..data_representation.PreprocessedImageDataInterface_file import PreprocessedImageDataInterface
from ..loss_functions import mse_real_zernike_FD
from .. import utilities as im
#from .. import imagetools as nip

class PSFZernikeBased_FD(PSFInterface):
    """
    PSF class that uses a 3D volume to describe the PSF.
    Should only be used with single-channel data.
    """
    def __init__(self,options=None) -> None:
        self.parameters = None
        self.data = None
        self.Zphase = None
        self.zT = None
        self.bead_kernel = None
        self.options = options
        self.default_loss_func = mse_real_zernike_FD
        self.psftype = 'scalar'
        return

    def calc_initials(self, data: PreprocessedImageDataInterface, start_time=None):
        """
        Provides initial values for the optimizable varibales for the fitter class.
        """
        self.data = data
        _, rois, _, _ = self.data.get_image_data()

        init_positions = np.zeros((rois.shape[0], len(rois.shape)-1))

        init_backgrounds = np.array(np.min(gaussian_filter(rois, [0, 2, 2, 2]), axis=(-3, -2, -1), keepdims=True))
        init_intensitiesL = np.sum(rois - init_backgrounds, axis=(-2, -1), keepdims=True)
        init_intensities = np.mean(init_intensitiesL,axis=1,keepdims=True)
        
        self.gen_bead_kernel()
        N = rois.shape[0]
        Nz = self.bead_kernel.shape[0]
        Lx = rois.shape[-1]
     
        imsz = self.data.image_size
        div = self.options.model.division
        yy1, xx1 = tf.meshgrid(tf.linspace(0,imsz[-2],imsz[-2]//div), tf.linspace(0,imsz[-1],imsz[-1]//div),indexing='ij')

        if self.psftype=='vector':
            self.calpupilfield('vector')
        else:
            self.calpupilfield('scalar')
        if self.options.model.const_pupilmag:
            self.n_max_mag = 0
        else:
            self.n_max_mag = 100
        
        sigma = np.ones((2,))*self.options.model.blur_sigma*np.pi

        
        #self.weight = np.array([np.median(init_intensities), 10, 0.1, 1],dtype=np.float32)
        #weight = [1e5,10] + list(np.array([0.1,4])/np.median(init_intensities)*2e4)
        init_backgrounds[init_backgrounds<0.1] = 0.1
        bgmean = np.median(init_backgrounds)
        wI = np.lib.scimath.sqrt(np.median(init_intensities))
        weight = [wI*100,bgmean] + list(np.array([1,20])/wI*40)

        self.weight = np.array(weight,dtype=np.float32)
        Zmap = np.zeros((2,self.Zk.shape[0])+xx1.shape,dtype = np.float32)
        Zmap[0,0] = 1.0/self.weight[3]
        
        init_backgrounds = np.ones((N,1,1,1),dtype = np.float32)*np.median(init_backgrounds,axis=0, keepdims=True) / self.weight[1]
        gxy = np.zeros((N,2),dtype=np.float32) 
        gI = np.ones((N,Nz,1,1),dtype = np.float32)*init_intensities
        # st = (self.bead_kernel.shape[0]-self.data.rois[0].shape[-3])//2
        # gI[:,st:Nz-st] = init_intensitiesL
        # gI[:,0:st] = np.abs(np.min(init_intensitiesL[:,0]))
        # gI[:,-st:] = np.abs(np.min(init_intensitiesL[:,-1]))
        self.varinfo = [dict(type='Nfit',id=0),
            dict(type='Nfit',id=0),
            dict(type='Nfit',id=0),
            dict(type='shared'),
            dict(type='shared'),
            dict(type='Nfit',id=0)]

        if self.options.model.var_photon:
            init_Intensity = gI/self.weight[0]
        else:
            init_Intensity = init_intensities / self.weight[0]
        return [init_positions.astype(np.float32),
                init_backgrounds.astype(np.float32),
                init_Intensity.astype(np.float32),
                Zmap,
                sigma.astype(np.float32),
                gxy], start_time
        
    def calc_forward_images(self, variables):
        """
        Calculate forward images from the current guess of the variables.
        Shifting is done by Fourier transform and applying a phase ramp.
        """


        pos, backgrounds, intensities, Zmap, sigma, gxy = variables
        c1 = self.spherical_terms
        n_max = self.n_max_mag
        Nk = np.min(((n_max+1)*(n_max+2)//2,self.Zk.shape[0]))
        mask = (c1<Nk) & (c1>0)
        c1 = c1[mask]

        cor = np.float32(self.data.centers)
        imsz = self.data.image_size
        Zcoeff1 = [None] * Zmap.shape[-3]
        Zcoeff2 = [None] * Zmap.shape[-3]
        Zmap = Zmap*self.weight[3]
        cor = np.float32(self.data.centers)
        for i in range(0,Zmap.shape[-3]):
            Zcoeff1[i] = tfp.math.batch_interp_regular_nd_grid(cor[self.ind[0]:self.ind[1],-2:],[0.0,0.0],np.float32(imsz[-2:]),Zmap[0,i],axis=-2)
            Zcoeff2[i] = tfp.math.batch_interp_regular_nd_grid(cor[self.ind[0]:self.ind[1],-2:],[0.0,0.0],np.float32(imsz[-2:]),Zmap[1,i],axis=-2)


        Zcoeff1 = tf.transpose(tf.stack(Zcoeff1),perm=(1,0))
        Zcoeff1 = tf.reshape(Zcoeff1,Zcoeff1.shape+(1,1))
        Zcoeff2 = tf.transpose(tf.stack(Zcoeff2),perm=(1,0))
        Zcoeff2 = tf.reshape(Zcoeff2,Zcoeff2.shape+(1,1))

        if self.options.model.symmetric_mag:
            if len(c1)>0:
                pupil_mag = tf.reduce_sum(self.Zk[c1]*tf.gather(Zcoeff1,indices=c1,axis=1),axis=1,keepdims=True)
        else:
            pupil_mag = tf.reduce_sum(self.Zk[1:Nk]*Zcoeff1[:,1:Nk],axis=1,keepdims=True)
        pupil_mag = pupil_mag + self.Zk[0]*tf.reduce_mean(Zcoeff1[:,0])
        pupil_mag = tf.math.maximum(pupil_mag,0)

        pupil_phase = tf.reduce_sum(self.Zk[3:]*Zcoeff2[:,3:],axis=1,keepdims=True)
        pupil = tf.complex(pupil_mag*tf.math.cos(pupil_phase),pupil_mag*tf.math.sin(pupil_phase))*self.aperture*self.apoid
                
        Nz = self.Zrange.shape[0]

        pos = tf.complex(tf.reshape(pos,pos.shape+(1,1,1)),0.0)

        phiz = -1j*2*np.pi*self.kz*(pos[:,0]+self.Zrange)
        phixy = 1j*2*np.pi*self.ky*pos[:,1]+1j*2*np.pi*self.kx*pos[:,2]
        if self.psftype == 'vector':
            I_res = 0.0
            for h in self.dipole_field:
                PupilFunction = pupil*tf.exp(phiz+phixy)*h
                psfA = im.cztfunc1(PupilFunction,self.paramxy)      
                I_res += psfA*tf.math.conj(psfA)*self.normf
        else:
            PupilFunction = pupil*tf.exp(phiz+phixy)
            I_res = im.cztfunc1(PupilFunction,self.paramxy)
            I_res = I_res*tf.math.conj(I_res)*self.normf

        filter2 = tf.exp(-2*sigma[1]*sigma[1]*self.kspace_x-2*sigma[0]*sigma[0]*self.kspace_y)
        filter2 = tf.complex(filter2/tf.reduce_max(filter2),0.0)
        I_blur = im.ifft3d(im.fft3d(I_res)*self.bead_kernel*filter2)
        psf_fit = tf.math.real(I_blur)*intensities*self.weight[0]
        Nz = psf_fit.shape[-3]
        st = (self.bead_kernel.shape[0]-self.data.rois[0].shape[-3])//2
        psf_fit = psf_fit[:,st:Nz-st]

        if self.options.model.estimate_drift:
            gxy = gxy*self.weight[2]
            psf_shift = self.applyDrfit(psf_fit,gxy)
            forward_images = psf_shift + backgrounds*self.weight[1]
        else:
            forward_images = psf_fit + backgrounds*self.weight[1]

        return forward_images

    def genpsfmodel(self,sigma,Zmap=None,cor=None,pupil=None,addbead=False):
        Zcoeff = None
        if pupil is None:
            imsz = self.data.image_size
            Zcoeff1 = [None] * Zmap.shape[-3]
            Zcoeff2 = [None] * Zmap.shape[-3]
            
            for i in range(0,Zmap.shape[-3]):
                Zcoeff1[i] = tfp.math.batch_interp_regular_nd_grid(cor,[0.0,0.0],np.float32(imsz[-2:]),Zmap[0,i],axis=-2)
                Zcoeff2[i] = tfp.math.batch_interp_regular_nd_grid(cor,[0.0,0.0],np.float32(imsz[-2:]),Zmap[1,i],axis=-2)


            Zcoeff1 = tf.transpose(tf.stack(Zcoeff1),perm=(1,0))
            Zcoeff1 = tf.reshape(Zcoeff1,Zcoeff1.shape+(1,1))
            Zcoeff2 = tf.transpose(tf.stack(Zcoeff2),perm=(1,0))
            Zcoeff2 = tf.reshape(Zcoeff2,Zcoeff2.shape+(1,1))

            Zcoeff = tf.stack([Zcoeff1,Zcoeff2])

            pupil_mag = tf.reduce_sum(self.Zk*Zcoeff1,axis=-3,keepdims=True)
            pupil_mag = tf.math.maximum(pupil_mag,0)
            pupil_phase = tf.reduce_sum(self.Zk*Zcoeff2,axis=-3,keepdims=True)
            pupil = tf.complex(pupil_mag*tf.math.cos(pupil_phase),pupil_mag*tf.math.sin(pupil_phase))*self.aperture*self.apoid

        filter2 = tf.exp(-2*sigma[1]*sigma[1]*self.kspace_x-2*sigma[0]*sigma[0]*self.kspace_y)
        filter2 = tf.complex(filter2/tf.reduce_max(filter2),0.0)

        phiz = -1j*2*np.pi*self.kz*self.Zrange
        if self.psftype == 'vector':
            I_res = 0.0
            for h in self.dipole_field:
                PupilFunction = pupil*tf.exp(phiz)*h
                psfA = im.cztfunc1(PupilFunction,self.paramxy)      
                I_res += psfA*tf.math.conj(psfA)*self.normf
        else:
            PupilFunction = pupil*tf.exp(phiz)
            I_res = im.cztfunc1(PupilFunction,self.paramxy)      
            I_res = I_res*tf.math.conj(I_res)*self.normf

        if addbead:
            I_model = np.real(im.ifft3d(im.fft3d(I_res)*filter2*self.bead_kernel))
        else:
            I_model = np.real(im.ifft3d(im.fft3d(I_res)*filter2))

        return I_model, Zcoeff, pupil

    def postprocess(self, variables):
        """
        Applies postprocessing to the optimized variables. In this case calculates
        real positions in the image from the positions in the roi. Also, normalizes
        psf and adapts intensities and background accordingly.
        """
        positions, backgrounds, intensities, Zmap,sigma,gxy = variables
        z_center = (self.Zrange.shape[-3] - 1) // 2

        Zmap = Zmap*self.weight[3]
        Zmap[0,0] = Zmap[0,0,0,0]
        cor = np.float32(self.data.centers)

        Nbead = positions.shape[0]
    
        
        I_model, Zcoeff, pupil = self.genpsfmodel(sigma,Zmap,cor[:,-2:])
        #I_model_bead, _, _ = self.genpsfmodel(sigma,Zmap,cor[:,-2:],addbead=True)

        pupil_mag = tf.reduce_sum(self.Zk*Zcoeff[0],axis=(0,1))/Nbead
        pupil_phase = tf.reduce_sum(self.Zk*Zcoeff[1],axis=(0,1))/Nbead
        pupil_avg = tf.complex(pupil_mag*tf.math.cos(pupil_phase),pupil_mag*tf.math.sin(pupil_phase))*self.aperture*self.apoid

        I_model_avg, _, _ = self.genpsfmodel(sigma,pupil=pupil_avg,addbead=True)

        # calculate global positions in images since positions variable just represents the positions in the rois
        images, _, centers, _ = self.data.get_image_data()
        original_shape = images.shape[-3:]
        #centers_with_z = np.concatenate((np.full((centers.shape[0], 1), z_center), centers[:,-2:]), axis=1)

        # use modulo operator to get rid of periodicity from FFT shifting
        #global_positions = centers_with_z - positions
        global_positions = np.swapaxes(np.vstack((positions[:,0]+z_center,centers[:,-2]-positions[:,-2],centers[:,-1]-positions[:,-1])),1,0)

        # make sure everything has correct dtype
        # this is probably not needed anymore (see Fitter)
        # but just left since it does no harm
        return [global_positions.astype(np.float32),
                backgrounds*self.weight[1], # already correct
                intensities*self.weight[0], # already correct
                I_model_avg,
                I_model,
                np.complex64(pupil),
                Zmap,
                Zcoeff,    
                sigma,           
                gxy*self.weight[2],
                variables] # already correct
    
    def res2dict(self,res):
        res_dict = dict(pos=res[0],
                        bg=np.squeeze(res[1]),
                        intensity=np.squeeze(res[2]),
                        I_model_bead = res[3],
                        I_model_all = res[4],
                        pupil = np.squeeze(res[5]),
                        zernike_map = np.squeeze(res[6]),
                        zernike_coeff = np.squeeze(res[7]),
                        sigma = res[8]/np.pi,
                        drift_rate=res[9],
                        offset=np.min(res[4]),
                        zernike_polynomial = self.Zk,
                        apodization = self.apoid,
                        cor_all = self.data.centers_all,
                        cor = self.data.centers)

        return res_dict