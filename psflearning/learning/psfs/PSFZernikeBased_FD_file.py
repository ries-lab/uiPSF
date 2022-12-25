import numpy as np
import scipy as sp
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.ndimage.filters import gaussian_filter
from .PSFInterface_file import PSFInterface
from ..data_representation.PreprocessedImageDataInterface_file import PreprocessedImageDataInterface
from ..loss_functions import mse_real_zernike_FD
from .. import utilities as im
from .. import imagetools as nip

class PSFZernikeBased_FD(PSFInterface):
    """
    PSF class that uses a 3D volume to describe the PSF.
    Should only be used with single-channel data.
    """
    def __init__(self, estdrift=False, varphoton=False,options=None) -> None:
        self.parameters = None
        self.data = None
        self.estdrift = estdrift
        self.varphoton = varphoton
        self.Zphase = None
        self.zT = None
        self.bead_kernel = None
        self.options = options
        self.default_loss_func = mse_real_zernike_FD
        return

    def calc_initials(self, data: PreprocessedImageDataInterface, start_time=None):
        """
        Provides initial values for the optimizable varibales for the fitter class.
        """
        self.data = data
        _, rois, _, _ = self.data.get_image_data()

        init_positions = np.zeros((rois.shape[0], len(rois.shape)-1))

        # beacuse of using nip init_backgrounds would be of type image
        # I prefer it to be a numpy array --> use np.array
        # TODO: or maybe just use scipy.ndimage.filters.gaussian_filter if it works similarly
        init_backgrounds = np.array(np.min(gaussian_filter(rois, [0, 2, 2, 2]), axis=(-3, -2, -1), keepdims=True))
        #init_intensities = np.max(rois - init_backgrounds, axis=(-3, -2, -1), keepdims=True)
        init_intensitiesL = np.sum(rois - init_backgrounds, axis=(-2, -1), keepdims=True)
        init_intensities = np.mean(init_intensitiesL,axis=1,keepdims=True)
        # TODO: instead of using first roi as initial guess, use average
        roi_avg = np.mean((rois - init_backgrounds),axis=0)
        
        N = rois.shape[0]
        Nz = self.data.bead_kernel.shape[0]
        Lx = rois.shape[-1]
     
        imsz = self.data.image_size
        div = 20
        yy1, xx1 = tf.meshgrid(tf.linspace(0,imsz[-2],imsz[-2]//div), tf.linspace(0,imsz[-1],imsz[-1]//div),indexing='ij')

        self.calpupilfield('scalar')
        if self.options.model.const_pupilmag:
            self.n_max_mag = 0
        else:
            self.n_max_mag = 100
        
        sigma = np.ones((2,))*self.options.model.blur_sigma*np.pi

        
        self.bead_kernel = tf.complex(self.data.bead_kernel,0.0)
        self.weight = np.array([np.median(init_intensities), 10, 0.1, 1],dtype=np.float32)
        Zmap = np.zeros((2,self.Zk.shape[0])+xx1.shape,dtype = np.float32)
        Zmap[0,0] = 1.0/self.weight[3]
        
        init_backgrounds[init_backgrounds<0.1] = 0.1
        init_backgrounds = np.ones((N,1,1,1),dtype = np.float32)*np.median(init_backgrounds,axis=0, keepdims=True) / self.weight[1]
        gxy = np.zeros((N,2),dtype=np.float32) 
        gI = np.ones((N,Nz,1,1),dtype = np.float32)*init_intensities
        st = (self.bead_kernel.shape[0]-self.data.rois[0].shape[-3])//2
        gI[:,st:Nz-st] = init_intensitiesL
        gI[:,0:st] = np.abs(np.min(init_intensitiesL[:,0]))
        gI[:,-st:] = np.abs(np.min(init_intensitiesL[:,-1]))
        
        if self.varphoton:
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
        # it would be nice to get rid of InverseModelling and/or NanoImagingPack
        # unfortunately im.ft3d behaves differently than tf.signal.fft3d or np.fft.fftn
        # and my physics/maths skills were not good enough to understand the difference
        # then I run out of time
        # for now it is working with im.ft3d but it would be nice to get rid of InverseModelling
        # TODO: try to get rid of InverseModelling

        pos, backgrounds, intensities, Zmap, sigma, gxy = variables
        c1 = self.spherical_terms
        n_max = self.n_max_mag
        Nk = np.min(((n_max+1)*(n_max+2)//2,self.Zk.shape[0]))
        mask = c1<Nk
        c1 = c1[mask]

        cor = np.float32(self.data.centers)
        imsz = self.data.image_size
        Zcoeff1 = [None] * Zmap.shape[-3]
        Zcoeff2 = [None] * Zmap.shape[-3]
        Zmap = Zmap*self.weight[3]
        cor = np.float32(self.data.centers)
        for i in range(0,Zmap.shape[-3]):
            Zcoeff1[i] = tfp.math.batch_interp_regular_nd_grid(cor[:,-2:],[0,0],imsz[-2:],Zmap[0,i],axis=-2)
            Zcoeff2[i] = tfp.math.batch_interp_regular_nd_grid(cor[:,-2:],[0,0],imsz[-2:],Zmap[1,i],axis=-2)


        Zcoeff1 = tf.transpose(tf.stack(Zcoeff1),perm=(1,0))
        Zcoeff1 = tf.reshape(Zcoeff1,Zcoeff1.shape+(1,1))
        Zcoeff2 = tf.transpose(tf.stack(Zcoeff2),perm=(1,0))
        Zcoeff2 = tf.reshape(Zcoeff2,Zcoeff2.shape+(1,1))

        if self.options.model.symmetric_mag:
            pupil_mag = tf.reduce_sum(self.Zk[c1]*tf.gather(Zcoeff1,indices=c1,axis=1),axis=1,keepdims=True)
        else:
            pupil_mag = tf.abs(tf.reduce_sum(self.Zk[0:Nk]*Zcoeff1[:,0:Nk],axis=1,keepdims=True))
        pupil_phase = tf.reduce_sum(self.Zk[4:]*Zcoeff2[:,4:],axis=1,keepdims=True)
        pupil = tf.complex(pupil_mag*tf.math.cos(pupil_phase),pupil_mag*tf.math.sin(pupil_phase))*self.aperture*self.apoid
                
        Nz = self.Zrange.shape[0]

        pos = tf.complex(tf.reshape(pos,pos.shape+(1,1,1)),0.0)

        phiz = 1j*2*np.pi*self.kz*(pos[:,0]+self.Zrange)
        phixy = 1j*2*np.pi*self.ky*pos[:,1]+1j*2*np.pi*self.kx*pos[:,2]

        PupilFunction = pupil*tf.exp(phiz+phixy)
        IntermediateImage = tf.transpose(im.cztfunc(PupilFunction,self.paramx),perm=(0,1,3,2))
        I_res = tf.transpose(im.cztfunc(IntermediateImage,self.paramy),perm=(0,1,3,2))
        
        I_res = I_res*tf.math.conj(I_res)*self.normf

        #filter2 = tf.exp(-2*sigma*sigma*self.kspace)
        filter2 = tf.exp(-2*sigma[1]*sigma[1]*self.kspace_x-2*sigma[0]*sigma[0]*self.kspace_y)

        filter2 = filter2/tf.reduce_max(filter2)
        filter2 = tf.complex(filter2,0.0)

        I_blur = im.ift3d(im.ft3d(I_res)*self.bead_kernel*filter2)
        psf_fit = tf.math.real(I_blur)*intensities*self.weight[0]
        Nz = psf_fit.shape[-3]
        st = (self.bead_kernel.shape[0]-self.data.rois[0].shape[-3])//2
        psf_fit = psf_fit[:,st:Nz-st]

        if self.estdrift:
            
            Nz = psf_fit.shape[-3]
            zv = np.expand_dims(np.linspace(0,Nz-1,Nz,dtype=np.float32)-Nz/2,axis=-1)
            if self.data.skew_const:
                sk = np.array([self.data.skew_const],dtype=np.float32)
                gxy = gxy*self.weight[2] 
                otf2d = im.ft(psf_fit,axes=[-1,-2])
                otf2dphase = otf2d[0:1]
                for i,g in enumerate(gxy):
                    dxy = -sk*zv+tf.round(sk*zv)                    
                    tmp = self.applyPhaseRamp(otf2d[i],dxy)
                    otf2dphase = tf.concat((otf2dphase,tf.expand_dims(tmp,axis=0)),axis=0)
            else:
                gxy = gxy*self.weight[2]
                otf2d = im.ft(psf_fit,axes=[-1,-2])
                otf2dphase = otf2d[0:1]
                for i,g in enumerate(gxy):
                    dxy = g*zv
                    
                    tmp = self.applyPhaseRamp(otf2d[i],dxy)
                    otf2dphase = tf.concat((otf2dphase,tf.expand_dims(tmp,axis=0)),axis=0)

            psf_shift = tf.math.real(im.ift(otf2dphase[1:],axes=[-1,-2]))
            forward_images = psf_shift + backgrounds*self.weight[1]
        else:
            forward_images = psf_fit + backgrounds*self.weight[1]

        return forward_images




    def postprocess(self, variables):
        """
        Applies postprocessing to the optimized variables. In this case calculates
        real positions in the image from the positions in the roi. Also, normalizes
        psf and adapts intensities and background accordingly.
        """
        positions, backgrounds, intensities, Zmap,sigma,gxy = variables
        z_center = (self.Zrange.shape[-3] - 1) // 2
        #z_center = 0
        c1 = self.spherical_terms
        imsz = self.data.image_size
        Zcoeff1 = [None] * Zmap.shape[-3]
        Zcoeff2 = [None] * Zmap.shape[-3]
        Zmap = Zmap*self.weight[3]
        cor = np.float32(self.data.centers)
        for i in range(0,Zmap.shape[-3]):
            Zcoeff1[i] = tfp.math.batch_interp_regular_nd_grid(cor[:,-2:],[0,0],imsz[-2:],Zmap[0,i],axis=-2)
            Zcoeff2[i] = tfp.math.batch_interp_regular_nd_grid(cor[:,-2:],[0,0],imsz[-2:],Zmap[1,i],axis=-2)


        Zcoeff1 = tf.transpose(tf.stack(Zcoeff1),perm=(1,0))
        Zcoeff1 = tf.reshape(Zcoeff1,Zcoeff1.shape+(1,1))
        Zcoeff2 = tf.transpose(tf.stack(Zcoeff2),perm=(1,0))
        Zcoeff2 = tf.reshape(Zcoeff2,Zcoeff2.shape+(1,1))

        pupil_mag = tf.reduce_sum(self.Zk*Zcoeff1,axis=(0,1))/Zcoeff1.shape[0]
        pupil_phase = tf.reduce_sum(self.Zk*Zcoeff2,axis=(0,1))/Zcoeff2.shape[0]
        pupil_avg = tf.complex(pupil_mag*tf.math.cos(pupil_phase),pupil_mag*tf.math.sin(pupil_phase))*self.aperture*self.apoid


        pupil_mag = tf.abs(tf.reduce_sum(self.Zk*Zcoeff1,axis=1,keepdims=True))
        pupil_phase = tf.reduce_sum(self.Zk*Zcoeff2,axis=1,keepdims=True)
        pupil = tf.complex(pupil_mag*tf.math.cos(pupil_phase),pupil_mag*tf.math.sin(pupil_phase))*self.aperture*self.apoid

        #filter2 = tf.exp(-2*sigma*sigma*self.kspace)
        filter2 = tf.exp(-2*sigma[1]*sigma[1]*self.kspace_x-2*sigma[0]*sigma[0]*self.kspace_y)

        filter2 = filter2/tf.reduce_max(filter2)
        filter2 = tf.complex(filter2,0.0)

        phiz = 1j*2*np.pi*self.kz*self.Zrange
        PupilFunction = pupil*tf.exp(phiz)
        IntermediateImage = tf.transpose(im.cztfunc(PupilFunction,self.paramx),perm=(0,1,3,2))
        I_res = tf.transpose(im.cztfunc(IntermediateImage,self.paramy),perm=(0,1,3,2))        
        I_res = np.real(I_res*tf.math.conj(I_res)*self.normf)
        I_model = np.real(im.ift3d(im.ft3d(I_res)*filter2))

        PupilFunction = pupil_avg*tf.exp(phiz)
        IntermediateImage = tf.transpose(im.cztfunc(PupilFunction,self.paramx),perm=(0,2,1))
        I_res = tf.transpose(im.cztfunc(IntermediateImage,self.paramy),perm=(0,2,1))        
        I_res = np.real(I_res*tf.math.conj(I_res)*self.normf)
        I_model_avg = np.real(im.ift3d(im.ft3d(I_res)*filter2))


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
                np.stack([Zcoeff1,Zcoeff2]),    
                sigma,           
                gxy*self.weight[2],
                variables] # already correct
    
    def res2dict(self,res):
        res_dict = dict(pos=res[0],
                        bg=np.squeeze(res[1]),
                        intensity=np.squeeze(res[2]),
                        I_model = res[3],
                        I_model_all = res[4],
                        pupil = np.squeeze(res[5]),
                        zernike_map = np.squeeze(res[6]),
                        zernike_coeff = np.squeeze(res[7]),
                        sigma = res[8]/np.pi,
                        drift_rate=res[9],
                        offset=np.min(res[3]),
                        zernike_polynomial = self.Zk,
                        apodization = self.apoid,
                        cor_all = self.data.centers_all,
                        cor = self.data.centers)

        return res_dict