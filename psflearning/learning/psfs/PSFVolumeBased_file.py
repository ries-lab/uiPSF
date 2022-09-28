import numpy as np
import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter
from .PSFInterface_file import PSFInterface
from ..data_representation.PreprocessedImageDataInterface_file import PreprocessedImageDataInterface
from ..loss_functions import mse_real
from .. import utilities as im
from .. import imagetools as nip

class PSFVolumeBased(PSFInterface):
    """
    PSF class that uses a 3D volume to describe the PSF.
    Should only be used with single-channel data.
    """
    def __init__(self, estdrift=False, varphoton=False, options = None) -> None:
        self.parameters = None
        self.data = None
        self.estdrift = estdrift
        self.varphoton = varphoton
        self.bead_kernel = None
        self.options = options
        self.default_loss_func = mse_real
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
        init_intensities = np.sum(rois - init_backgrounds, axis=(-2, -1), keepdims=True)
        init_intensities = np.mean(init_intensities,axis=1,keepdims=True)
        # TODO: instead of using first roi as initial guess, use average
        roi_avg = np.mean((rois - init_backgrounds),axis=0)
        
        N = rois.shape[0]
        Nz = rois.shape[-3]
        #init_psf_params = roi_avg/np.mean(init_intensities)
        
        self.bead_kernel = tf.complex(self.data.bead_kernel,0.0)
        self.weight = np.array([np.quantile(init_intensities,0.1), 10, 0.1, 0.1],dtype=np.float32)
        #np.quantile(init_intensities,0.1)
        init_psf_params = np.zeros(rois[0].shape)+0.002/self.weight[3]
        init_backgrounds[init_backgrounds<0.1] = 0.1
        init_backgrounds = np.ones((N,1,1,1),dtype = np.float32)*np.median(init_backgrounds,axis=0, keepdims=True) / self.weight[1]
        gxy = np.zeros((N,2),dtype=np.float32) 
        gI = np.ones((N,Nz,1,1),dtype = np.float32)*init_intensities
        
        if self.varphoton:
            init_Intensity = gI/self.weight[0]
        else:
            init_Intensity = init_intensities / self.weight[0]
        return [init_positions.astype(np.float32),
                init_backgrounds.astype(np.float32),
                init_Intensity.astype(np.float32),
                init_psf_params.astype(np.float32),
                gxy],start_time
        
    def calc_forward_images(self, variables):
        """
        Calculate forward images from the current guess of the variables.
        Shifting is done by Fourier transform and applying a phase ramp.
        """

        pos, backgrounds, intensities, I_model, gxy = variables

        I_blur = im.ift3d(im.ft3d(I_model*self.weight[3])*self.bead_kernel)
        I_otfs = im.ft3d(I_blur)*tf.complex(intensities*0.0+1.0,0.0)
        I_res = im.ift3d(self.applyPhaseRamp(I_otfs,pos))*tf.complex(intensities*self.weight[0],0.0)  

        psf_fit = tf.math.real(I_res)
        if self.estdrift:
            psfsize = im.shapevec(I_model)
            Nz = psfsize[0]
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
        positions, backgrounds, intensities, psf_params,gxy = variables
        I_model = psf_params*self.weight[3]
        z_center = (psf_params.shape[-3] - 1) // 2
        #z_center = 0
        # calculate global positions in images since positions variable just represents the positions in the rois
        images, _, centers, _ = self.data.get_image_data()
        original_shape = images.shape[-3:]
        centers_with_z = np.concatenate((np.full((centers.shape[0], 1), z_center), centers[:,-2:]), axis=1)

        # use modulo operator to get rid of periodicity from FFT shifting
        global_positions = centers_with_z - positions
            
        # make sure everything has correct dtype
        # this is probably not needed anymore (see Fitter)
        # but just left since it does no harm
        return [global_positions.astype(np.float32),
                backgrounds*self.weight[1], # already correct
                intensities*self.weight[0], # already correct
                I_model,
                gxy*self.weight[2],
                variables] # already correct

    def res2dict(self,res):
        res_dict = dict(pos=res[0],
                        I_model=res[3],
                        bg=np.squeeze(res[1]),
                        intensity=np.squeeze(res[2]),
                        drift_rate=res[4],
                        offset=np.min(res[3]),
                        cor_all = self.data.centers_all,
                        cor = self.data.centers)

        return res_dict