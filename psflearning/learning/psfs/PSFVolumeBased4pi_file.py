
import numpy as np
import tensorflow as tf


from scipy.ndimage.filters import gaussian_filter
from .PSFInterface_file import PSFInterface
from ..data_representation.PreprocessedImageDataInterface_file import PreprocessedImageDataInterface
from ..loss_functions import mse_real_4pi
from .. import utilities as im
from .. import imagetools as nip

class PSFVolumeBased4pi(PSFInterface):
    def __init__(self, max_iter: int=None,options=None) -> None:
        
        self.parameters = None
        self.updateflag = None
        self.data = None
        self.Zphase = None
        self.zT = None
        self.bead_kernel = None
        self.default_loss_func = mse_real_4pi
        self.options = options
        if max_iter is None:
            self.max_iter = 10
        else:
            self.max_iter = max_iter


    def calc_initials(self, data: PreprocessedImageDataInterface,start_time=None):
        """
        Provides initial values for the optimizable varibales for the fitter class.
        """
        self.data = data
        _, rois, _, _ = self.data.get_image_data() # TODO: check if file_idx are returned at all
        N = rois.shape[0]
        Nz = rois.shape[-3]
        I_data, A_data, _, init_phi = self.psf2IAB(rois)
        #init_phi = np.reshape(init_phi,(I_data.shape[0],1,1,1))
        init_phi = np.zeros((I_data.shape[0],1,1,1))
        init_positions = np.zeros([I_data.shape[0],len(I_data.shape)-1]).astype(np.float32)               
        init_backgrounds = np.min(gaussian_filter(I_data, [0, 2, 2, 2]), axis=(-3, -2, -1), keepdims=True)
        init_intensities = np.sum(I_data - init_backgrounds, axis=(-2, -1), keepdims=True)     
        init_intensities = np.mean(init_intensities,axis=1,keepdims=True)  
        self.gen_bead_kernel(isVolume=True)

        self.zT = self.data.zT
        #self.weight = np.array([np.quantile(init_intensities,0.1), 20, 0.1, 0.1],dtype=np.float32)
        #weight = [1e4,20] + list(np.array([0.3,0.2])/np.median(init_intensities)*2e4)
        wI = np.lib.scimath.sqrt(np.median(init_intensities))
        weight = [N*wI,20] + list(np.array([1,1])*N/wI)
        self.weight = np.array(weight,dtype=np.float32)
        I1 = np.zeros(I_data[0].shape,dtype=np.float32)+0.002 / self.weight[3]
        A1 = np.ones(I1.shape, dtype=np.float32)*(1+1j)*0.002/2/np.sqrt(2)/self.weight[3]    
        phase_dm = self.options.fpi.phase_dm
        phase = np.reshape(np.array(phase_dm)*-1,(len(phase_dm),1,1,1,1)).astype(np.float32)

        self.calpupilfield('scalar',Nz)
        self.Zphase = (np.linspace(-Nz/2+0.5,Nz/2-0.5,Nz,dtype=np.float32).reshape(Nz,1,1))*2*np.pi
        
        init_backgrounds[init_backgrounds<0.1] = 0.1
        init_backgrounds = np.ones((N,1,1,1),dtype = np.float32)*np.median(init_backgrounds,axis=0, keepdims=True) / self.weight[1]
        
        gxy = np.zeros((N,2),dtype=np.float32) 
        gI = np.ones((N,Nz,1,1),dtype = np.float32)*init_intensities
        self.varinfo = [dict(type='Nfit',id=0),
            dict(type='Nfit',id=0),
            dict(type='Nfit',id=0),
            dict(type='Nfit',id=0),
            dict(type='shared'),
            dict(type='shared'),
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
                init_phi.astype(np.float32),
                I1.astype(np.float32), 
                np.real(A1).astype(np.float32),
                np.imag(A1).astype(np.float32), 
                phase, 
                gxy], start_time


    def calc_forward_images(self, variables):
        """
        Calculate forward images from the current guess of the variables.
        """

   
        pos = variables[0]
        bg = variables[1]
        intensity_abs = variables[2]*self.weight[0]
        intensity_phase = tf.complex(tf.math.cos(variables[3]),tf.math.sin(variables[3]))
        I_model = variables[4]*self.weight[3]        
        A_model = tf.complex(variables[5],variables[6])*self.weight[3]
        phase0 = tf.complex(tf.math.cos(variables[7]), tf.math.sin(variables[7]))
        Zphase = self.Zphase/self.zT  
        zphase = tf.complex(tf.math.cos(Zphase),tf.math.sin(Zphase))
        

        I_model = tf.complex(I_model,0.0)
        I_otfs = im.fft3d(I_model)*tf.complex(intensity_abs*0.0+1.0,0.0)
        pos = tf.complex(tf.reshape(pos,pos.shape+(1,1,1)),0.0)
        I_res = im.ifft3d(I_otfs*self.phaseRamp(pos))    
        I_res = tf.math.real(I_res)
   
        A_otfs = im.fft3d(A_model*zphase)*intensity_phase
        A_res = im.ifft3d(A_otfs*self.phaseRamp(pos))

        psf0 = (I_res)*tf.math.abs(phase0) + tf.math.real(A_res*phase0)*2
        psf_otfs = im.fft3d(tf.complex(psf0,0.0))*tf.expand_dims(tf.expand_dims(self.bead_kernel,axis=0),axis=0)
        psfmodel = tf.math.real(im.ifft3d(psf_otfs)) 

        if self.options.model.estimate_drift:
            gxy = variables[8]*self.weight[2]
            psf_shift = self.applyDrfit(psfmodel,gxy)
            psf_shift = psf_shift* intensity_abs + bg*self.weight[1]
            forward_images = tf.transpose(psf_shift, perm = [1,0,2,3,4]) 
        else:
            psfmodel = psfmodel* intensity_abs + bg*self.weight[1]
            forward_images = tf.transpose(psfmodel, perm = [1,0,2,3,4]) 
        return forward_images

    

    def postprocess(self, variables):
        """
        Applies postprocessing to the optimized variables. In this case calculates
        real positions in the image from the positions in the roi. Also, normalizes
        psf and adapts intensities and background accordingly.
        """


        positions = variables[0]
        backgrounds = variables[1] * self.weight[1]  
        intensities = variables[2] * np.exp(1j*variables[3]) * self.weight[0]
        I_model = variables[4]*self.weight[3]
        A_model = (variables[5] + 1j*variables[6])*self.weight[3]
        phase = variables[7]
        gxy = variables[8]*self.weight[2]
        z_center = I_model.shape[-3] // 2

        # calculate global positions in images since positions variable just represents the positions in the rois
        images, _, centers, _ = self.data.get_image_data()

        centers_with_z = np.concatenate((np.full((centers.shape[0], 1), z_center), centers), axis=1)
        global_positions = centers_with_z - positions
        return [global_positions, 
                backgrounds, 
                intensities, 
                I_model, 
                A_model, 
                phase,
                gxy,
                np.flip(I_model,axis=-3),
                np.flip(A_model,axis=-3),
                variables]

    
    def res2dict(self,res):
        res_dict = dict(pos=res[0],                    
                        bg=np.squeeze(res[1]),
                        intensity=np.squeeze(res[2]),
                        I_model=res[3],
                        A_model=res[4],
                        phase_dm = np.squeeze(res[5]),
                        drift_rate=res[6],
                        I_model_reverse=res[7],
                        A_model_reverse=res[8],
                        offset=np.min(res[3]-2*np.abs(res[4])),
                        Zphase = self.Zphase,
                        cor_all = self.data.centers_all,
                        cor = self.data.centers)

        return res_dict