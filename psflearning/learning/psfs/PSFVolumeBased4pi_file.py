
import numpy as np
import tensorflow as tf


from scipy.ndimage.filters import gaussian_filter
from .PSFInterface_file import PSFInterface
from ..data_representation.PreprocessedImageDataInterface_file import PreprocessedImageDataInterface
from ..loss_functions import mse_real_4pi
from .. import utilities as im
from .. import imagetools as nip

class PSFVolumeBased4pi(PSFInterface):
    def __init__(self, max_iter: int=None, estdrift=False, varphoton=False,options=None) -> None:
        
        self.parameters = None
        self.updateflag = None
        self.data = None
        self.Zphase = None
        self.zT = None
        self.bead_kernel = None
        self.default_loss_func = mse_real_4pi
        self.estdrift = estdrift
        self.varphoton = varphoton
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

        I_data, A_data, _, init_phi = self.psf2IAB(rois)
        #I_data = np.sum(rois,axis=-4)/rois.shape[-4]
        #init_phi = np.reshape(init_phi,(I_data.shape[0],1,1,1))
        init_phi = np.zeros((I_data.shape[0],1,1,1))
        init_positions = np.zeros([I_data.shape[0],len(I_data.shape)-1]).astype(np.float32)               
        init_backgrounds = np.min(gaussian_filter(I_data, [0, 2, 2, 2]), axis=(-3, -2, -1), keepdims=True)
        #init_intensities = np.max(I_data - init_backgrounds, axis=(-3, -2, -1), keepdims=True)
        init_intensities = np.sum(I_data - init_backgrounds, axis=(-2, -1), keepdims=True)     
        init_intensities = np.mean(init_intensities,axis=1,keepdims=True)  

        self.bead_kernel = tf.complex(self.data.bead_kernel,0.0)
        self.zT = self.data.zT
        self.weight = np.array([np.quantile(init_intensities,0.1), 20, 0.1, 0.1],dtype=np.float32)
        I1 = np.zeros(I_data[0].shape,dtype=np.float32)+0.002 / self.weight[3]
        A1 = np.ones(I1.shape, dtype=np.float32)*(1+1j)*0.002/2/np.sqrt(2)/self.weight[3]    
        phase_dm = self.options['phase_dm']
        phase = np.reshape(np.array(phase_dm)*-1,(len(phase_dm),1,1,1,1)).astype(np.float32)
        #phase = np.reshape(np.array([2/3,0,-2/3])*np.pi,(3,1,1,1,1)).astype(np.float32)
        #phase = np.reshape(np.array([0])*np.pi,(1,1,1,1,1)).astype(np.float32)
        N = rois.shape[0]
        Nz = rois.shape[-3]
        Zphase = tf.cast(2*np.pi*nip.zz(I_data[0].shape),tf.float32)
        self.Zphase = Zphase
        
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
        #intensity = intensity_abs*intensity_phase
        I_model = variables[4]*self.weight[3]        
        A_model = tf.complex(variables[5],variables[6])*self.weight[3]
        phase0 = tf.complex(tf.math.cos(variables[7]), tf.math.sin(variables[7]))
        Zphase = self.Zphase/self.zT  
        zphase = tf.complex(tf.math.cos(Zphase),tf.math.sin(Zphase))
        
        I_otfs = im.ft3d(I_model)*tf.complex(intensity_abs*0.0+1.0,0.0)
        I_res = im.ift3d(self.applyPhaseRamp(I_otfs,pos))    
        I_res = tf.math.real(I_res)
   
        A_otfs = im.ft3d(A_model*zphase)*intensity_phase
        A_res = im.ift3d(self.applyPhaseRamp(A_otfs,pos))

        psf0 = (I_res)*tf.math.abs(phase0) + tf.math.real(A_res*phase0)*2
        psf_otfs = im.ft3d(psf0)*tf.expand_dims(tf.expand_dims(self.bead_kernel,axis=0),axis=0)
        psfmodel = tf.math.real(im.ift3d(psf_otfs)) * intensity_abs + bg*self.weight[1]

        if self.estdrift:
            psf_fit = tf.transpose(psfmodel,perm=[1,2,0,3,4]) 
            psfsize = im.shapevec(I_model)
            Nz = psfsize[0]
            zv = np.expand_dims(np.linspace(0,Nz-1,Nz,dtype=np.float32)-Nz/2,axis=-1)

            gxy = variables[8]*self.weight[2]
            otf2d = im.ft(psf_fit,axes=[-1,-2])
            otf2dphase = otf2d[0:1]
            for i,g in enumerate(gxy):
                dxy = g*zv
                
                tmp = self.applyPhaseRamp(otf2d[i],dxy)
                otf2dphase = tf.concat((otf2dphase,tf.expand_dims(tmp,axis=0)),axis=0)

            psf_shift = tf.math.real(im.ift(otf2dphase[1:],axes=[-1,-2])) 


            forward_images = tf.transpose(psf_shift, perm = [0,2,1,3,4]) 
        else:
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


        return [global_positions, backgrounds, intensities, I_model, A_model, phase,gxy,variables]

    
    def res2dict(self,res):
        res_dict = dict(pos=res[0],                    
                        bg=np.squeeze(res[1]),
                        intensity=np.squeeze(res[2]),
                        I_model=res[3],
                        A_model=res[4],
                        phase_dm = np.squeeze(res[5]),
                        drift_rate=res[6],
                        offset=np.min(res[3]-2*np.abs(res[4])),
                        Zphase = self.Zphase,
                        cor_all = self.data.centers_all,
                        cor = self.data.centers)

        return res_dict