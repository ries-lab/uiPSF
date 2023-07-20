
import numpy as np
import tensorflow as tf


from scipy.ndimage.filters import gaussian_filter
from .PSFInterface_file import PSFInterface
from ..data_representation.PreprocessedImageDataInterface_file import PreprocessedImageDataInterface
from ..loss_functions import mse_zernike_4pi
from .. import utilities as im
from .. import imagetools as nip

class PSFZernikeBased4pi(PSFInterface):
    def __init__(self, max_iter: int=None,options=None) -> None:
        
        self.parameters = None
        self.updateflag = None
        self.data = None
        self.Zphase = None
        self.zT = None
        self.dphase = None
        self.bead_kernel = None
        self.default_loss_func = mse_zernike_4pi
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
        
        init_intensities = np.sum(I_data - init_backgrounds, axis=(-2, -1), keepdims=True)     
        init_intensities = np.mean(init_intensities,axis=1,keepdims=True)  

        self.gen_bead_kernel()
        N = rois.shape[0]
        Nz = self.bead_kernel.shape[0]
        Lx = rois.shape[-1]
        self.calpupilfield('scalar')
        if self.options.model.const_pupilmag:
            self.n_max_mag = 0
        else:
            self.n_max_mag = 100


        sigma = np.ones((2,))*self.options.model.blur_sigma*np.pi

        self.Zphase = (np.linspace(-Nz/2+0.5,Nz/2-0.5,Nz,dtype=np.float32).reshape(Nz,1,1))*2*np.pi

        self.zT = self.data.zT
        #self.weight = np.array([np.median(init_intensities), 10, 0.1, 0.2,0.2,0.1],dtype=np.float32)
        #weight = [5e4,20] + list(np.array([0.1,0.3,0.3,0.1])/np.median(init_intensities)*2e4)
        wI = np.lib.scimath.sqrt(np.median(init_intensities))
        weight = [wI*40,20] + list(np.array([1,1,1,1])/wI*40)
        self.weight = np.array(weight,dtype=np.float32)
        
        init_Zcoeff_mag = np.zeros((2,self.Zk.shape[0],1,1))
        init_Zcoeff_mag[:,0,0,0] = [1,1]/self.weight[4]
        init_Zcoeff_phase = np.zeros((2,self.Zk.shape[0],1,1))
        
        
        phase_dm = self.options.fpi.phase_dm
        phase0 = np.reshape(np.array(phase_dm),(len(phase_dm),1,1,1,1)).astype(np.float32)
        
        init_backgrounds[init_backgrounds<0.1] = 0.1
        init_backgrounds = np.ones((N,1,1,1),dtype = np.float32)*np.median(init_backgrounds,axis=0, keepdims=True) / self.weight[1]
        
        gxy = np.zeros((N,2),dtype=np.float32) 
        gI = np.ones((N,rois.shape[-3],1,1),dtype = np.float32)*init_intensities
        alpha = np.array([0.8])/self.weight[5]
        init_pos_shift = np.zeros(init_positions.shape)
        self.varinfo = [dict(type='Nfit',id=0),
            dict(type='Nfit',id=0),
            dict(type='Nfit',id=0),
            dict(type='Nfit',id=0),
            dict(type='shared'),
            dict(type='shared'),
            dict(type='shared'),
            dict(type='shared'),
            dict(type='Nfit',id=0),
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
                init_Zcoeff_mag.astype(np.float32),
                init_Zcoeff_phase.astype(np.float32),
                sigma.astype(np.float32),
                alpha.astype(np.float32),
                init_pos_shift.astype(np.float32),
                phase0.astype(np.float32),                
                gxy], start_time


    def calc_forward_images(self, variables):
        """
        Calculate forward images from the current guess of the variables.
        """
                
        pos, bg, intensity_abs,intensity_phase, Zcoeffmag, Zcoeffphase, sigma,alpha,pos_d,phase0, gxy = variables
        intensity_phase = tf.complex(tf.math.cos(intensity_phase),tf.math.sin(intensity_phase))
        phase0 = tf.complex(tf.math.cos(phase0+self.dphase),tf.math.sin(phase0+self.dphase))
        pos = tf.complex(tf.reshape(pos,pos.shape+(1,1,1)),0.0)
        pos_d = tf.complex(tf.reshape(pos_d,pos_d.shape+(1,1,1)),0.0)
        c1 = self.spherical_terms
        n_max = self.n_max_mag
        Nk = np.min(((n_max+1)*(n_max+2)//2,self.Zk.shape[0]))
        mask = c1<Nk
        c1 = c1[mask]
        if self.options.model.symmetric_mag:
            pupil_mag1 = tf.reduce_sum(self.Zk[c1]*tf.gather(Zcoeffmag[0],indices=c1)*self.weight[4],axis=0)
            pupil_mag2 = tf.reduce_sum(self.Zk[c1]*tf.gather(Zcoeffmag[1],indices=c1)*self.weight[4],axis=0)

        else:
            pupil_mag1 = tf.reduce_sum(self.Zk[0:Nk]*Zcoeffmag[0][0:Nk]*self.weight[4],axis=0)
            pupil_mag2 = tf.reduce_sum(self.Zk[0:Nk]*Zcoeffmag[1][0:Nk]*self.weight[4],axis=0)
        pupil_mag1 = tf.math.maximum(pupil_mag1,0)
        pupil_mag2 = tf.math.maximum(pupil_mag2,0)

        pupil_phase = tf.reduce_sum(self.Zk[1:]*Zcoeffphase[0][1:]*self.weight[3],axis=0)
        pupil1 = tf.complex(pupil_mag1*tf.math.cos(pupil_phase),pupil_mag1*tf.math.sin(pupil_phase))*self.aperture*(self.apoid)

                
        pupil_phase = tf.reduce_sum(self.Zk*Zcoeffphase[1]*self.weight[3],axis=0)
        pupil2 = tf.complex(pupil_mag2*tf.math.cos(pupil_phase),pupil_mag2*tf.math.sin(pupil_phase))*self.aperture*(self.apoid)   

        phiz = -1j*2*np.pi*self.kz*(pos[:,0]+self.Zrange)
        phixy = 1j*2*np.pi*self.ky*pos[:,1]+1j*2*np.pi*self.kx*pos[:,2]
        phiz_d = 1j*2*np.pi*self.kz*(pos_d[:,0]+self.Zrange)
        phixy_d = 1j*2*np.pi*self.ky*pos_d[:,1]+1j*2*np.pi*self.kx*pos_d[:,2]

        PupilFunction = (pupil1*tf.exp(-phiz)*intensity_phase + pupil2*tf.exp(phiz)*phase0)*tf.exp(phixy)
        I_m = im.cztfunc1(PupilFunction,self.paramxy)   
        I_m = I_m*tf.math.conj(I_m)*self.normf/2.0

        PupilFunction1 = pupil1*tf.exp(-phiz)*tf.exp(phixy)
        I1 = im.cztfunc1(PupilFunction1,self.paramxy)   
        I1 = I1*tf.math.conj(I1)*self.normf/2.0

        PupilFunction2 = pupil2*tf.exp(phiz)*tf.exp(phixy)
        I2 = im.cztfunc1(PupilFunction2,self.paramxy)   
        I2 = I2*tf.math.conj(I2)*self.normf/2.0

        I_w = I1+I2
        alpha = tf.complex(alpha*self.weight[5],0.0)
        I_res = alpha*I_m + (1-alpha)*I_w
        filter2 = tf.exp(-2*sigma[1]*sigma[1]*self.kspace_x-2*sigma[0]*sigma[0]*self.kspace_y)

        filter2 = tf.complex(filter2/tf.reduce_max(filter2),0.0)
        I_blur = im.ifft3d(im.fft3d(I_res)*self.bead_kernel*filter2)
        
        psf_fit = tf.math.real(I_blur)
        Nz = psf_fit.shape[-3]
        st = (self.bead_kernel.shape[0]-self.data.rois[0].shape[-3])//2
        psf_fit = psf_fit[:,:,st:Nz-st]
                

        if self.options.model.estimate_drift:
            gxy = gxy*self.weight[2]
            psf_shift = self.applyDrfit(psf_fit,gxy)
            psf_shift = psf_shift*intensity_abs*self.weight[0] + bg*self.weight[1]
            forward_images = tf.transpose(psf_shift, perm = [1,0,2,3,4]) 
        else:
            psf_fit = psf_fit*intensity_abs*self.weight[0] + bg*self.weight[1]
            forward_images = tf.transpose(psf_fit,[1,0,2,3,4])
        return forward_images

    def genpsfmodel(self,Zcoeffmag, Zcoeffphase, sigma, alpha):
        phase0 = np.reshape(np.array([-2/3,0,2/3])*np.pi+self.dphase,(3,1,1,1)).astype(np.float32)
        phase0 = tf.complex(tf.math.cos(phase0),tf.math.sin(phase0))

        pupil_mag = tf.abs(tf.reduce_sum(self.Zk*Zcoeffmag[0],axis=0))
        pupil_phase = tf.reduce_sum(self.Zk[1:]*Zcoeffphase[0][1:],axis=0)
        pupil1 = tf.complex(pupil_mag*tf.math.cos(pupil_phase),pupil_mag*tf.math.sin(pupil_phase))*self.aperture*(self.apoid)

        pupil_mag = tf.abs(tf.reduce_sum(self.Zk*Zcoeffmag[1],axis=0))
        pupil_phase = tf.reduce_sum(self.Zk*Zcoeffphase[1],axis=0)
        pupil2 = tf.complex(pupil_mag*tf.math.cos(pupil_phase),pupil_mag*tf.math.sin(pupil_phase))*self.aperture*(self.apoid)  

        phiz = -1j*2*np.pi*self.kz*(self.Zrange)

        PupilFunction = (pupil1*tf.exp(-phiz) + pupil2*tf.exp(phiz)*phase0)
        I_m = im.cztfunc1(PupilFunction,self.paramxy)   
        I_m = I_m*tf.math.conj(I_m)*self.normf/2.0

        PupilFunction1 = pupil1*tf.exp(-phiz)
        I1 = im.cztfunc1(PupilFunction1,self.paramxy)   
        I1 = I1*tf.math.conj(I1)*self.normf/2.0

        PupilFunction2 = pupil2*tf.exp(phiz)
        I2 = im.cztfunc1(PupilFunction2,self.paramxy)  
        I2 = I2*tf.math.conj(I2)*self.normf/2.0

        I_w = I1+I2
        
        I_res = alpha*I_m + (1-alpha)*I_w
        filter2 = tf.exp(-2*sigma[1]*sigma[1]*self.kspace_x-2*sigma[0]*sigma[0]*self.kspace_y)
        filter2 = tf.complex(filter2/tf.reduce_max(filter2),0.0)
        
        Zphase = -self.Zphase/self.zT  
        zphase = tf.complex(tf.math.cos(Zphase),tf.math.sin(Zphase))
        #psf_model_bead = np.real(im.ifft3d(im.fft3d(I_res)*self.bead_kernel*filter2))
        psf_model = np.real(im.ifft3d(im.fft3d(I_res)*filter2))
        
        
        I_model,A_model,_,_ = self.psf2IAB(np.expand_dims(psf_model,axis=0))
        A_model = A_model[0]*zphase
        #I_model_bead,A_model_bead,_,_ = self.psf2IAB(np.expand_dims(psf_model_bead,axis=0))
        #A_model_bead = A_model_bead[0]*zphase

        return psf_model[1], I_model[0], A_model, pupil1, pupil2

    def postprocess(self, variables):
        """
        Applies postprocessing to the optimized variables. In this case calculates
        real positions in the image from the positions in the roi. Also, normalizes
        psf and adapts intensities and background accordingly.
        """
        pos, bg, intensity_abs,intensity_phase, Zcoeffmag, Zcoeffphase, sigma, alpha,pos_d,phasec, gxy = variables
        
        intensity_phase = tf.complex(tf.math.cos(intensity_phase),tf.math.sin(intensity_phase))
        intensities = intensity_abs*self.weight[0]*intensity_phase
        alpha = tf.complex(alpha*self.weight[5],0.0)

        Zcoeffmag=Zcoeffmag*self.weight[4]
        Zcoeffphase=Zcoeffphase*self.weight[3]
        psf_model, I_model, A_model, pupil1, pupil2 = self.genpsfmodel(Zcoeffmag,Zcoeffphase,sigma,alpha)
        gxy = gxy*self.weight[2]

        z_center = (I_model.shape[-3] - 1) // 2

        # calculate global positions in images since positions variable just represents the positions in the rois
        images, _, centers, _ = self.data.get_image_data()

        global_positions = np.swapaxes(np.vstack((pos[:,0]+z_center,centers[:,-2]-pos[:,-2],centers[:,-1]-pos[:,-1])),1,0)


        return [global_positions.astype(np.float32), 
                bg*self.weight[1], 
                intensities, 
                I_model, 
                A_model, 
                np.complex64(pupil1),
                np.complex64(pupil2),
                sigma,
                np.real(alpha),
                pos_d,
                phasec,
                gxy,
                Zcoeffmag,
                Zcoeffphase,
                np.flip(I_model,axis=-3),
                np.flip(A_model,axis=-3),
                variables]

    
    def res2dict(self,res):
        res_dict = dict(pos=res[0],                    
                        bg=np.squeeze(res[1]),
                        intensity=np.squeeze(res[2]),
                        I_model=res[3],
                        A_model=res[4],
                        pupil1 = res[5],
                        pupil2 = res[6],
                        sigma = np.squeeze(res[7])/np.pi,
                        modulation_depth = res[8],
                        obj_misalign = res[9],
                        phase_dm = np.squeeze(res[10]),
                        drift_rate=res[11],
                        zernike_coeff_mag = np.squeeze(res[12]),
                        zernike_coeff_phase = np.squeeze(res[13]),
                        I_model_reverse=res[14],
                        A_model_reverse=res[15],
                        offset=np.min(res[3]-2*np.abs(res[4])),
                        Zphase = np.array(self.Zphase),
                        zernike_polynomial = self.Zk,
                        apodization = self.apoid,
                        cor_all = self.data.centers_all,
                        cor = self.data.centers)

        return res_dict