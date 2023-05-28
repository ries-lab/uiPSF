
import numpy as np
import tensorflow as tf


from scipy.ndimage.filters import gaussian_filter
from .PSFInterface_file import PSFInterface
from ..data_representation.PreprocessedImageDataInterface_file import PreprocessedImageDataInterface
from ..loss_functions import mse_zernike_4pi_smlm
from .. import utilities as im
from .. import imagetools as nip
from ..loclib import localizationlib


class PSFZernikeBased4pi_smlm(PSFInterface):
    def __init__(self, max_iter: int=None,options=None) -> None:
        
        self.parameters = None
        self.updateflag = None
        self.data = None
        self.Zphase = None
        self.zT = None
        self.dphase = None
        self.bead_kernel = None
        self.default_loss_func = mse_zernike_4pi_smlm
        self.options = options
        self.Zoffset = None
        if max_iter is None:
            self.max_iter = 10
        else:
            self.max_iter = max_iter


    def calc_initials(self, data: PreprocessedImageDataInterface,start_time=None):
        """
        Provides initial values for the optimizable varibales for the fitter class.
        """
        self.data = data
        _, rois, centers, frames = self.data.get_image_data()
        options = self.options
        alpha = np.array([0.8])
        #self.stagepos = options.insitu.stage_pos/self.data.pixelsize_z
        init_sigma = np.ones((2,),dtype=np.float32)*options.model.blur_sigma*np.pi

        if hasattr(self,'initpsf'):
            I_model = self.initpsf
            A_model = self.initA
            Nz = I_model.shape[0]
            if self.Zoffset is None:
            #    self.estzoffset(Nz)
                self.Zoffset = -Nz/2+0.5
            self.calpupilfield('scalar', Nz,'insitu')
            self.Zphase = (np.linspace(-Nz/2+0.5,Nz/2-0.5,Nz,dtype=np.float32).reshape(Nz,1,1))*2*np.pi
            self.zT = self.data.zT
            
        else:
            #self.estzoffset()
            #self.Zrange -=self.Zrange[0]-self.Zoffset
            Nz = np.int32(self.options.insitu.z_range/self.data.pixelsize_z+1)
            self.calpupilfield('scalar', Nz,'insitu')
            self.Zphase = (np.linspace(-Nz/2+0.5,Nz/2-0.5,Nz,dtype=np.float32).reshape(Nz,1,1))*2*np.pi
            self.zT = self.data.zT
            self.Zoffset = -Nz/2+0.5
            init_Zcoeffmag = np.zeros((2,self.Zk.shape[0],1,1),dtype=np.float32)
            init_Zcoeffmag[:,0,0,0] = 1
            init_Zcoeffphase = np.zeros((2,self.Zk.shape[0],1,1),dtype=np.float32)
            init_Zcoeffphase[0,options.insitu.zernike_index,0,0] = options.insitu.zernike_coeff
            init_Zcoeffphase[1,options.insitu.zernike_index,0,0] = -1.0*np.array(options.insitu.zernike_coeff)
        
            I_init, I_model,A_model,_,_ = self.genpsfmodel(init_Zcoeffmag,init_Zcoeffphase,init_sigma,alpha)

        self.I_model = I_model
        self.A_model = A_model

        if self.options.model.const_pupilmag:
            self.n_max_mag = 0
        else:
            self.n_max_mag = 100



        self.weight = np.array([1e4, 100, 20, 0.2,0.2,0.1],dtype=np.float32)
        self.pos_weight = self.weight[2]
        init_Zcoeff_mag = np.zeros((2,self.Zk.shape[0],1,1))
        init_Zcoeff_mag[:,0,0,0] = [1,1]/self.weight[4]
        init_Zcoeff_phase = np.zeros((2,self.Zk.shape[0],1,1))
                    
        init_intensities = np.zeros((rois.shape[0], 1,1))
        init_backgrounds = np.zeros((rois.shape[0], 1,1))
        init_phi = np.zeros((rois.shape[0], 1,1))
        init_positions = np.zeros((rois.shape[0], 3))


        
        init_backgrounds[init_backgrounds<0.1] = 0.1
        init_backgrounds = init_backgrounds / self.weight[1]
        
        alpha = np.array([0.8])/self.weight[5]
        self.varinfo = [dict(type='Nfit',id=0),
            dict(type='Nfit',id=0),
            dict(type='Nfit',id=0),
            dict(type='Nfit',id=0),
            dict(type='shared'),
            dict(type='shared'),
            dict(type='shared'),
            dict(type='shared')]

        init_Intensity = init_intensities / self.weight[0]

        return [init_positions.astype(np.float32), 
                init_backgrounds.astype(np.float32), 
                init_Intensity.astype(np.float32),
                init_phi.astype(np.float32),
                init_Zcoeff_mag.astype(np.float32),
                init_Zcoeff_phase.astype(np.float32),
                init_sigma.astype(np.float32),
                alpha.astype(np.float32)], start_time


    def calc_forward_images(self, variables):
        """
        Calculate forward images from the current guess of the variables.
        """
                
        pos, bg, intensity_abs,intensity_phase, Zcoeffmag, Zcoeffphase, sigma,alpha = variables
        intensity_phase = tf.complex(tf.math.cos(intensity_phase*self.weight[2]),tf.math.sin(intensity_phase*self.weight[2]))
        phase0 = tf.complex(tf.math.cos(self.dphase),tf.math.sin(self.dphase))
        pos = tf.complex(tf.reshape(pos*self.weight[2],pos.shape+(1,1)),0.0)
        c1 = self.spherical_terms
        n_max = self.n_max_mag
        Nk = np.min(((n_max+1)*(n_max+2)//2,self.Zk.shape[0]))
        mask = c1<Nk
        c1 = c1[mask]
        if self.options.model.symmetric_mag:
            pupil_mag1 = tf.abs(tf.reduce_sum(self.Zk[c1]*tf.gather(Zcoeffmag[0],indices=c1)*self.weight[4],axis=0))
            pupil_mag2 = tf.abs(tf.reduce_sum(self.Zk[c1]*tf.gather(Zcoeffmag[1],indices=c1)*self.weight[4],axis=0))

        else:
            pupil_mag1 = tf.abs(tf.reduce_sum(self.Zk[0:Nk]*Zcoeffmag[0][0:Nk]*self.weight[4],axis=0))
            pupil_mag2 = tf.abs(tf.reduce_sum(self.Zk[0:Nk]*Zcoeffmag[1][0:Nk]*self.weight[4],axis=0))

        pupil_phase = tf.reduce_sum(self.Zk[1:]*Zcoeffphase[0][1:]*self.weight[3],axis=0)
        pupil1 = tf.complex(pupil_mag1*tf.math.cos(pupil_phase),pupil_mag1*tf.math.sin(pupil_phase))*self.aperture*(self.apoid)

                
        pupil_phase = tf.reduce_sum(self.Zk*Zcoeffphase[1]*self.weight[3],axis=0)
        pupil2 = tf.complex(pupil_mag2*tf.math.cos(pupil_phase),pupil_mag2*tf.math.sin(pupil_phase))*self.aperture*(self.apoid)   

        phiz = 1j*2*np.pi*(self.kz-self.k)*(pos[:,0])
        phixy = 1j*2*np.pi*self.ky*pos[:,1]+1j*2*np.pi*self.kx*pos[:,2]

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
        I_blur = im.ifft3d(im.fft3d(I_res)*filter2)
        
        forward_images = tf.math.real(I_blur)*intensity_abs*self.weight[0] + bg*self.weight[1]
                


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

        phiz = 1j*2*np.pi*self.kz*(self.Zrange)

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
        
        Zphase = self.Zphase/self.zT  
        zphase = tf.complex(tf.math.cos(Zphase),tf.math.sin(Zphase))
        #psf_model_bead = np.real(im.ifft3d(im.fft3d(I_res)*self.bead_kernel*filter2))
        psf_model = np.real(im.ifft3d(im.fft3d(I_res)*filter2))
        
        
        I_model,A_model,_,_ = self.psf2IAB(np.expand_dims(psf_model,axis=0))
        A_model = A_model[0]*zphase
        #I_model_bead,A_model_bead,_,_ = self.psf2IAB(np.expand_dims(psf_model_bead,axis=0))
        #A_model_bead = A_model_bead[0]*zphase

        return psf_model[1], I_model[0], A_model, pupil1, pupil2
    
    def partitiondata(self,zf,LL):
        _, rois, centers, frames = self.data.get_image_data()
        
        nbin = self.options.insitu.partition_size[0]
        count,edge = np.histogram(zf,nbin)
        ind = np.digitize(zf,edge)
        Npsf = self.options.insitu.partition_size[1]
        rois1 = []
        rois1_avg = []
        zf1 = []
        cor = []
        fid = []
        id = []
        ids = np.array(range(0,zf.shape[0]))
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
            id.append(ids[mask][indsample])
        rois1 = np.concatenate(rois1,axis=0)
        rois1_avg = np.stack(rois1_avg)
        zf1 = np.concatenate(zf1,axis=0)
        id = np.concatenate(id,axis=0)
        
        self.data.rois = rois1
        self.data.centers = np.concatenate(cor,axis=0)
        self.data.frames = np.concatenate(fid,axis=0)
        return zf1, rois1_avg, id

    def postprocess(self, variables):
        """
        Applies postprocessing to the optimized variables. In this case calculates
        real positions in the image from the positions in the roi. Also, normalizes
        psf and adapts intensities and background accordingly.
        """
        pos, bg, intensity_abs,intensity_phase, Zcoeffmag, Zcoeffphase, sigma, alpha = variables
        res = variables.copy()
        intensity_phase = tf.complex(tf.math.cos(intensity_phase*self.weight[2]),tf.math.sin(intensity_phase*self.weight[2]))
        intensities = intensity_abs*self.weight[0]*intensity_phase
        alpha = tf.complex(alpha*self.weight[5],0.0)
        pos = pos*self.weight[2]

        Zcoeffmag=Zcoeffmag*self.weight[4]
        Zcoeffphase=Zcoeffphase*self.weight[3]
        psf_model, I_model, A_model, pupil1, pupil2 = self.genpsfmodel(Zcoeffmag,Zcoeffphase,sigma,alpha)

        #z_center = (I_model.shape[-3] - 1) // 2

        # calculate global positions in images since positions variable just represents the positions in the rois
        images, _, centers, _ = self.data.get_image_data()

        global_positions = np.swapaxes(np.vstack((pos[:,0],centers[:,-2]-pos[:,-2],centers[:,-1]-pos[:,-1])),1,0)


        return [global_positions.astype(np.float32), 
                bg*self.weight[1], 
                intensities, 
                I_model, 
                A_model, 
                psf_model,
                np.complex64(pupil1),
                np.complex64(pupil2),
                sigma,
                np.real(alpha),
                Zcoeffmag,
                Zcoeffphase,
                res]

    
    def res2dict(self,res):
        res_dict = dict(pos=res[0],                    
                        bg=np.squeeze(res[1]),
                        intensity=np.squeeze(res[2]),
                        I_model=res[3],
                        A_model=res[4],
                        psf_model = res[5],
                        pupil1 = res[6],
                        pupil2 = res[7],
                        sigma = np.squeeze(res[8])/np.pi,
                        modulation_depth = res[9],
                        zernike_coeff_mag = np.squeeze(res[10]),
                        zernike_coeff_phase = np.squeeze(res[11]),
                        offset=np.min(res[3]-2*np.abs(res[4])),
                        Zphase = np.array(self.Zphase),
                        zernike_polynomial = self.Zk,
                        apodization = self.apoid,
                        zoffset = self.Zoffset,
                        cor_all = self.data.centers_all,
                        cor = self.data.centers)

        return res_dict