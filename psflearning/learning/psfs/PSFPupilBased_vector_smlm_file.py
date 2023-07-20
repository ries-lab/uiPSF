import numpy as np
import scipy as sp
import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter
from .PSFInterface_file import PSFInterface
from ..data_representation.PreprocessedImageDataInterface_file import PreprocessedImageDataInterface
from ..loss_functions import mse_real_pupil_smlm
from .. import utilities as im
from .. import imagetools as nip
from ..loclib import localizationlib

class PSFPupilBased_vector_smlm(PSFInterface):
    """
    PSF class that uses a 3D volume to describe the PSF.
    Should only be used with single-channel data.
    """
    def __init__(self, options=None) -> None:
        self.parameters = None
        self.data = None

        self.Zphase = None
        self.zT = None
        self.options = options
        self.default_loss_func = mse_real_pupil_smlm
        self.pos_weight = 1
        self.Zoffset = None
        return

    def calc_initials(self, data: PreprocessedImageDataInterface, start_time=None):
        """
        Provides initial values for the optimizable varibales for the fitter class.
        """
        options = self.options
        self.data = data
        _, rois, centers, frames = self.data.get_image_data()
        pixelsize_z = np.array(self.data.pixelsize_z)
        xsz =options.model.pupilsize

        self.stagepos = options.insitu.stage_pos/self.data.pixelsize_z
        if hasattr(self,'initpsf'):
            I_init = self.initpsf
            Nz = I_init.shape[0]
            if self.Zoffset is None:
                self.estzoffset(Nz)
            self.calpupilfield('vector', Nz,'insitu')
            self.Zrange -=self.Zrange[0]-self.Zoffset
        elif not options.insitu.zernike_index:
            self.estzoffset()
            I_init = self.estzernike(start_time=start_time)
        else:
            self.estzoffset()
            self.Zrange -=self.Zrange[0]-self.Zoffset
            init_sigma = np.ones((2,),dtype=np.float32)*options.model.blur_sigma*np.pi
            init_Zcoeff = np.zeros((2,self.Zk.shape[0],1,1),dtype=np.float32)
            init_Zcoeff[0,0,0,0] = 1
            init_Zcoeff[1,options.insitu.zernike_index,0,0] = options.insitu.zernike_coeff
            I_init = self.genpsfmodel(init_sigma,Zcoeff=init_Zcoeff)
        
        dll = localizationlib(usecuda=True)
        locres = dll.loc_ast(rois,I_init,pixelsize_z,start_time=start_time)

        xp = locres[-1]['x']
        yp = locres[-1]['y']
        zp = locres[-1]['z']     
        photon = locres[0][2]  
        bg = locres[0][3]
        a = 0.99
        a1 = options.insitu.min_photon
        mask = (xp>np.quantile(xp,1-a)) & (xp<np.quantile(xp,a)) & (yp>np.quantile(yp,1-a)) & (yp<np.quantile(yp,a)) & (zp>np.quantile(zp,1-a)) & (zp<np.quantile(zp,a))
        mask = mask.flatten() & (locres[2]>np.quantile(locres[2],0.1)) & (photon>np.quantile(photon,a1))
        

        self.data.rois = rois[mask]
        self.data.centers = centers[mask,:]
        self.data.frames = frames[mask]
        initz = zp.flatten()[mask]
        LL = locres[2][mask]
        
        if self.options.insitu.partition_data:
            initz, roisavg = self.partitiondata(initz,LL)
            
        _, rois, _, _ = self.data.get_image_data()
        self.zweight = np.ones(initz.shape+(1,1),dtype=np.float32)

        init_positions = np.zeros((rois.shape[0], 3))
       
        init_positions[:,0] = initz+np.real(self.Zoffset)
        init_backgrounds = np.array(np.min(gaussian_filter(rois, [0, 2, 2]), axis=(-2, -1), keepdims=True))
        init_intensities = np.sum(rois - init_backgrounds, axis=(-2, -1), keepdims=True)
                                       
        if self.options.model.const_pupilmag:
            self.n_max_mag = 0
        else:
            self.n_max_mag = 100

        
        #self.weight = np.array([np.median(init_intensities), 10, 5, 10, 10, 5],dtype=np.float32) # [I, bg, pos, coeff, stagepos]
        weight = [1e4,10] + list(np.array([1,10,10,1])/np.median(init_intensities)*2e4)
        self.weight = np.array(weight,dtype=np.float32)
        sigma = np.ones((2,))*self.options.model.blur_sigma*np.pi*self.options.model.bin
        self.init_sigma = sigma
        self.pos_weight = self.weight[2]

        init_pupil = np.zeros((xsz,xsz))+(1+0.0*1j)/self.weight[4]

        init_backgrounds[init_backgrounds<0.1] = 0.1
        init_backgrounds = init_backgrounds / self.weight[1]
        init_Intensity = init_intensities / self.weight[0]
        init_positions = init_positions / self.weight[2]
        init_stagepos = np.ones((1,))*self.stagepos / self.weight[5]
        self.init_stagepos = init_stagepos.astype(np.float32)

        self.varinfo = [dict(type='Nfit',id=0),
            dict(type='Nfit',id=0),
            dict(type='Nfit',id=0),
            dict(type='shared'),
            dict(type='shared'),
            dict(type='shared'),
            dict(type='shared')]

        return [init_positions.astype(np.float32),
                init_backgrounds.astype(np.float32),
                init_Intensity.astype(np.float32),
                np.real(init_pupil).astype(np.float32),
                np.imag(init_pupil).astype(np.float32),
                sigma.astype(np.float32), 
                init_stagepos.astype(np.float32)], start_time
        
    def calc_forward_images(self, variables):
        """
        Calculate forward images from the current guess of the variables.
        Shifting is done by Fourier transform and applying a phase ramp.
        """
        pos, backgrounds, intensities, pupilR,pupilI, sigma, stagepos = variables

        if self.options.model.const_pupilmag:
            pupil_mag = tf.complex(1.0,0.0)
        else:
            pupil_mag = tf.complex(pupilR*self.weight[4],0.0)

        pupil = tf.complex(tf.math.cos(pupilI*self.weight[3]),tf.math.sin(pupilI*self.weight[3]))*pupil_mag*self.aperture*self.apoid
        
        pos = tf.complex(tf.reshape(pos*self.weight[2],pos.shape+(1,1)),0.0)
        if self.options.insitu.var_stagepos:
            stagepos = tf.complex(stagepos*self.weight[5],0.0)
        else:
            stagepos = tf.complex(self.init_stagepos*self.weight[5],0.0)

        phiz = 1j*2*np.pi*(self.kz_med*pos[:,0]*self.zweight[self.ind[0]:self.ind[1]]-self.kz*stagepos)
        phixy = 1j*2*np.pi*self.ky*pos[:,1]+1j*2*np.pi*self.kx*pos[:,2]
        I_res = 0.0
        for h in self.dipole_field:
            PupilFunction = pupil*tf.exp(phiz+phixy)*h
            psfA = im.cztfunc1(PupilFunction,self.paramxy)       
            I_res += psfA*tf.math.conj(psfA)*self.normf

        bin = self.options.model.bin
        if not self.options.model.var_blur:
            sigma = self.init_sigma
        filter2 = tf.exp(-2*sigma[1]*sigma[1]*self.kspace_x-2*sigma[0]*sigma[0]*self.kspace_y)
        filter2 = tf.complex(filter2/tf.reduce_max(filter2),0.0)
        I_blur = im.ifft2d(im.fft2d(I_res)*filter2)
        I_blur = tf.expand_dims(tf.math.real(I_blur),axis=-1)
        kernel = np.ones((bin,bin,1,1),dtype=np.float32)
        I_blur_bin = tf.nn.convolution(I_blur,kernel,strides=(1,bin,bin,1),padding='SAME',data_format='NHWC')

        psf_fit = I_blur_bin[...,0]*intensities*self.weight[0]
        
        forward_images = psf_fit + backgrounds*self.weight[1]

        return forward_images

    def estzoffset(self,Nz=None):
        if Nz is None:
            Nz = np.int32(self.options.insitu.z_range/self.data.pixelsize_z+1)
        self.calpupilfield('vector', Nz,'insitu')
        self.Zrange += self.stagepos*self.options.imaging.RI.med/self.options.imaging.RI.imm
        if self.Zrange[0]<0:
            self.Zrange -=self.Zrange[0]
        init_sigma = np.ones((2,),dtype=np.float32)*self.options.model.blur_sigma*np.pi
        init_Zcoeff = np.zeros((2,self.Zk.shape[0],1,1),dtype=np.float32)
        init_Zcoeff[0,0,0,0] = 1
        I_init = self.genpsfmodel(init_sigma,Zcoeff=init_Zcoeff)
        ccz = np.argmax(np.max(I_init,axis=(-1,-2)))
        self.Zrange += self.Zrange[ccz]-self.Zrange[Nz//2]
        if self.Zrange[0]<0:
            self.Zrange -=self.Zrange[0]
        self.Zoffset = self.Zrange[0]

        return

    def estzernike(self,start_time = None):
        pixelsize_z = np.array(self.data.pixelsize_z)
        init_sigma = np.ones((2,),dtype=np.float32)*self.options.model.blur_sigma*np.pi

        llmax = -1e6
        LLavg = []
        if self.options.insitu.zkorder_rank == 'H':
            zkrange = range(21,45)
        else:
            zkrange = range(4,21)
        coeffamp = np.abs(self.options.insitu.zernike_coeff)
        for k in zkrange:
            for val in [-coeffamp,coeffamp]:
                init_Zcoeff = np.zeros((2,self.Zk.shape[0],1,1),dtype=np.float32)
                init_Zcoeff[0,0,0,0] = 1
                init_Zcoeff[1,k,0,0] = val
                I_init = self.genpsfmodel(init_sigma,Zcoeff=init_Zcoeff)
                
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

    def genpsfmodel(self,sigma,Zcoeff=None,stagepos=None,pupil=None):
        if pupil is None:
            pupil_mag = tf.abs(tf.reduce_sum(self.Zk*Zcoeff[0],axis=0))
            pupil_phase = tf.reduce_sum(self.Zk*Zcoeff[1],axis=0)
            pupil = tf.complex(pupil_mag*tf.math.cos(pupil_phase),pupil_mag*tf.math.sin(pupil_phase))*self.aperture*self.apoid

        if stagepos is None:
            stagepos = self.stagepos

        zrange = self.Zrange
        phiz = 1j*2*np.pi*(self.kz_med*zrange-self.kz*stagepos)  
        phixy = 1j*2*np.pi*self.ky*0.0+1j*2*np.pi*self.kx*0.0
        I_res = 0.0
        for h in self.dipole_field:
            PupilFunction = pupil*tf.exp(phiz+phixy)*h
            psfA = im.cztfunc1(PupilFunction,self.paramxy)          
            I_res += psfA*tf.math.conj(psfA)*self.normf

        
        filter2 = tf.exp(-2*sigma[1]*sigma[1]*self.kspace_x-2*sigma[0]*sigma[0]*self.kspace_y)

        filter2 = tf.complex(filter2/tf.reduce_max(filter2),0.0)
        I_blur = im.ifft3d(im.fft3d(I_res)*filter2)
        I_blur = tf.expand_dims(tf.math.real(I_blur),axis=-1)
        bin = self.options.model.bin
        kernel = np.ones((bin,bin,1,1),dtype=np.float32)
        I_blur_bin = tf.nn.convolution(I_blur,kernel,strides=(1,bin,bin,1),padding='SAME',data_format='NHWC')
        I_model = np.real(I_blur_bin[...,0])
        
        return I_model

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
        for ii in range(1,nbin+1):
            mask = (ind==ii)
            im1 = rois[mask]
            Nslice = np.min((Npsf,im1.shape[0]))            
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
        positions, backgrounds, intensities, pupilR,pupilI,sigma, stagepos = variables
        positions = positions*self.weight[2]
        pupil_mag = tf.complex(pupilR*self.weight[4],0.0)
        pupil = tf.complex(tf.math.cos(pupilI*self.weight[3]),tf.math.sin(pupilI*self.weight[3]))*pupil_mag*self.aperture*self.apoid
        pupil_real = [pupilR*self.weight[4],pupilI*self.weight[3]]
        stagepos = stagepos*self.weight[5]
        I_model = self.genpsfmodel(sigma,stagepos=stagepos,pupil=pupil)
        # calculate global positions in images since positions variable just represents the positions in the rois
        images, _, centers, _ = self.data.get_image_data()
        original_shape = images.shape[-3:]
        
        global_positions = np.swapaxes(np.vstack((positions[:,0],centers[:,-2]-positions[:,-2],centers[:,-1]-positions[:,-1])),1,0)

        return [global_positions.astype(np.float32),
                backgrounds*self.weight[1], # already correct
                intensities*self.weight[0], # already correct
                I_model,
                np.complex64(pupil),
                pupil_real,
                sigma,
                stagepos*self.data.pixelsize_z,
                variables] # already correct
    

    def res2dict(self,res):
        res_dict = dict(pos=res[0],
                        bg=np.squeeze(res[1]),
                        intensity=np.squeeze(res[2]),
                        I_model = res[3],
                        pupil = res[4],
                        pupil_real = res[5],
                        sigma = res[6]/np.pi,
                        stagepos = res[7],
                        offset=np.min(res[3]),
                        zernike_polynomial = self.Zk,
                        apodization = self.apoid,
                        zoffset = self.Zoffset,
                        cor_all = self.data.alldata['centers'],
                        cor = self.data.centers)

        return res_dict