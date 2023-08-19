"""
Copyright (c) 2022      Ries Lab, EMBL, Heidelberg, Germany
All rights reserved     

@author: Sheng Liu, Jonas Hellgoth
"""

import numpy as np
import scipy as sp
import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter
from .PSFInterface_file import PSFInterface
from ..data_representation.PreprocessedImageDataInterface_file import PreprocessedImageDataInterface
from ..loss_functions import mse_real_pupil
from .. import utilities as im
from .. import imagetools as nip

class PSFPupilBased(PSFInterface):
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
        self.initpupil = None
        self.defocus = np.float32(0)
        self.default_loss_func = mse_real_pupil
        self.psftype = 'scalar'
        return

    def calc_initials(self, data: PreprocessedImageDataInterface, start_time=None):
        """
        Provides initial values for the optimizable varibales for the fitter class.
        """
        self.data = data
        _, rois, _, _ = self.data.get_image_data()

        options = self.options
        if options.model.with_IMM:
            init_positions = np.zeros((rois.shape[0], len(rois.shape)))
        else:
            init_positions = np.zeros((rois.shape[0], len(rois.shape)-1))

        init_backgrounds = np.array(np.min(gaussian_filter(rois, [0, 2, 2, 2]), axis=(-3, -2, -1), keepdims=True))
        init_intensitiesL = np.sum(rois - init_backgrounds, axis=(-2, -1), keepdims=True)
        init_intensities = np.mean(init_intensitiesL,axis=1,keepdims=True)
        
        self.gen_bead_kernel()
        N = rois.shape[0]
        Nz = self.bead_kernel.shape[0]
        Lx = rois.shape[-1]
        xsz =options.model.pupilsize

        if self.psftype == 'vector':
            self.calpupilfield('vector')
        else:
            self.calpupilfield('scalar')
        #self.sincfilter = np.sinc(np.sqrt(self.kspace_x))*np.sinc(np.sqrt(self.kspace_y))
        self.const_mag = options.model.const_pupilmag
        #self.bead_kernel = tf.complex(self.data.bead_kernel,0.0)
        #self.weight = np.array([np.median(init_intensities), 10, 0.1, 10, 10],dtype=np.float32)
        #weight = [1e4,10] + list(np.array([0.1,5,2])/np.median(init_intensities)*2e4)
        wI = np.lib.scimath.sqrt(np.median(init_intensities))
        weight = [wI*40,20] + list(np.array([1,30,30])/wI*40)
        self.weight = np.array(weight,dtype=np.float32)
        sigma = np.ones((2,))*self.options.model.blur_sigma*np.pi
        self.init_sigma = sigma

        init_pupil = np.zeros((xsz,xsz))+(1+0.0*1j)/self.weight[4]
        init_backgrounds[init_backgrounds<0.1] = 0.1
        init_backgrounds = np.ones((N,1,1,1),dtype = np.float32)*np.median(init_backgrounds,axis=0, keepdims=True) / self.weight[1]
        gxy = np.zeros((N,2),dtype=np.float32) 
        gI = np.ones((N,Nz,1,1),dtype = np.float32)*init_intensities
        self.varinfo = [dict(type='Nfit',id=0),
            dict(type='Nfit',id=0),
            dict(type='Nfit',id=0),
            dict(type='shared'),
            dict(type='shared'),
            dict(type='shared'),
            dict(type='Nfit',id=0)]
       
        if options.model.var_photon:
            init_Intensity = gI/self.weight[0]
        else:
            init_Intensity = init_intensities / self.weight[0]
        return [init_positions.astype(np.float32),
                init_backgrounds.astype(np.float32),
                init_Intensity.astype(np.float32),
                np.real(init_pupil).astype(np.float32),
                np.imag(init_pupil).astype(np.float32),
                sigma.astype(np.float32),
                gxy], start_time
        
    def calc_forward_images(self, variables):
        """
        Calculate forward images from the current guess of the variables.
        Shifting is done by Fourier transform and applying a phase ramp.
        """

        pos, backgrounds, intensities, pupilR, pupilI, sigma,gxy = variables

        if self.const_mag:
            pupil_mag = tf.complex(1.0,0.0)
        else:
            pupil_mag = tf.complex(pupilR*self.weight[4],0.0)
        #pupil = tf.complex(tf.math.cos(pupilI*self.weight[3]),tf.math.sin(pupilI*self.weight[3]))*pupil_mag*self.aperture*self.apoid

        if self.initpupil is not None:
            pupil = self.initpupil
        else:
            pupil = tf.complex(tf.math.cos(pupilI*self.weight[3]),tf.math.sin(pupilI*self.weight[3]))*pupil_mag*self.aperture*self.apoid

        Nz = self.Zrange.shape[0]
        pos = tf.complex(tf.reshape(pos,pos.shape+(1,1,1)),0.0)
        phiz = -1j*2*np.pi*self.kz*(pos[:,0]+self.Zrange+self.defocus)
        if pos.shape[1]>3:
            phixy = 1j*2*np.pi*self.ky*pos[:,2]+1j*2*np.pi*self.kx*pos[:,3]
            phiz = 1j*2*np.pi*(self.kz_med*pos[:,1]-self.kz*(pos[:,0]+self.Zrange))
        else:
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

        bin = self.options.model.bin
        if not self.options.model.var_blur:
            sigma = self.init_sigma
        filter2 = tf.exp(-2*sigma[1]*sigma[1]*self.kspace_x-2*sigma[0]*sigma[0]*self.kspace_y)
        filter2 = tf.complex(filter2/tf.reduce_max(filter2),0.0)
        I_blur = im.ifft3d(im.fft3d(I_res)*self.bead_kernel*filter2)
        #I_blur = im.ifft3d(im.fft3d(I_res)*filter2)
        I_blur = tf.expand_dims(tf.math.real(I_blur),axis=-1)
        
        kernel = np.ones((1,bin,bin,1,1),dtype=np.float32)
        I_blur_bin = tf.nn.convolution(I_blur,kernel,strides=(1,1,bin,bin,1),padding='SAME',data_format='NDHWC')

        psf_fit = I_blur_bin[...,0]*intensities*self.weight[0]
        
        st = (self.bead_kernel.shape[0]-self.data.rois[0].shape[-3])//2
        psf_fit = psf_fit[:,st:Nz-st]

        if self.options.model.estimate_drift:
            gxy = gxy*self.weight[2]
            psf_shift = self.applyDrfit(psf_fit,gxy)
            forward_images = psf_shift + backgrounds*self.weight[1]
        else:
            forward_images = psf_fit + backgrounds*self.weight[1]

        return forward_images

    def genpsfmodel(self,sigma,pupil,addbead=False):
        phiz = -1j*2*np.pi*self.kz*(self.Zrange+self.defocus)
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
    
        bin = self.options.model.bin

        filter2 = tf.exp(-2*sigma[1]*sigma[1]*self.kspace_x-2*sigma[0]*sigma[0]*self.kspace_y)
        filter2 = tf.complex(filter2/tf.reduce_max(filter2),0.0)
        if addbead:
            I_blur = np.real(im.ifft3d(im.fft3d(I_res)*filter2*self.bead_kernel))
        else:
            I_blur = np.real(im.ifft3d(im.fft3d(I_res)*filter2))
        I_blur = tf.expand_dims(tf.math.real(I_blur),axis=-1)
        
        kernel = np.ones((bin,bin,1,1),dtype=np.float32)
        I_model = tf.nn.convolution(I_blur,kernel,strides=(1,bin,bin,1),padding='SAME',data_format='NHWC')
        I_model = I_model[...,0]

        return I_model

    def postprocess(self, variables):
        """
        Applies postprocessing to the optimized variables. In this case calculates
        real positions in the image from the positions in the roi. Also, normalizes
        psf and adapts intensities and background accordingly.
        """
        positions, backgrounds, intensities, pupilR,pupilI,sigma,gxy = variables
        z_center = (self.Zrange.shape[-3] - 1) // 2
        bin = self.options.model.bin
        positions[:,1:] = positions[:,1:]/bin

        pupil_mag = tf.complex(pupilR*self.weight[4],0.0)
        if self.initpupil is not None:
            pupil = self.initpupil
        else:
            pupil = tf.complex(tf.math.cos(pupilI*self.weight[3]),tf.math.sin(pupilI*self.weight[3]))*pupil_mag*self.aperture*self.apoid

        I_model = self.genpsfmodel(sigma,pupil)
        I_model_bead = self.genpsfmodel(sigma,pupil,addbead=True)
        #I_model_bead = np.real(im.ifft3d(im.fft3d(I_res)*self.bead_kernel*filter2))

        images, _, centers, _ = self.data.get_image_data()
        if positions.shape[1]>3:
            global_positions = np.swapaxes(np.vstack((positions[:,0]+z_center,positions[:,1],centers[:,-2]-positions[:,-2],centers[:,-1]-positions[:,-1])),1,0)
        else:
            global_positions = np.swapaxes(np.vstack((positions[:,0]+z_center,centers[:,-2]-positions[:,-2],centers[:,-1]-positions[:,-1])),1,0)

        return [global_positions.astype(np.float32),
                backgrounds*self.weight[1], # already correct
                intensities*self.weight[0], # already correct
                I_model_bead,
                I_model,
                np.complex64(pupil),
                sigma,
                gxy*self.weight[2],
                np.flip(I_model,axis=-3),
                variables] # already correct


    def res2dict(self,res):
        res_dict = dict(pos=res[0],
                        bg=np.squeeze(res[1]),
                        intensity=np.squeeze(res[2]),
                        I_model_bead = res[3],
                        I_model = res[4],
                        pupil = res[5],
                        sigma = res[6]/np.pi,
                        drift_rate=res[7],
                        I_model_reverse = res[8],
                        offset=np.min(res[4]),
                        apodization = self.apoid,
                        cor_all = self.data.centers_all,
                        cor = self.data.centers)    
        return res_dict