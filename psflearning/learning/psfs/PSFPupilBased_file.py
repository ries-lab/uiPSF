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
    def __init__(self, estdrift=False, varphoton=False,options=None) -> None:
        self.parameters = None
        self.data = None
        self.estdrift = estdrift
        self.varphoton = varphoton
        self.Zphase = None
        self.zT = None
        self.bead_kernel = None
        self.options = options
        self.initpupil = None
        self.default_loss_func = mse_real_pupil
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
        xsz =options.model.pupilsize
        
        self.calpupilfield('scalar')
        self.const_mag = options.model.const_pupilmag
        self.bead_kernel = tf.complex(self.data.bead_kernel,0.0)
        self.weight = np.array([np.median(init_intensities), 10, 0.1, 10, 10],dtype=np.float32)
        sigma = np.ones((2,))*self.options.model.blur_sigma*np.pi

        init_pupil = np.zeros((xsz,xsz))+(1+0.0*1j)/self.weight[4]
        init_backgrounds[init_backgrounds<0.1] = 0.1
        init_backgrounds = np.ones((N,1,1,1),dtype = np.float32)*np.median(init_backgrounds,axis=0, keepdims=True) / self.weight[1]
        gxy = np.zeros((N,2),dtype=np.float32) 
        st = (self.bead_kernel.shape[0]-self.data.rois[0].shape[-3])//2
        gI = np.ones((N,Nz,1,1),dtype = np.float32)*init_intensities
        #gI[:,st:Nz-st] = init_intensitiesL
        #gI[:,0:st] = np.abs(np.min(init_intensitiesL[:,0]))
        #gI[:,-st:] = np.abs(np.min(init_intensitiesL[:,-1]))
        
        if self.varphoton:
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
        pupil = tf.complex(tf.math.cos(pupilI*self.weight[3]),tf.math.sin(pupilI*self.weight[3]))*pupil_mag*self.aperture*self.apoid

        if self.initpupil is not None:
            pupil = self.initpupil
        else:
            pupil = tf.complex(tf.math.cos(pupilI*self.weight[3]),tf.math.sin(pupilI*self.weight[3]))*pupil_mag*self.aperture*self.apoid


        Nz = self.Zrange.shape[0]

        pos = tf.complex(tf.reshape(pos,pos.shape+(1,1,1)),0.0)
                
        phiz = 1j*2*np.pi*self.kz*(pos[:,0]+self.Zrange)
        if pos.shape[1]>3:
            phixy = 1j*2*np.pi*self.ky*pos[:,2]+1j*2*np.pi*self.kx*pos[:,3]
            phiz = 1j*2*np.pi*(self.kz_med*pos[:,1]-self.kz*(pos[:,0]-self.Zrange))
        else:
            phixy = 1j*2*np.pi*self.ky*pos[:,1]+1j*2*np.pi*self.kx*pos[:,2]
            #IMMphase = 0.0

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
        
        st = (self.bead_kernel.shape[0]-self.data.rois[0].shape[-3])//2
        psf_fit = psf_fit[:,st:Nz-st]

        if self.estdrift:
            
            Nz = psf_fit.shape[-3]
            zv = np.expand_dims(np.linspace(0,Nz-1,Nz,dtype=np.float32)-Nz/2,axis=-1)
            if self.data.skew_const:
                sk = np.array([self.data.skew_const],dtype=np.float32)
                #gxy = gxy*0.0 + sk
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
        positions, backgrounds, intensities, pupilR,pupilI,sigma,gxy = variables
        z_center = (self.Zrange.shape[-3] - 1) // 2
        #z_center = 0
        #pupil =  tf.complex(pupilR,pupilI)*self.weight[3]
        #pupil_mag = self.aperture/np.sqrt(np.sum(self.aperture))
        pupil_mag = tf.complex(pupilR*self.weight[4],0.0)
        if self.initpupil is not None:
            pupil = self.initpupil
        else:
            pupil = tf.complex(tf.math.cos(pupilI*self.weight[3]),tf.math.sin(pupilI*self.weight[3]))*pupil_mag*self.aperture*self.apoid

        phiz = 1j*2*np.pi*self.kz*self.Zrange
        PupilFunction = pupil*tf.exp(phiz)

        IntermediateImage = tf.transpose(im.cztfunc(PupilFunction,self.paramx),perm=(0,2,1))
        I_res = tf.transpose(im.cztfunc(IntermediateImage,self.paramy),perm=(0,2,1))        
        I_res = I_res*tf.math.conj(I_res)*self.normf
    
        I_res = np.real(I_res)
        #filter2 = tf.exp(-2*sigma*sigma*self.kspace)
        filter2 = tf.exp(-2*sigma[1]*sigma[1]*self.kspace_x-2*sigma[0]*sigma[0]*self.kspace_y)

        filter2 = filter2/tf.reduce_max(filter2)
        filter2 = tf.complex(filter2,0.0)
        I_model = np.real(im.ift3d(im.ft3d(I_res)*filter2))
        I_model_bead = np.real(im.ift3d(im.ft3d(I_res)*self.bead_kernel*filter2))

        # calculate global positions in images since positions variable just represents the positions in the rois
        images, _, centers, _ = self.data.get_image_data()
        original_shape = images.shape[-3:]
        Nbead = centers.shape[0]
        if positions.shape[1]>3:
            global_positions = np.swapaxes(np.vstack((positions[:,0]+z_center,positions[:,1],centers[:,-2]-positions[:,-2],centers[:,-1]-positions[:,-1])),1,0)
            #centers_with_z = np.concatenate((np.full((Nbead, 1), z_center),np.zeros((Nbead,1)), centers[:,-2:]), axis=1)
        else:
            global_positions = np.swapaxes(np.vstack((positions[:,0]+z_center,centers[:,-2]-positions[:,-2],centers[:,-1]-positions[:,-1])),1,0)
            #centers_with_z = np.concatenate((np.full((Nbead, 1), z_center), centers[:,-2:]), axis=1)

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
                sigma,
                gxy*self.weight[2],
                variables] # already correct


    def res2dict(self,res):
        res_dict = dict(pos=res[0],
                        bg=np.squeeze(res[1]),
                        intensity=np.squeeze(res[2]),
                        I_model = res[3],
                        pupil = res[4],
                        sigma = res[5]/np.pi,
                        drift_rate=res[6],
                        offset=np.min(res[3]),
                        apodization = self.apoid,
                        cor_all = self.data.centers_all,
                        cor = self.data.centers)    
        return res_dict