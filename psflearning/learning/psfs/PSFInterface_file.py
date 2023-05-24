from abc import ABCMeta, abstractmethod
import pickle

import numpy as np
import tensorflow as tf
import scipy.special as spf

from ..data_representation.PreprocessedImageDataInterface_file import PreprocessedImageDataInterface
from .. import utilities as im
from .. import imagetools as nip

class PSFInterface():
    """
    Interface that ensures consistency and compatability between all old and new implementations of data classes, fitters and psfs.
    Classes implementing this interafce define a psf model/parametrization. They describe how the parameters of the psf are used to calculate a forward image
    at a specific position. They also provide initial values and postprocessing of the variables for the fitter,
    since they depend on the nature of the psf model/parametrization.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def calc_initials(self, data: PreprocessedImageDataInterface) -> list:
        """
        Calculates the initial values for the optimizable variables.
        """
        raise NotImplementedError("You need to implement a 'calc_initials' method in your psf class.")

    @abstractmethod
    def calc_forward_images(self, variables: list) -> tf.Tensor:
        """
        Calculates the forward images.
        """
        raise NotImplementedError("You need to implement a 'calc_forward_images' method in your psf class.")

    @abstractmethod
    def postprocess(self, variables: list) -> list:
        """
        Postprocesses the optimized variables. For example, normalizes the psf or calculates global positions.
        """
        raise NotImplementedError("You need to implement a 'postprocess' method in your psf class.")

    def save(self, filename: str) -> None:
        """
        Save object to file.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(filename: str):
        """
        Load object from file.
        """
        with open(filename, "rb") as f:
            self = pickle.load(f)
        return self

    def calpupilfield(self,fieldtype='vector',Nz=None,datatype='bead'):
        if Nz is None:
            Nz = self.bead_kernel.shape[0]
        bin = self.options.model.bin
        Lx = self.data.rois.shape[-1]*bin        
        Ly = self.data.rois.shape[-2]*bin
        Lz = self.data.rois.shape[-3]
        xsz =self.options.model.pupilsize
     
        xrange = np.linspace(-Lx/2+0.5,Lx/2-0.5,Lx)
        [xx,yy] = np.meshgrid(xrange,xrange)
        pkx = xx/Lx
        pky = yy/Lx     
        self.kspace = np.float32(pkx*pkx+pky*pky)
        self.kspace_x = np.float32(pkx*pkx)
        self.kspace_y = np.float32(pky*pky)

        pixelsize_x = self.data.pixelsize_x/bin
        pixelsize_y = self.data.pixelsize_y/bin
        NA = self.options.imaging.NA
        emission_wavelength = self.options.imaging.emission_wavelength
        nimm = self.options.imaging.RI.imm
        nmed = self.options.imaging.RI.med
        ncov = self.options.imaging.RI.cov
        n_max = self.options.model.n_max
        Zk = im.genZern1(n_max,xsz)

        n1 = np.array(range(-1,n_max,2))
        self.spherical_terms = (n1+1)*(n1+2)//2

        pupilradius = 1
        krange = np.linspace(-pupilradius+pupilradius/xsz,pupilradius-pupilradius/xsz,xsz)
        [xx,yy] = np.meshgrid(krange,krange)
        kr = np.lib.scimath.sqrt(xx**2+yy**2)
        kz = np.lib.scimath.sqrt((nimm/emission_wavelength)**2-(kr*NA/emission_wavelength)**2)

        cos_imm = np.lib.scimath.sqrt(1-(kr*NA/nimm)**2)
        cos_med = np.lib.scimath.sqrt(1-(kr*NA/nmed)**2)
        cos_cov = np.lib.scimath.sqrt(1-(kr*NA/ncov)**2)
        kz_med = nmed/emission_wavelength*cos_med
        FresnelPmedcov = 2*nmed*cos_med/(nmed*cos_cov+ncov*cos_med)
        FresnelSmedcov = 2*nmed*cos_med/(nmed*cos_med+nmed*cos_med)
        FresnelPcovimm = 2*ncov*cos_cov/(ncov*cos_imm+nimm*cos_cov)
        FresnelScovimm = 2*ncov*cos_cov/(ncov*cos_cov+nimm*cos_imm)
        Tp = FresnelPmedcov*FresnelPcovimm
        Ts = FresnelSmedcov*FresnelScovimm

        phi = np.arctan2(yy,xx)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        sin_med = kr*NA/nmed

        pvec = Tp*np.stack([cos_med*cos_phi,cos_med*sin_phi,-sin_med])
        svec = Ts*np.stack([-sin_phi,cos_phi,np.zeros(cos_phi.shape)])

        hx = cos_phi*pvec-sin_phi*svec
        hy = sin_phi*pvec+cos_phi*svec
        h = np.concatenate((hx,hy),axis=0)
        if self.options.model.with_apoid:
            apoid = 1/np.lib.scimath.sqrt(cos_med)
        else:
            apoid = 1/np.lib.scimath.sqrt(1.0)

        imszx = Lx*pixelsize_x/2.0*NA/emission_wavelength
        imszy = Lx*pixelsize_y/2.0*NA/emission_wavelength

        #self.paramx = im.prechirpz(pupilradius,imszx,xsz,Lx)
        #self.paramy = im.prechirpz(pupilradius,imszy,xsz,Lx)
        kpixelsize = 2.0*NA/emission_wavelength/xsz
        self.paramxy = im.prechirpz1(kpixelsize,pixelsize_x,pixelsize_y,xsz,Lx)

        self.aperture = np.complex64(kr<1)
        pupil = self.aperture*apoid
        if fieldtype=='scalar':
            #self.normf = np.complex64(((pixelsize_x*NA/emission_wavelength)**2)/3.0)
            self.normf = np.complex64(pixelsize_x*pixelsize_y/np.sum(pupil*tf.math.conj(pupil)*kpixelsize*kpixelsize))
        elif fieldtype=='vector':
            #self.normf = np.complex64(((pixelsize_x*NA/emission_wavelength)**2)/6.0)
            self.normf = np.complex64(pixelsize_x*pixelsize_y/np.sum(pupil*tf.math.conj(pupil)*kpixelsize*kpixelsize))

        #if datatype == 'bead':
        #    self.Zrange = -1*np.linspace(-Nz/2+0.5,Nz/2-0.5,Nz,dtype=np.complex64).reshape((Nz,1,1))
        #elif datatype == 'insitu':
        self.Zrange = np.linspace(-Nz/2+0.5,Nz/2-0.5,Nz,dtype=np.complex64).reshape((Nz,1,1))
        self.kx = np.complex64(xx*NA/emission_wavelength)*pixelsize_x
        self.ky = np.complex64(yy*NA/emission_wavelength)*pixelsize_y
        self.kz = np.complex64(kz)*self.data.pixelsize_z
        self.kz_med = np.complex64(kz_med)*self.data.pixelsize_z
        self.k = np.complex64(nimm/emission_wavelength)*self.data.pixelsize_z
        self.dipole_field = np.complex64(h)
        #self.dipole_field = np.complex64([1.0])
        self.apoid = np.complex64(apoid)
        self.nimm = nimm
        self.nmed = nmed
        self.Zk = np.float32(Zk)

        # only for bead data, precompute phase ramp
        Lx = self.data.rois.shape[-1]      
        Ly = self.data.rois.shape[-2]
        Lz = self.data.rois.shape[-3]

        self.zv = np.linspace(0,Lz-1,Lz,dtype=np.float32).reshape(Lz,1,1)-Lz/2
        self.kxv = np.linspace(-Lx/2+0.5,Lx/2-0.5,Lx,dtype=np.float32)/Lx
        self.kyv = (np.linspace(-Ly/2+0.5,Ly/2-0.5,Ly,dtype=np.float32).reshape(Ly,1))/Ly
        self.kzv = (np.linspace(-Lz/2+0.5,Lz/2-0.5,Lz,dtype=np.float32).reshape(Lz,1,1))/Lz


    def gen_bead_kernel(self,isVolume = False):
        pixelsize_z = self.data.pixelsize_z
        bead_radius = self.data.bead_radius
        if isVolume:
            Nz = self.data.rois.shape[-3]
            bin = 1
        else:
            Nz = self.data.rois.shape[-3]+np.int32(bead_radius//pixelsize_z)*2+4
            bin = self.options.model.bin
        
        Lx = self.data.rois.shape[-1]*bin
        pixelsize_x = self.data.pixelsize_x/bin
        pixelsize_y = self.data.pixelsize_y/bin

        xrange = np.linspace(-Lx/2+0.5,Lx/2-0.5,Lx)+1e-6
        zrange = np.linspace(-Nz/2+0.5,Nz/2-0.5,Nz)
        [xx,yy,zz] = np.meshgrid(xrange,xrange,zrange)
        xx = np.swapaxes(xx,0,2)
        yy = np.swapaxes(yy,0,2)
        zz = np.swapaxes(zz,0,2)

        pkx = 1/Lx/pixelsize_x
        pky = 1/Lx/pixelsize_y
        pkz = 1/Nz/pixelsize_z
        if bead_radius>0:
            Zk0 = np.sqrt((xx*pkx)**2+(yy*pky)**2+(zz*pkz)**2)*bead_radius
            mu = 1.5
            kernel = spf.jv(mu,2*np.pi*Zk0)/(Zk0**mu)*bead_radius**3
            kernel = kernel/np.max(kernel)
            kernel = np.float32(kernel)
        else:
            kernel = np.ones((Nz,Lx,Lx),dtype=np.float32)
        self.bead_kernel = tf.complex(kernel,0.0)

        return 


    def applyPhaseRamp(self, img, shiftvec):
        """
        Applies a frequency ramp as a phase factor according to the shiftvec to a Fourier transform to shift the image.
		Identical to implementation in InverseModelling. Just removed if-statement (0) that does not make sense for me and prevent my code to work correctly.
		img: input Fourier transform tensor
		shiftvec: real-space shifts
		"""
        # TODO: no im
        res = im.totensor(img)
        myshape = im.shapevec(res)
        ShiftDims = shiftvec.shape[-1]
        for d in range(1, ShiftDims+1):
            myshifts = shiftvec[..., -d] 
            for ed in range(len(myshape) - len(myshifts.shape)): 
                myshifts = tf.expand_dims(myshifts,-1)
            res = res * tf.exp(tf.complex(im.totensor(0.0), 2.0 * np.pi * myshifts * nip.ramp1D(myshape[-d], ramp_dim = -d, freq='ftfreq')))
        return res

    def phaseRamp(self,pos):
        if pos.shape[1]==2:
            shiftphase = 1j*2*np.pi*(self.kxv*pos[:,1]+self.kyv*pos[:,0])
        if pos.shape[1]==3:
            shiftphase = 1j*2*np.pi*(self.kxv*pos[:,2]+self.kyv*pos[:,1]+self.kzv*pos[:,0])

        return tf.exp(shiftphase)

    def applyDrfit(self,psfin,gxy):
        otf2d = im.fft2d(tf.complex(psfin,0.0))
        if self.data.skew_const:
            sk = np.array([self.data.skew_const],dtype=np.float32)+np.zeros(gxy.shape,dtype=np.float32)
            sk = np.reshape(sk,sk.shape+(1,1,1))
            dxy = tf.complex(-sk*self.zv+tf.round(sk*self.zv),0.0) 
            shiftphase = self.phaseRamp(dxy)

        else:
            gxy = tf.complex(tf.reshape(gxy,gxy.shape+(1,1,1)),0.0)*self.zv
            shiftphase = self.phaseRamp(gxy)
        psf_shift = tf.math.real(im.ifft2d(otf2d*shiftphase))

        return psf_shift

    def psf2IAB(self, ROIs):
        G = np.zeros(ROIs.shape, dtype = np.complex64)
        G[:,0] = ROIs[:,0]*np.exp(-2*np.pi/3*1j)+ROIs[:,1]+ROIs[:,2]*np.exp(2*np.pi/3*1j)
        G[:,1] = np.sum(ROIs,axis=1)
        G[:,2] = ROIs[:,0]*np.exp(2*np.pi/3*1j)+ROIs[:,1]+ROIs[:,2]*np.exp(-2*np.pi/3*1j) # G[:,2] = np.conj(G[:,0])
        # solving above equations for ROIs and redefine it as O
        O = np.zeros(ROIs.shape, dtype = np.complex64)
        O[:,0] = 1/3*(G[:,0]*np.exp(2*np.pi/3*1j)+G[:,1]+G[:,2]*np.exp(-2*np.pi/3*1j))
        O[:,1] = 1/3*np.sum(G,axis=1)
        O[:,2] = 1/3*(G[:,0]*np.exp(-2*np.pi/3*1j)+G[:,1]+G[:,2]*np.exp(2*np.pi/3*1j)) # O[:,2] = np.conj(O[:,0])
        # above derivation is purely based on the definition of FFT and the fact that cos(2pi/3) and cos(4pi/3) are all equal to -0.5.
        # it is true for PSF at any 3 phases, however, if the 3 phases are exactly at [-2pi/3, 0, 2pi/3], then G can be used to represent the complex IAB model, where
        I = np.real(G[:,1])/3
        A = G[:,0]/3
        B = G[:,2]/3 # B = np.conj(A) 

        a = np.squeeze(np.sum(np.real(A[0]),axis = (-1,-2)))
        b = np.squeeze(np.sum(np.imag(A[0]),axis = (-1,-2)))

        y1 = np.squeeze(np.sum((ROIs[:,2]-ROIs[:,0])/np.sqrt(3),axis = (-1,-2)))
        y2 = np.squeeze(np.sum(ROIs[:,1]-np.sum(ROIs,axis = 1)/3,axis = (-1,-2)))

        q = np.squeeze(1j*(a*y1-b*y2) + (a*y2+b*y1))
        if len(q.shape)>1:
            phi = -1*np.median(np.angle(q),axis=1)
        else:
            phi = -1*np.median(np.angle(q))


        return I, A, B, phi

