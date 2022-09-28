from abc import ABCMeta, abstractmethod
import pickle

import numpy as np
import tensorflow as tf

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

    def calpupilfield(self,fieldtype,Nz=None):
        if Nz is None:
            Nz = self.data.bead_kernel.shape[0]
        Lx = self.data.rois.shape[-1]        
        xsz =self.options['pupilsize']
     
        xrange = np.linspace(-Lx/2+0.5,Lx/2-0.5,Lx)
        [xx,yy] = np.meshgrid(xrange,xrange)
        pkx = xx/Lx
        pky = yy/Lx     
        self.kspace = np.float32(pkx*pkx+pky*pky)
        self.kspace_x = np.float32(pkx*pkx)
        self.kspace_y = np.float32(pky*pky)

        pixelsize_x = self.data.pixelsize_x
        pixelsize_y = self.data.pixelsize_y
        NA = self.options['NA']
        emission_wavelength = self.options['emission_wavelength']
        nimm = self.options['RI_imm']
        nmed = self.options['RI_med']
        ncov = self.options['RI_cov']
        n_max = self.options['n_max']
        out = im.genZern(n_max,xsz,NA,emission_wavelength,nimm,pixelsize_x,applymask=False)
        Zk = out[0]        
        signm = out[-1]%2*2-1
        signm[0] = 0
        self.signm = np.reshape(signm,(len(signm),1,1)).astype(np.float32)

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
        if self.options['with_apoid']:
            apoid = 1/np.lib.scimath.sqrt(cos_med)/1.225
        else:
            apoid = 1/np.lib.scimath.sqrt(1.0)

        imszx = Lx*pixelsize_x/2.0*NA/emission_wavelength
        imszy = Lx*pixelsize_y/2.0*NA/emission_wavelength

        self.paramx = im.prechirpz(pupilradius,imszx,xsz,Lx)
        self.paramy = im.prechirpz(pupilradius,imszy,xsz,Lx)

        self.aperture = np.complex64(kr<1)
        if fieldtype=='scalar':
            self.normf = np.complex64(((pixelsize_x*NA/emission_wavelength)**2)/3.0)
        elif fieldtype=='vector':
            self.normf = np.complex64(((pixelsize_x*NA/emission_wavelength)**2)/6.0)


        
        self.Zrange = -1*np.linspace(-Nz/2+0.5,Nz/2-0.5,Nz,dtype=np.complex64).reshape((Nz,1,1))
        self.kx = np.complex64(xx*NA/emission_wavelength)*pixelsize_x
        self.ky = np.complex64(yy*NA/emission_wavelength)*pixelsize_y
        self.kz = np.complex64(kz)*self.data.pixelsize_z
        self.kz_med = np.complex64(kz_med)*self.data.pixelsize_z
        self.dipole_field = np.complex64(h)
        #self.dipole_field = np.complex64([1.0])
        self.apoid = np.complex64(apoid)
        self.nimm = nimm
        self.nmed = nmed
        self.Zk = np.float32(Zk)

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

