
"""
Copyright (c) 2022      Ries Lab, EMBL, Heidelberg, Germany
All rights reserved     Heintzmann Lab, Friedrich-Schiller-University Jena, Germany

@author: Rainer Heintzmann, Sheng Liu, Jonas Hellgoth
"""

import tensorflow as tf
import numpy as np
import scipy as sp
from math import factorial 
from . import imagetools as nip
import numbers
from scipy import ndimage
import scipy.fft as fft
defaultTFDataType="float32"
defaultTFCpxDataType="complex64"
#%%
# The functions below are Tensorflow now

def ft(tfin, axes=None, useMulShift=False):
    """
    performs a centered Fourier transformation.

    FFTs perform a fully complex Fourier transformation.

    Parameters
    ----------
    tfin : tensorflow array
        The array to be transformed
    axes : list of axes to be transformed. None means all axes

    Returns
    -------
    tensorflow array
        The transformed array

    See also
    -------
    ft, ConvolveOTF

    Example
    -------

    """
    tfin = tocomplex(tfin)
    ndims = tfin.shape.ndims
    if axes is not None:
        tdims = len(axes)
    else:
        tdims = ndims
    with tf.name_scope('ft'):
        if tdims == 1:
            if axes is not None:
                axes = list(axes)[0]
                if nip.dimToPositive(axes, ndims) == ndims - 1:
                    if useMulShift:
                        return preFFTShiftMultiply(tf.signal.fft(postFFTShiftMultiply(tfin, maxdim=1)), maxdim=1) / np.sqrt(np.prod(shapevec(tfin)[-1]))
                    else:
                        return postFFTShift(tf.signal.fft(preFFTShift(tfin, maxdim=1)), maxdim=1) / np.sqrt(np.prod(shapevec(tfin)[-1]))
                else:
                    perm = getPerm(axes, len(tfin.shape))
                    tfin = tf.transpose(tfin, perm)
                    if useMulShift:
                        res = preFFTShiftMultiply(tf.signal.fft(postFFTShiftMultiply(tfin, maxdim=1)), maxdim=1) / np.sqrt(np.prod(shapevec(tfin)[-1]))
                    else:
                        res = postFFTShift(tf.signal.fft(preFFTShift(tfin, maxdim=1)), maxdim=1) / np.sqrt(np.prod(shapevec(tfin)[-1]))
                    res = tf.transpose(res, inverse_perm(perm))
                    return res #tf.transpose(res, [-1, axes])
            else:
                if useMulShift:
                    return preFFTShiftMultiply(tf.signal.fft(postFFTShiftMultiply(tfin, maxdim=1)), maxdim=1) / np.sqrt(np.prod(shapevec(tfin)))
                else:
                    return postFFTShift(tf.signal.fft(preFFTShift(tfin, maxdim=1)), maxdim=1) / np.sqrt(np.prod(shapevec(tfin)))
        elif tdims == 2:  # tfin.shape[-3]==1
            if axes is not None:
                perm = getPerm(axes, len(tfin.shape))
                tfin = tf.transpose(tfin, perm)
                if useMulShift:
                    res = preFFTShiftMultiply(tf.signal.fft2d(postFFTShiftMultiply(tfin, maxdim=2)), maxdim=2) / np.sqrt(np.prod(shapevec(tfin)[-2:]))
                else:
                    res = postFFTShift(tf.signal.fft2d(preFFTShift(tfin, maxdim=2)), maxdim=2) / np.sqrt(np.prod(shapevec(tfin)[-2:]))
                res = tf.transpose(res, inverse_perm(perm))
                return res
            else:
                if useMulShift:
                    return preFFTShiftMultiply(tf.signal.fft2d(postFFTShiftMultiply(tfin, maxdim=2)), maxdim=2) / np.sqrt(np.prod(shapevec(tfin)))
                else:
                    return postFFTShift(tf.signal.fft2d(preFFTShift(tfin, maxdim=2)), maxdim=2) / np.sqrt(np.prod(shapevec(tfin)))
        else:
#        elif tfin.shape.ndims == 3:
            if axes is not None:
                perm = getPerm(axes, len(tfin.shape))
                tfin = tf.transpose(tfin, perm)
                if useMulShift:
                    res = preFFTShiftMultiply(tf.signal.fft3d(postFFTShiftMultiply(tfin, maxdim=3)), maxdim=3) / np.sqrt(np.prod(shapevec(tfin))[-3:])
                else:
                    res = postFFTShift(tf.signal.fft3d(preFFTShift(tfin, maxdim=3)), maxdim=3) / np.sqrt(np.prod(shapevec(tfin)[-3:]))
                res = tf.transpose(res, inverse_perm(perm))
                return res
            else:
                if useMulShift:
                    return preFFTShiftMultiply(tf.signal.fft3d(postFFTShiftMultiply(tfin, maxdim=3)), maxdim=3) / np.sqrt(np.prod(shapevec(tfin)))
                else:
                    return postFFTShift(tf.signal.fft3d(preFFTShift(tfin, maxdim=3)), maxdim=3) / np.sqrt(np.prod(shapevec(tfin)))

def ft3d(tfin, useMulShift=False):
    return ft(tfin, axes=[-1,-2,-3], useMulShift=useMulShift)

def getPerm(axes, ndims):
    """
    returns the permutation to be used in "tf.transpose" when a series of axes is supplied
    :param axes: axes to "transform"
    :param ndims: total number of dimensions
    :return: permutation vector
    """
    if len(axes) > 3:
        raise ValueError("Tensorflow can only FFT up to 3D arrays.")
    perm = np.arange(ndims)
    for d in range(len(axes)):
        if axes[d] < 0:
            axes[d] = ndims + axes[d]
    axes = np.sort(np.array(axes))

    for d in range(len(axes)):
        tmp = perm[ndims-d-1]
        perm[ndims-d-1] = perm[axes[d]] # axes[d]
        perm[axes[d]] = tmp
    return perm

def ift(tfin, axes = None, useMulShift=False):
    """
    performs a centered inverse Fourier transformation.

    FFTs perform a fully complex Fourier transformation .

    Parameters
    ----------
    tfin : tensorflow array
        The array to be transformed

    Returns
    -------
    tensorflow array
        The transformed array

    See also
    -------
    ift, ConvolveOTF, PSF2ROTF

    Example
    -------

    """
    tfin = tocomplex(tfin)
    ndims = tfin.shape.ndims
    if axes is not None:
        tdims = len(axes)
    else:
        tdims = ndims
    with tf.name_scope('ift'):
        if tdims == 1:
            if axes is not None:
                axes = list(axes)[0]
                if nip.dimToPositive(axes, ndims) == ndims - 1:
                    if useMulShift:
                        return preFFTShiftMultiply(tf.signal.ifft(postFFTShiftMultiply(tfin, maxdim=1)), maxdim=1) * np.sqrt(np.prod(shapevec(tfin)[-1]))
                    else:
                        return postFFTShift(tf.signal.ifft(preFFTShift(tfin, maxdim=1)), maxdim=1) * np.sqrt(np.prod(shapevec(tfin)[-1]))
                else:
                    perm = getPerm(axes, len(tfin.shape))
                    tfin = tf.transpose(tfin, perm)
                    # tfin = tf.transpose(tfin, [-1, axes])
                    if useMulShift:
                        res = preFFTShiftMultiply(tf.signal.ifft(postFFTShiftMultiply(tfin, maxdim=1)), maxdim=1) * np.sqrt(np.prod(shapevec(tfin)[-1]))
                    else:
                        res = postFFTShift(tf.signal.ifft(preFFTShift(tfin, maxdim=1)), maxdim=1) * np.sqrt(np.prod(shapevec(tfin)[-1]))
                    res = tf.transpose(res, inverse_perm(perm))
                    return res # tf.transpose(res, [-1, axes])
            else:
                if useMulShift:
                    return preFFTShiftMultiply(tf.signal.ifft(postFFTShiftMultiply(tfin, maxdim=1)), maxdim=1) * np.sqrt(np.prod(shapevec(tfin)))
                else:
                    return postFFTShift(tf.signal.ifft(preFFTShift(tfin, maxdim=1)), maxdim=1) * np.sqrt(np.prod(shapevec(tfin)))
        elif tdims == 2 or tfin.shape[-3]==1:
            if axes is not None:
                perm = getPerm(axes, len(tfin.shape))
                tfin = tf.transpose(tfin, perm)
                # tfin = tf.transpose(tfin, [-1, axes[0]]);tfin = tf.transpose(tfin, [-2, axes[1]])
                if useMulShift:
                    res = preFFTShiftMultiply(tf.signal.ifft2d(postFFTShiftMultiply(tfin, maxdim=2)), maxdim=2) * np.sqrt(np.prod(shapevec(tfin))[-2:])
                else:
                    res = postFFTShift(tf.signal.ifft2d(preFFTShift(tfin, maxdim=2)), maxdim=2) * np.sqrt(np.prod(shapevec(tfin)[-2:]))
                res = tf.transpose(res, inverse_perm(perm))
                # res = tf.transpose(res, [-2, axes[1]]); res = tf.transpose(res, [-1, axes[0]])
                return res
            if useMulShift:
                return preFFTShiftMultiply(tf.signal.ifft2d(postFFTShiftMultiply(tfin, maxdim=2)), maxdim=2) * np.sqrt(np.prod(shapevec(tfin)[[-2, -1]]))
            else:
                return postFFTShift(tf.signal.ifft2d(preFFTShift(tfin, maxdim=2)), maxdim=2) * np.sqrt(np.prod(shapevec(tfin)[[-2, -1]]))
        else:
#        elif tfin.shape.ndims == 3:
            if axes is not None:
                perm = getPerm(axes, len(tfin.shape))
                tfin = tf.transpose(tfin, perm)
                if useMulShift:
                    res = preFFTShiftMultiply(tf.signal.ifft3d(postFFTShiftMultiply(tfin, maxdim=3)), maxdim=3) * np.sqrt(np.prod(shapevec(tfin))[-3:])
                else:
                    res = postFFTShift(tf.signal.ifft3d(preFFTShift(tfin, maxdim=3)), maxdim=3) * np.sqrt(np.prod(shapevec(tfin)[-3:]))
                res = tf.transpose(res, inverse_perm(perm))
                return res
            if useMulShift:
                return preFFTShiftMultiply(tf.signal.ifft3d(postFFTShiftMultiply(tfin, maxdim=3)), maxdim=3) * np.sqrt(np.prod(shapevec(tfin)[[-2, -1]]))
            else:
                return postFFTShift(tf.signal.ifft3d(preFFTShift(tfin, maxdim=3)), maxdim=3) * np.sqrt(np.prod(shapevec(tfin)[[-2, -1]]))

def ift3d(tfin, useMulShift=False):
    return ift(tfin, axes=[-1,-2,-3], useMulShift=useMulShift)

def preFFTShiftMultiply(tfin, maxdim=None, axes=None):
    """
    shifts the coordinate space before an FFT by Multiplying the result of the FFT with alternating factors:
    This has to be applied AFTER the FFT operation

    performs an fftshift operation, cyclicly wrapping around by a shift vector corresponding to the middle.
    The middle will end up at the zero coordinate pixel.

    Parameters
    ----------
    tfin : tensorflow array to be shifted

    Returns
    -------
    shifted tensorflow array
    """
    if maxdim is None:
        maxdim = tfin.shape.ndims
    if axes is None:
        axes = tuple(range(maxdim))  # tf.range(0, tf.size(tf.shape(tfin)))

    with tf.name_scope('preFFTShiftMultiply'):
        sz = shapevec(tfin)
        ndims = tfin.shape.ndims
        for ax in axes:
            pos = nip.FreqRamp1D(sz[ax], (sz[ax]//2), ax - ndims) # nip.castdim(np.mod(nip.ramp1D(sz[ax])+sz[ax],2),ndims,ax)*2-1.0 # + sz[ax] is a trick to have the correct alignment for uneven sizes
            tfin *= pos # makes a copy which is shifted
    return tfin

def preFFTShift(tfin, maxdim=None, axis=None):
    """
    shifts the coordinate space before an FFT.

    performs an fftshift operation, cyclicly wrapping around by a shift vector corresponding to the middle.
    The middle will end up at the zero coordinate pixel.

    Parameters
    ----------
    tfin : tensorflow array to be shifted

    Returns
    -------
    shifted tensorflow array

    """
    if axis is not None:
        shiftall = -mid(tfin)
        shiftby = nip.zeros(shapevec(tfin), dtype="int32")
        shiftby[axis] = shiftall[axis].astype("int32")
    else:
        shiftby = -mid(tfin)

    axes = tuple(range(tfin.shape.ndims)) # tf.range(0, tf.size(tf.shape(tfin)))
    if not maxdim is None:
        shiftby = shiftby[-maxdim:]
        axes = axes[-maxdim:]
    with tf.name_scope('preFFTShift'):
        return tf.roll(tfin, shift=shiftby, axis=axes)  # makes a copy which is shifted

def postFFTShiftMultiply(tfin, maxdim=None, axes=None):
    """
    shifts the coordinate space after an FFT by multiplying factors before the FFT.

    performs an fftshift operation, cyclicly wrapping around by a shift vector corresponding to the middle.
    The middle will end up at the zero coordinate pixel.

    Parameters
    ----------
    tfin : tensorflow array to be shifted

    Returns
    -------
    shifted tensorflow array

    """
    if maxdim is None:
        maxdim = tfin.shape.ndims
    if axes is None:
        axes = tuple(range(maxdim))  # tf.range(0, tf.size(tf.shape(tfin)))

    with tf.name_scope('postFFTShiftMultiply'):
        sz = shapevec(tfin)
        ndims = tfin.shape.ndims
        for ax in axes:
            pos = nip.FreqRamp1D(sz[ax],sz[ax]//2,ax - ndims, cornerFourier=True) # nip.castdim(np.mod(nip.ramp1D(sz[ax])+sz[ax],2),ndims,ax)*2-1.0 # + sz[ax] is a trick to have the correct alignment for uneven sizes
            tfin *= pos # makes a copy which is shifted
    return tfin

def postFFTShift(tfin, maxdim=None, axis=None):
    """
    shifts the coordinate space after an FFT.

    performs an fftshift operation, cyclicly wrapping around by a shift vector corresponding to the middle.
    The middle will end up at the zero coordinate pixel.

    Parameters
    ----------
    tfin : tensorflow array to be shifted

    Returns
    -------
    shifted tensorflow array

    """
    if axis is not None:
        shiftall = mid(tfin)
        shiftby = nip.zeros(shapevec(tfin), dtype="int32")
        shiftby[axis] = shiftall[axis].astype("int32")
    else:
        shiftby = mid(tfin)
    axes = tuple(range(tfin.shape.ndims)) # tf.range(0, tf.size(tf.shape(tfin)))
    if not maxdim is None:
        shiftby = shiftby[-maxdim:]
        axes = axes[-maxdim:]
    with tf.name_scope('postFFTShift'):
        return tf.roll(tfin, shift=shiftby, axis=axes)  # makes a copy which is shifted

def mid(tfin):
    """
    helper function to get the mid-point in integer coordinates of tensorflow arrays.

    It calculates the floor of the integer division of the shape vector. This is useful to get the zero coordinate in Fourier space

    Parameters
    ----------
    tfin : tensorflow array to be convolved with the PSF

    Returns
    -------
    vector to mid point

    """
#    return tf.floordiv(tf.shape(tfin),2)
    return shapevec(tfin)//2

def tocomplex(tfin):
    tfin=totensor(tfin)
    if iscomplex(tfin):
        return tfin
    else:
        return tf.complex(tfin, 0.0)  # tfin*0.0

def totensor(img):
    if istensor(img):
        return img
    if (not isinstance(0.0,numbers.Number)) and ((img.dtype==defaultTFDataType) or (img.dtype==defaultTFCpxDataType)):
        img=tf.constant(img)
    else:
        if iscomplex(img):
            img=tf.constant(img,defaultTFCpxDataType)
        else:
            img=tf.constant(img,defaultTFDataType)
    return img

def iscomplex(mytype):
    mytype=str(datatype(mytype))
    return (mytype == "complex64") or (mytype == "complex128") or (mytype == "complex64_ref") or (mytype == "complex128_ref") or (mytype=="<dtype: 'complex64'>") or (mytype=="<dtype: 'complex128'>")  

def shapevec(tfin):
    """
        returns the shape of a tensor as a numpy ndarray
    """
    if istensor(tfin):
        return np.array(tfin.shape.as_list())
    else:
        return np.array(tfin.shape)


def istensor(tfin):
    return isinstance(tfin,tf.Tensor) or isinstance(tfin,tf.Variable)

def datatype(tfin):
    if istensor(tfin):
        return tfin.dtype
    else:
        if isinstance(tfin,np.ndarray):
            return tfin.dtype.name
        return tfin # assuming this is already the type

def inverse_perm(perm):
    """
    inverts a permutation vector such that permuting dimensions will be reverted.
    :param perm: the permuation vector to invert
    :return: the invers permutation vector to apply via np.transpose()
    """
    myinv = np.empty_like(perm)
    myinv[perm] = np.arange(len(myinv), dtype=myinv.dtype)
    return myinv



def psf2cspline_np(psf):
    # calculate A
    A = np.zeros((64, 64))
    for i in range(1, 5):
        dx = (i-1)/3
        for j in range(1, 5):
            dy = (j-1)/3
            for k in range(1, 5):
                dz = (k-1)/3
                for l in range(1, 5):
                    for m in range(1, 5):
                        for n in range(1, 5):
                            A[(i-1)*16+(j-1)*4+k - 1, (l-1)*16+(m-1)*4+n - 1] = dx**(l-1) * dy**(m-1) * dz**(n-1)
    
    # upsample psf with factor of 3
    psf_up = ndimage.zoom(psf, 3.0, mode='grid-constant', grid_mode=True)[1:-1, 1:-1, 1:-1]
    A = np.float32(A)
    coeff = calsplinecoeff(A,psf,psf_up)
    return coeff


def calsplinecoeff(A,psf,psf_up):
    # calculate cspline coefficients
    coeff = np.zeros((64, psf.shape[0]-1, psf.shape[1]-1, psf.shape[2]-1))
    for i in range(coeff.shape[1]):
        for j in range(coeff.shape[2]):
            for k in range(coeff.shape[3]):
                temp = psf_up[i*3 : 3*(i+1)+1, j*3 : 3*(j+1)+1, k*3 : 3*(k+1)+1]
                #x = sp.linalg.solve(A, temp.reshape(64))
                x = sp.linalg.solve(A,temp.flatten())
                coeff[:, i, j, k] = x

    return coeff

def nl2noll(n,l):
    mm = abs(l)
    j = n * (n + 1) / 2 + 1 + max(0, mm - 1)
    if ((l > 0) & (np.mod(n, 4) >= 2)) | ((l < 0) & (np.mod(n, 4) <= 1)):
       j = j + 1
    
    return int(j)

def noll2nl(j):
    n = np.ceil((-3 + np.sqrt(1 + 8*j)) / 2)
    l = j - n * (n + 1) / 2 - 1
    if np.mod(n, 2) != np.mod(l, 2):
       l = l + 1
    
    if np.mod(j, 2) == 1:
       l= -l
    
    return int(n),int(l)

def genZern(n_max,xsz):
    coeff = np.zeros((n_max+1,n_max+1,n_max//2+1))
    for n in range(0,n_max+1):
        for m in range(n%2,n+1,2):
            if m==0:
                g = np.sqrt(n+1)
            else:
                g = np.sqrt(2*n+2)
            for k in range(0,(n-m)//2+1):
                coeff[n][m][k] = g*((-1)**k)*factorial(n-k)/factorial(k)/factorial((n+m)//2-k)/factorial((n-m)//2-k)
    
    pkx = 1/xsz
    xrange = np.linspace(-xsz/2+0.5,xsz/2-0.5,xsz)
    [xx,yy] = np.meshgrid(xrange,xrange)
    kr = np.lib.scimath.sqrt((xx*pkx)**2+(yy*pkx)**2)
    aperture = np.ones(kr.shape)
    rho = kr*2
    theta = np.arctan2(yy,xx)*aperture
   
    thetas = np.array(range(0,n_max+1)).reshape((n_max+1,1,1))*theta
    rhos = rho**np.array(range(0,n_max+1)).reshape((n_max+1,1,1))

    Nk = (n_max+1)*(n_max+2)//2
    
    Z = np.ones((Nk,xsz,xsz))
    l = 0
    Z[0] = aperture

    index_n = np.zeros((Nk,))
    index_m = np.zeros((Nk,))
    for n in range(1,n_max+1):
        for m in range(n%2,n+1,2):
            
            c = coeff[n][m][np.where(coeff[n][m]!=0)]
            c = c.reshape((len(c),1,1))
            p = np.sum(c*rhos[list(range(n,m-1,-2))],axis=0)*aperture
            if n%2==0:
                p = 1.0*p
            if m==0:
                l += 1
                Z[l] = p
            else:
                if l%2==0:
                    l += 1
                    Z[l] = p*np.cos(thetas[m])
                    l += 1
                    Z[l] = p*np.sin(thetas[m])
                else:
                    l += 1
                    Z[l] = p*np.sin(thetas[m])
                    l += 1
                    Z[l] = p*np.cos(thetas[m])     
            if l>Nk-2:
                break
    
    for j in range(Nk):
        index_n[j],index_m[j] = noll2nl(j+1)

    return Z, aperture,index_n,index_m

def  prechirpz(xsize,qsize,N,M):

    L = N+M-1
    sigma = 2*np.pi*xsize*qsize/N/M
    Afac = np.exp(2*1j*sigma*(1-M))
    Bfac = np.exp(2*1j*sigma*(1-N))
    sqW = np.exp(2*1j*sigma)
    W = sqW**2
    Gfac = (2*xsize/N)*np.exp(1j*sigma*(1-N)*(1-M))

    Utmp = np.zeros((N,1),dtype=np.complex)
    A = np.zeros((N,1),dtype=np.complex)
    Utmp[0] = sqW*Afac
    A[0] = 1.0
    for i in range(1,N):
        A[i] = Utmp[i-1]*A[i-1]
        Utmp[i] = Utmp[i-1]*W
    
    
    Utmp = np.zeros((M,1),dtype=np.complex)
    B = np.ones((M,1),dtype=np.complex)
    Utmp[0] = sqW*Bfac
    B[0] = Gfac
    for i in range(1,M):
        B[i] = Utmp[i-1]*B[i-1]
        Utmp[i] = Utmp[i-1]*W
    

    Utmp = np.zeros((max(N,M)+1,1),dtype=np.complex)
    Vtmp = np.zeros((max(N,M)+1,1),dtype=np.complex)
    Utmp[0] = sqW
    Vtmp[0] = 1.0
    for i in range(1,max(N,M)+1):
        Vtmp[i] = Utmp[i-1]*Vtmp[i-1]
        Utmp[i] = Utmp[i-1]*W
    
    D = np.ones((L,1),dtype=np.complex)
    for i in range(0,M):
        D[i] = np.conj(Vtmp[i])
    
    for i in range(0,N):
        D[-i-1] = np.conj(Vtmp[i+1])
    
    
    D = fft.fft(D,axis=0)
    A = A.transpose()
    B = B.transpose()
    D = D.transpose()
    return A, B, D


def cztfunc(datain,param):
 
    A = param[0]
    B = param[1]
    D = param[2]
    N = A.shape[1]
    M = B.shape[1]
    L = D.shape[1]
    K = datain.shape[-2]

    Amt = np.repeat(A,K,axis=0)
    Bmt = np.repeat(B,K,axis=0)
    Dmt = np.repeat(D,K,axis=0)

    cztin = tf.concat((Amt*datain,tf.zeros(datain.shape[0:-1]+(L-N),tf.complex64)),axis=-1)
    temp = Dmt*tf.signal.fft(cztin)
    cztout = tf.signal.ifft(temp)
    dataout = Bmt*cztout[...,0:M]
  
    return dataout
