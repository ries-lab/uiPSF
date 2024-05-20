
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
from scipy.ndimage.filters import gaussian_filter
defaultTFDataType="float32"
defaultTFCpxDataType="complex64"
#%%
# The functions below are Tensorflow now

def fft3d(tfin):
    return tf.signal.fftshift(tf.signal.fft3d(tf.signal.fftshift(tfin,axes=[-1,-2,-3])),axes=[-1,-2,-3])

def ifft3d(tfin):
    return tf.signal.ifftshift(tf.signal.ifft3d(tf.signal.ifftshift(tfin,axes=[-1,-2,-3])),axes=[-1,-2,-3])

def fft2d(tfin):
    return tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(tfin,axes=[-1,-2])),axes=[-1,-2])

def ifft2d(tfin):
    return tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(tfin,axes=[-1,-2])),axes=[-1,-2])



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
    
    return np.int32(j)

def noll2nl(j):
    n = np.ceil((-3 + np.sqrt(1 + 8*j)) / 2)
    l = j - n * (n + 1) / 2 - 1
    if np.mod(n, 2) != np.mod(l, 2):
       l = l + 1
    
    if np.mod(j, 2) == 1:
       l= -l
    
    return np.int32(n),np.int32(l)

def radialpoly(n,m,rho):
    if m==0:
        g = np.sqrt(n+1)
    else:
        g = np.sqrt(2*n+2)
    r = np.zeros(rho.shape)
    for k in range(0,(n-m)//2+1):
        coeff = g*((-1)**k)*factorial(n-k)/factorial(k)/factorial((n+m)//2-k)/factorial((n-m)//2-k)
        p = rho**(n-2*k)
        r += coeff*p

    return r

def genZern1(n_max,xsz):
    Nk = (n_max+1)*(n_max+2)//2
    Z = np.ones((Nk,xsz,xsz))
    pkx = 2/xsz
    xrange = np.linspace(-xsz/2+0.5,xsz/2-0.5,xsz)
    [xx,yy] = np.meshgrid(xrange,xrange)
    rho = np.lib.scimath.sqrt((xx*pkx)**2+(yy*pkx)**2)
    phi = np.arctan2(yy,xx)

    for j in range(0,Nk):
        [n,l] = noll2nl(j+1)
        m = np.abs(l)
        r = radialpoly(n,m,rho)
        if l<0:
            Z[j] = r*np.sin(phi*m)
        else:
            Z[j] = r*np.cos(phi*m)
    return Z




def prechirpz1(kpixelsize,pixelsize_x,pixelsize_y,N,M):
    krange = np.linspace(-N/2+0.5,N/2-0.5,N,dtype=np.float32)
    [xxK,yyK] = np.meshgrid(krange,krange)
    xrange = np.linspace(-M/2+0.5,M/2-0.5,M,dtype=np.float32)
    [xxR,yyR] = np.meshgrid(xrange,xrange)
    a = 1j*np.pi*kpixelsize
    A = np.exp(a*(pixelsize_x*xxK*xxK+pixelsize_y*yyK*yyK))
    C = np.exp(a*(pixelsize_x*xxR*xxR+pixelsize_y*yyR*yyR))

    brange = np.linspace(-(N+M)/2+1,(N+M)/2-1,N+M-1,dtype=np.float32)
    [xxB,yyB] = np.meshgrid(brange,brange)
    B = np.exp(-a*(pixelsize_x*xxB*xxB+pixelsize_y*yyB*yyB))
    Bh = tf.signal.fft2d(B)

    return A,Bh,C


def cztfunc1(datain,param):
    A = param[0]
    Bh = param[1]
    C = param[2]
    N = A.shape[0]
    L = Bh.shape[0]
    M = C.shape[0]

    Apad = tf.concat((A*datain/N,tf.zeros(datain.shape[0:-1]+(L-N),tf.complex64)),axis=-1)
    Apad = tf.concat((Apad,tf.zeros(Apad.shape[0:-2]+(L-N,Apad.shape[-1]),tf.complex64)),axis=-2)
    Ah = tf.signal.fft2d(Apad)
    cztout = tf.signal.ifft2d(Ah*Bh/L)
    dataout = C*cztout[...,-M:,-M:]

    return dataout

def gen_layers(imagesize,sigma,poolsize,roisize):
    # create a gaussian difference convolutional layer
    kernelsize = tuple(np.array(sigma)*3+1)
    conv2d = tf.keras.layers.Conv2D(1, kernelsize, strides=(1, 1), padding="same")
    conv2d.build(input_shape=(None, imagesize[-2], imagesize[-1], 1))
    kernel = np.zeros(kernelsize)
    cc = np.array(kernelsize)//2
    kernel[cc[0],cc[1]] = 1
    kernel = gaussian_filter(kernel, list(np.array(sigma)*0.75))-gaussian_filter(kernel, sigma)
    kernel = kernel.reshape(kernel.shape+(1,1))
    bias = np.zeros(1)
    conv2d.set_weights([kernel,bias])

    # create a max pooling layer
    max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=tuple(poolsize),strides=(1, 1), padding="same")

    # create a uniform convolutional layer
    conv_uniform = tf.keras.layers.Conv2D(1, tuple(roisize), strides=(1, 1), padding="same")
    conv_uniform.build(input_shape=(None, imagesize[-2], imagesize[-1], 1))
    kernel = np.ones(tuple(roisize)+(1,1))
    bias = np.zeros(1)
    conv_uniform.set_weights([kernel,bias])

    return conv2d, max_pool_2d, conv_uniform

def crop_rois(imgin,conv2d,max_pool_2d,conv_uniform,thresh,roisize):
    # apply gaussian difference filter
    img = np.reshape(imgin,imgin.shape+(1,))
    imgconv = conv2d(img)
    # apply max pooling
    imgmax = max_pool_2d(imgconv)
    # find local maxima
    mask = tf.math.equal(imgmax,imgconv)
    locmax = tf.cast(mask,tf.float32) * imgconv
    thresh = np.quantile(locmax[mask],1-1e-4) * thresh
    # remove overlapping peaks
    locmax1= tf.cast(tf.math.greater(locmax,thresh),tf.float32)
    locmaxf = conv_uniform(locmax1)
    locmax1 = locmaxf*locmax1
    coords = tf.where((locmax1>0)&(locmax1<2))
    coords = np.array(coords[:,:-1])
    # remove peaks at the image border
    sz = roisize
    mask = (coords[:,1]-sz[0]//2 >=0) & (coords[:,1]-sz[0]//2+sz[0] <= img.shape[1]) & (coords[:,2]-sz[1]//2 >=0) & (coords[:,2]-sz[1]//2+sz[1] <= img.shape[2])
    coords = coords[mask]
    # crop rois
    rois = np.zeros((coords.shape[0],)+tuple(sz))
    for k,cor in enumerate(coords):
        rois[k] = imgin[cor[0],cor[1]-sz[0]//2:cor[1]-sz[0]//2+sz[0],cor[2]-sz[1]//2:cor[2]-sz[1]//2+sz[1]]

    return rois,coords
    