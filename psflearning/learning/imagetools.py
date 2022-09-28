"""
Copyright (c) 2022      Ries Lab, EMBL, Heidelberg, Germany
All rights reserved     Heintzmann Lab, Friedrich-Schiller-University Jena, Germany

@author: Rainer Heintzmann, Sheng Liu
"""


import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndimage
from scipy.spatial import distance
from scipy import cluster as cluster
import warnings
import numbers


def extractMultiPeaks(im, ROIsize, sigma=None, threshold_rel=None, alternateImg=None, kernel=(3,3,3), borderDist=None, FOV=None):
    """
    extracts ROIs around the local maxima in im after gaussian filtering
    :param im: image to extract from
    :param ROIsize: multidimensional size of the ROI to extract around each maximum. If fewer dimensions are given the others are not extracted and the original size is kept.
    :param sigma: size of the Gaussian filter kernel
    :param  threshold_abs, threshold_rel: absolute and relative thesholds to extract peaks
    :param min_distance: minimum distance to keep around maxima
    :return: tuple of (the n-dimensional ROIs stacked along an extra dimension and center coordinates)
    """
    if sigma is not None and np.linalg.norm(sigma) > 0:
        im2 = gaussian_filter(im, sigma)
    else:
        im2 = im
    coordinates = localMax(im2, threshold_rel=threshold_rel, kernel=kernel)
    coordinates = np.array(coordinates)
    if coordinates.size>0:
        if borderDist is not None:        
            borderDist = np.array(borderDist)
            inBorder = np.all(coordinates-borderDist >= 0,axis=1) & np.all(im.shape - coordinates - borderDist >= 0, axis=1)
            coordinates=coordinates[inBorder,:]
            #values=values[inBorder]
        if FOV is not None:        
            fov = np.array(FOV)
            #inFov = (coordinates[:,-1]>= fov[0]-fov[2]/2) & (coordinates[:,-1] <= fov[0]+fov[2]/2) & (coordinates[:,-2]>= fov[1]-fov[3]/2) & (coordinates[:,-2] <= fov[1]+fov[3]/2)
            coord_r = (coordinates[:,-1]-fov[1])**2+(coordinates[:,-2]-fov[0])**2
            inFov = coord_r<(fov[2]**2)
            coordinates=coordinates[inFov,:]

    if alternateImg is not None:
        im = alternateImg
    centers = np.round(coordinates).astype(np.int)
    if len(ROIsize) < centers.shape[-1]:
        centers = centers[:,-len(ROIsize):]
    if coordinates.size>0:
        ROIs = multiROIExtract(im, centers, ROIsize=ROIsize)  # , origin="center"
    else:
        ROIs = None
    return ROIs, centers

def extractMultiPeaks_smlm(im, ROIsize, sigma=None, threshold_rel=None, alternateImg=None, kernel=(3,3,3), borderDist=None, min_dist=None,FOV=None):
    """
    extracts ROIs around the local maxima in im after gaussian filtering
    :param im: image to extract from
    :param ROIsize: multidimensional size of the ROI to extract around each maximum. If fewer dimensions are given the others are not extracted and the original size is kept.
    :param sigma: size of the Gaussian filter kernel
    :param  threshold_abs, threshold_rel: absolute and relative thesholds to extract peaks
    :param min_distance: minimum distance to keep around maxima
    :return: tuple of (the n-dimensional ROIs stacked along an extra dimension and center coordinates)
    """
    if sigma is not None and np.linalg.norm(sigma) > 0:
        im2 = gaussian_filter(im, list(np.array(sigma)*0.75))-gaussian_filter(im,sigma)
    else:
        im2 = im
    coordinates = localMax(im2, threshold_rel=threshold_rel, kernel=kernel)
    coordinates = np.array(coordinates)
    if coordinates.size>0:
        if borderDist is not None:        
            borderDist = np.array(borderDist)
            inBorder = np.all(coordinates-borderDist >= 0,axis=1) & np.all(im.shape - coordinates - borderDist >= 0, axis=1)
            coordinates=coordinates[inBorder,:]
            #values=values[inBorder]
        if FOV is not None:        
            fov = np.array(FOV)
            #inFov = (coordinates[:,-1]>= fov[0]-fov[2]/2) & (coordinates[:,-1] <= fov[0]+fov[2]/2) & (coordinates[:,-2]>= fov[1]-fov[3]/2) & (coordinates[:,-2] <= fov[1]+fov[3]/2)
            coord_r = (coordinates[:,-1]-fov[1])**2+(coordinates[:,-2]-fov[0])**2
            inFov = coord_r<(fov[2]**2)
            coordinates=coordinates[inFov,:]

    if alternateImg is not None:
        im = alternateImg
    centers = np.round(coordinates).astype(np.int)
    if len(ROIsize) < centers.shape[-1]:
        centers = centers[:,-len(ROIsize):]
    if coordinates.size>0:
        #if (min_dist is not None) & (centers.shape[0]>1):
        #   centers = combine_close_cor(centers, min_dist)
        ROIs = multiROIExtract_smlm(im, centers, ROIsize=ROIsize)  # , origin="center"
    else:
        ROIs = None
    return ROIs, centers
    
def localMax(img, threshold_rel=None, kernel=(3,3,3)):
    imgMax = ndimage.maximum_filter(img, size=kernel)
    imgMax = (imgMax == img) * img # extracts only the local maxima but leaves the values in
    mask = imgMax==img
    if threshold_rel is not None:
        thresh = np.quantile(img[mask],1-1e-4) * threshold_rel
        labels, num_labels = ndimage.label(imgMax > thresh)
    else:
        labels, num_labels = ndimage.label(imgMax)

    # Get the positions of the maxima
    coords = ndimage.measurements.center_of_mass(img, labels=labels, index=np.arange(1, num_labels + 1))

    # Get the maximum value in the labels
    #values = ndimage.measurements.maximum(img, labels=labels, index=np.arange(1, num_labels + 1))
    return coords


def multiROIExtract(im, centers, ROIsize):
    """
    extracts multiple ROIs indicated by a list of center corrdinates and stacks them in another dimension
    :param im: image to extract from
    :param centers: list or tuple of center coordinates. Leading entries are ignored in each vector if more entries than dimensions.
    :param ROIsize: multidimensional size of the ROI to extract. If fewer dimensions are given the others are not extracted and the original size is kept.
    :return: the stacked extractions. If the ROI overlaps the border zeros will be padded.
    """
    listOfROIs = []
    for centerpos in centers:
        if len(centerpos) > im.ndim:
            centerpos=centerpos[-im.ndim:]
        if len(ROIsize) < len(centerpos):
            centerpos = centerpos[-len(ROIsize):]

        myROI = extract(im, ROIsize=ROIsize, centerpos=centerpos)
        #myROI = im[ROIcoords(centerpos, ROIsize, im.ndim)]
        listOfROIs.append(myROI)
    return np.stack(listOfROIs)


def multiROIExtract_smlm(im, centers, ROIsize):

    listOfROIs = []
    for centerpos in centers:
        myROI = im[ROIcoords(centerpos, ROIsize, im.ndim)]
        listOfROIs.append(myROI)
    return np.stack(listOfROIs)
 
def combine_close_cor(centers, min_dist):

    dis = distance.pdist(centers)
    link = cluster.hierarchy.linkage(dis,'complete')
    Tc = cluster.hierarchy.fcluster(link,t=min_dist,criterion ='distance')
    cor = np.zeros((np.max(Tc),2),dtype=np.int32)
    for t in range(0,np.max(Tc)):
        maskT = (Tc==(t+1))
        if np.sum(maskT)>1:
            cor[t] = np.mean(centers[maskT],axis=0)
        else:
            cor[t] = centers[maskT]

    return cor
def extract(img, ROIsize=None, centerpos=None, PadValue=0.0, checkComplex=True):
    '''
    EXTRACT a part in an n-dimensional array based on stating the destination ROI size and center in the source
    :param img: Input image
    :param ROIsize: region of interest to extract ((minx,maxx),(miny,maxy))
    :param centerpos: center of the ROI in source image to extract. Coordinates are measured from the corner being (0,0,..)
    :param PadValue: Value to assign to the padded area. If PadValue==None, no padding is performed and the non-existing regions are pruned.
    :param checkComplex: ToDO: What is this used for?
    :return: an extracted image

    Example1:
    import NanoImagingPack as nip
    im = nip.readim()
    im.extract([128,128]) #EXTRACT an ROI of 128*128 from centre of image

    Example1:
    import NanoImagingPack as nip
    im = nip.readim()
    im.extract([128,128],[128,128]) #EXTRACT an ROI of 128*128 with coordinate 128,128 as centre
    '''

    if checkComplex:
        if np.iscomplexobj(img):
            raise ValueError(
                "Found complex-valued input image. For Fourier-space extraction use extractFt, which handles the borders or use checkComplex=False as an argument to this function")

    mysize = img.shape

    if ROIsize is None:
        ROIsize = mysize
    else:
        ROIsize = expanddimvec(ROIsize, len(mysize), mysize)

    mycenter = [sd // 2 for sd in mysize]
    if centerpos is None:
        centerpos = mycenter
    else:
        centerpos = coordsToPos(expanddimvec(centerpos, img.ndim, othersizes=mycenter), mysize)

    #    print(ROIcoords(centerpos,ROIsize,img.ndim))
    res = img[ROIcoords(centerpos, ROIsize, img.ndim)]
    if PadValue is None:
        return res
    else:  # perform padding
        pads = [(max(0, ROIsize[d] // 2 - centerpos[d]), max(0, centerpos[d] + ROIsize[d] - mysize[d] - ROIsize[d] // 2)) for d in range(img.ndim)]
        #        print(pads)
        resF = np.pad(res, tuple(pads), 'constant', constant_values=PadValue)
        return resF
    

def expanddimvec(shape, ndims, othersizes=None, trailing=False, value=1):
    '''
        expands an nd tuple (e.g image shape) to the necessary number of dimension by inserting leading (or trailing) dimensions
        ----------
        img: input image to expand
        ndims: number of dimensions to expand to
        trailing (default:False) : append trailing dimensions rather than dimensions at the front of the size vector
        othersizes (defatul:None) : do not expand with ones, but rather use the provided sizes

        see also:
        castdimvec
    '''
    if shape is None:
        return None
    if isinstance(shape,numbers.Number):
        shape=(shape,)
    else:
        shape=tuple(shape)
    missingdims=ndims-len(shape)
    if missingdims > 0:
        if othersizes is None:
            if trailing:
                return shape+(missingdims)*(value,)
            else:
                return (missingdims)*(value,)+shape
        else:
            if trailing:
                return shape+tuple(othersizes[-missingdims::])
            else:
                return tuple(othersizes[0:missingdims])+shape
    else:
        return shape[-ndims:]


def coordsToPos(coords,ashape):
    '''
        converts a coordinate vector to a list of all-positive number using a given shape.

        coords: list, tuple or np.array of positions (mixed positive and negative)
        ashape: vector of shape with the same length

    '''
    mylen=len(coords)
    assert(mylen==len(ashape))
    return [coords[d]+(coords[d]<0)*ashape[d] for d in range(mylen)]



def ROIcoords(center,asize,ndim=None):
    """
        constructs a coordinate vector which can be used by numpy for an array acccess
        center: list or tuple of center coordinates
        asize: size of the ROI to extract. Will automatically be limited by the array sizes when applied
        ndim (default=None): total number of dimensions of the array (generates preceeding ":" for access)
    """

    if ndim==None:
        ndim=len(center)

    slices=[]
    if ndim>len(center):
        slices=(ndim-len(center))*slice(None)
    for d in range(ndim-len(center),ndim): # only specify the last dimensions
        asp = asize[d]
        if asp < 0:
            raise ValueError("ashape has to be >= 0")
        astart = center[d]-asp//2
        astop = astart + asp
        slices.append(slice(max(astart,0),max(astop,0)))

    return tuple(slices)


def ramp1D(mysize=256, ramp_dim=-1, placement='center', freq=None, pixelsize=None):
    """
    creates a 1D-ramp along only one dimension. The trailing dimension sizes are all one.

    This guarantees a fast performance for functions such as rr, since the broadcasting features of Phython are going to deal with the other dimensions.

    standart size is 256
    placement:
        center: 0 is at center
                if x is even, it has one more value in the positve:
                    e.g. size_x = 100 -> goes from -50 to 49
                         size_x = 101 -> goes from -50 to 50
        negative : goes from negative size_x to 0
        positive : goes from 0 size_x to positive
        freq : if "ftfreq" is given, the Fourier-space frequency scale (roughly -0.5 to 0.5) is used.
        int number: is the index where the center is!
    """
    if isinstance(mysize, np.ndarray):
        mysize = mysize.shape
    if freq != None and not placement == 'center':
        raise ValueError(
            "ramp1D: Illegal placement: (" + placement + "). (freq=" + freq + ") argument can only be used with (center) placement.")
    if placement == 'negative':
        miniramp = np.arange(-mysize + 1, 1, 1)
    elif placement == 'positive' or placement == 'corner':
        miniramp = np.arange(0, mysize, 1)
    elif placement == 'center':
        miniramp = np.arange(-mysize // 2 + np.mod(mysize, 2), mysize // 2 + np.mod(mysize, 2), 1)
    elif (type(placement) == int or type(placement) == float):
        miniramp = np.arange(0, mysize, 1) - placement
    else:
        try:
            if np.issubdtype(placement.dtype, np.number):
                miniramp = np.arange(0, mysize, 1) - placement
            else:
                raise ValueError(
                    'ramp: unknown placement value. allowed are negative, positive, corner, and center or an offset value as an np.number')
        except AttributeError:
            print(placement)
            raise ValueError(
                'ramp: unknown placement value. allowed are negative, positive, placement, and center or an offset value as an np.number')
    if freq == "ftfreq":
        miniramp = miniramp / mysize
    elif freq == "ftradfreq":
        miniramp = miniramp * 2.0 * np.pi / mysize
    elif freq == "fftfreq":
        miniramp = np.fft.fftfreq(mysize) # , pixelsize
    elif freq == "rfftfreq":
        miniramp = np.fft.rfftfreq(mysize) # , pixelsize
    elif freq == "fftradfreq":
        miniramp = np.fft.fftfreq(mysize,1.0 / 2.0 / np.pi) # pixelsize
    elif freq == "rfftradfreq":
        miniramp = np.fft.rfftfreq(mysize, 1.0 / 2.0 / np.pi) # pixelsize
    elif not freq == None:
        raise ValueError(
            "unknown option for freq. Valid options are ftfreq, ftradfreq, fftfreq, rfftfreq, fftradfreq and rfftradfreq.")
    #        miniramp=miniramp*(np.pi/(mysize//2))

    if ramp_dim > 0:
        miniramp = expanddim(miniramp, ramp_dim + 1, trailing=False)  # expands to this dimension numbe by inserting trailing axes. Also converts to
    elif ramp_dim < -1:
        miniramp = expanddim(miniramp, -ramp_dim, trailing=True)  # expands to this dimension numbe by inserting prevailing axes. Also converts to
    
    return miniramp



def expanddim(img, ndims, trailing=None):
    """
        expands an nd image to the necessary number of dimension by inserting leading dimensions
        ----------
        img: input image to expand
        ndims: number of dimensions to expand to. If negative, this will be interpreted to expand to abs(ndims) with trailing=True, if trailing is None.
        trailing (default:False) : append trailing dimensions rather than dimensions at the front of the size vector

        Example:
            import NanoImagingPack as nip
            expanddim(nip.readim(),-3)
    """
    if trailing is None:
        trailing = ndims < 0

    if ndims < 0:
        ndims = -ndims
    res = np.reshape(img, expanddimvec(img.shape, ndims, None, trailing))

    return res


def zz(mysize=(256, 256), placement='center', freq=None, pixelsize=None):
    """
    creates a ramp in z direction
    standart size is 256 X 256
    placement:
        center: 0 is at cetner
                if y is even, it has one more value in the positve:
                    e.g. size_y = 100 -> goes from -50 to 49
        negative : goes from negative size_y to 0
        positvie : goes from 0 size_y to positive
    """
    myplacement=placement
    if (type(placement) is list) or (type(placement) is np.array):
        myplacement = placement[-3]
    return (ramp(mysize, -3, myplacement, pixelsize=pixelsize))

def ramp(mysize=(256, 256), ramp_dim=-1, placement='center', freq=None, shift=False, rftdir=-1, pixelsize=None):
    """
    creates a ramp in the given direction direction
    standard size is 256 X 256
    placement:
        center: 0 is at center
                if x is even, it has one more value in the positve:
                    e.g. size_x = 100 -> goes from -50 to 49
                         size_x = 101 -> goes from -50 to 50
        negative : goes from negative size_x to 0
        positive : goes from 0 size_x to positive
        freq : if "freq" is given, the Fourier-space frequency scale (roughly -0.5 to 0.5) is used.
        int number: is the index where the center is!
    """
    mysize = unifysize(mysize)

    ndims = len(mysize)
    if ramp_dim >= ndims:
        raise ValueError(
            "ramp dimension (" + str(ramp_dim) + ") has to be smaller than number of available dimensions (" + str(
                ndims) + ") specified by the size vector")
    if (-ramp_dim) > ndims:
        raise ValueError(
            "negative ramp dimension has to be smaller or equal than number of available dimensions specified by the size vector")
    if ramp_dim >= 0:
        ramp_dim = ramp_dim - ndims  # 0 in a 2D image should become -2
    if rftdir >= 0:  # WHAT IS THIS??? CK 16.03.2019
        rftdir = rftdir - ndims  # 0 in a 2D image should become -2

    if freq == "rfreq" and ramp_dim != rftdir:  # CAREFUL: The frequency-based methods have sometimes already been pre-shifted.
        freq = "freq"
        mysize[rftdir] = (mysize[rftdir] + 1) // 2

    myramp = ramp1D(mysize[ramp_dim], ramp_dim, placement, freq, pixelsize)
    mysize[ramp_dim] = myramp.shape[ramp_dim]  # since the rfreq option changes the size

    #    if freq=="rfreq" and ramp_dim==rftdir:  # CAREFUL: The frequency-based methods have sometimes already been pre-shifted.
    #            myramp =  np.fft.fftshift(myramp)
    if freq == "freq" and not shift:  # CAREFUL: The frequency-based methods have sometimes already been pre-shifted.
        myramp = np.fft.fftshift(myramp)

    res = ones(mysize)
    res *= myramp


  
    return res


def unifysize(mysize):
    if isinstance(mysize, list) or isinstance(mysize, tuple) or isinstance(mysize, np.ndarray):
        return list(mysize)
    else:
        return list(mysize.shape)

def ones(s, dtype=None, order='C', ax=None):
    if isnp(s):
        s=s.shape
    res = np.ones(s,dtype,order)
    if ax is not None:
        res = castdim(res, wanteddim=ax)
    return res

def isnp(animg):
    return isinstance(animg,np.ndarray)


def castdim(img, ndims=None, wanteddim=0):
    """
        expands a 1D image to the necessary number of dimension casting the dimension to a wanted one
        it orients a vector along the -wanteddim direction
        ----------
        img: input image to expand
        ndims: number of dimensions to expand to
        wanteddim: number that the one-D axis should end up in (default:0)
    """
    return np.reshape(img, castdimvec(img.shape, ndims, wanteddim))

def castdimvec(mysize, ndims=None, wanteddim=0):
    """
        expands a shape tuple to the necessary number of dimension casting the dimension to a wanted one
        ----------
        img: input image to expand
        ndims: number of dimensions to expand to. If None, wanteddim is used to determine the maximal size of dims
        wanteddim: number that the one-D axis should end up in (default:0)

        see also:
        expanddimvec
    """
    mysize = tuple(mysize)
    if ndims is None:
        if wanteddim >= 0:
            ndims = wanteddim + 1
        else:
            ndims = - wanteddim
    if wanteddim<0:
        wanteddim = ndims+wanteddim
    if wanteddim+len(mysize) > ndims:
        raise ValueError("castdim: ndims is smaller than requested total size including the object to place.")
    newshape = wanteddim*(1,)+mysize+(ndims-wanteddim-len(mysize))*(1,)
    return newshape


def FreqRamp1D(length,  k, d):  # cossqr is not better
    """
    creates a one-dimensional frequency ramp oriented along direction d. It can optionally be softerend at the transition region  to the nearest full pixel frequency to yield a smooth transition.
    :param length: length of the image to generate
    :param k: k-vector along this dimension
    :param d: dimension to orient it in
    :param relwidth: width of the transition region
    :param smooth: flag. If True, rounding to the nearest integer pixel will be applied
    :param func: transition function
    :param cornerFourier: If True, the phase ramp will be computed for a Fourier-layout in the corner. This is important when applying it to unshifted Fourier-space
    :return: the complex valued frequency ramp

    Example:
    import NanoImagingPack as nip
    a = nip.FreqRamp1D(100, 10, 12.3, -1)
    """

    res = k

    myramp = ramp1D(length, ramp_dim=d, freq='ftfreq')
    return np.exp(1j * 2 * np.pi * res * myramp)



def zeros(s, dtype=None, order='C',  ax=None):
    if isnp(s):
        s = s.shape
    res = np.zeros(s, dtype, order)
    if ax is not None:
        res = castdim(res, wanteddim=ax)
    return res

def dimToPositive(dimpos,ndims):
    """
        converts a dimension position to a positive number using a given length.

        dimpos: dimension to adress
        ndims: total number of dimensions

    """
    return dimpos+(dimpos<0)*ndims *ndims 