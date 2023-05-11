import os

import h5py
import numpy as np
import scipy as sp
import scipy.special as spf

import matplotlib.pyplot as plt
from .. import imagetools as nip
from tkinter import messagebox as mbox
import sys

from .PreprocessedImageDataInterface_file import PreprocessedImageDataInterface

class PreprocessedImageDataSingleChannel(PreprocessedImageDataInterface):
    """
    Class that handles preprocessed data for single-channel case.
    Provides access to images data (rois, centers, etc.) for fitter and psf classes.
    """
    def __init__(self, images, is4pi=None) -> None:
        # TODO: instead of using a boolean flag one could think of using a string
        # this would allow for more options
        # the question is if this makes sense or if it makes more sense to create a new class
        # for other types of psfs
        # here we used the flag since almost everything is identical, only the check function
        # and the func_2Dimage are different
        if is4pi is None or is4pi is False:
            self.is4pi = False
            self.num_dims = 4
            self.dim_names = "images, z, y, x"
            self.func_2Dimage = lambda ims: np.max(ims, axis=-3) # used in find_rois()
        elif is4pi is True:
            self.is4pi = True
            self.num_dims = 5
            self.dim_names = "images, phi, z, y, x"
            self.func_2Dimage = lambda ims: np.max(ims[0], axis=0) # used in find_rois()
        else:
            raise ValueError("is4pi should be True or False.")

        self.images = None
        self.check_and_init_images(images) # check if input is valid
        self.rois = []
        self.centers = []
        self.file_idxs = []
        self.rois_available = False
        self.min_border_dist = None # needed in cut_new_rois()
        self.skew_const = None
        self.zT = None
        return

    def check_and_init_images(self, images):
        """
        Checks if input is valid and initializes image attribute. 
        """
        try:
            # check if everything has same shape
            # and cast to float32 --> allows correct loss calculation in fitter
            self.images = np.array(images, dtype=np.float32)
        except:
            raise ValueError("Was not able to convert input to numpy array.\nCheck that dimensions are the same for all channels and for all images.")
        
        if self.images.ndim != self.num_dims:
            raise ValueError(f"Input needs to have {self.num_dims} dimensions: {self.dim_names}.")

        return

    def find_rois(self, roi_size, gaus_sigma, min_border_dist, max_threshold, max_kernel,FOV=None, min_center_dist=None, max_bead_number=None):
        """
        Cuts out rois around local maxima.
        """
        self.min_border_dist = min_border_dist #  needed in cut_new_rois()

        all_rois = []
        all_centers = []
        file_idxs = []

        for file_idx, image in enumerate(self.images):
            # TODO: try/except, since nip.extractMuliPeaks throws error if no roi is found
            if len(roi_size)>2:
                im2 = image
            else:
                im2 = self.func_2Dimage(image)
            rois, centers = nip.extractMultiPeaks(im2, ROIsize=roi_size, sigma=gaus_sigma,
                                                borderDist=min_border_dist, threshold_rel=max_threshold,
                                                alternateImg=image, kernel=max_kernel)
            
            # remove rois/centers that are to close together
            if rois is not None:
                if min_center_dist is None:
                    min_center_dist = np.hypot(roi_size[-2], roi_size[-1])
                rois, centers = self.remove_close_rois(rois, centers, min_center_dist)
                if FOV is not None:        
                    fov = np.array(FOV)
                    #inFov = (coordinates[:,-1]>= fov[0]-fov[2]/2) & (coordinates[:,-1] <= fov[0]+fov[2]/2) & (coordinates[:,-2]>= fov[1]-fov[3]/2) & (coordinates[:,-2] <= fov[1]+fov[3]/2)
                    coord_r = (centers[:,-1]-fov[1])**2+(centers[:,-2]-fov[0])**2
                    inFov = coord_r<(fov[2]**2)
                    rois = rois[inFov]
                    centers = centers[inFov]

                all_rois.append(rois)
                all_centers.append(centers)
                file_idxs += [file_idx] * rois.shape[0]
            if max_bead_number:
                if len(file_idxs)>max_bead_number:
                    break
        L = np.min((max_bead_number,len(file_idxs)))
        # convert to numpy arrays and make sure everything has correct dtypes
        if not all_rois:
            #mbox.showerror("segmentation error","no bead is found")
            raise RuntimeError('no bead is found')
        self.rois = np.concatenate(all_rois)[0:L].astype(np.float32)
        self.centers = np.concatenate(all_centers)[0:L].astype(np.int32)
        self.centers_all = np.concatenate(all_centers)[0:L].astype(np.int32)
        self.file_idxs = np.array(file_idxs)[0:L].astype(np.int32)
        self.rois_available = True
        self.image_size = self.images.shape
        return


    def remove_close_rois(self, rois, centers, min_dist):
        """
        Calculates the distance between all rois/centers and removes the ones
        that are to close to each other in order to ensure that there is only
        one signal/bead per roi.
        """
        # TODO: there is one corner case that is not handled here:
        # if two beads are close together and one (and only one) is to close to border
        # in this case only the rois that is not to close to the border is cut
        # but since the other one is not the first one is not filtered out here
        # so it could be possible that there are two beads visible in one roi...
        dist_matrix = sp.spatial.distance_matrix(centers, centers)
        keep_matrix_idxs = np.where((0 == dist_matrix) | (dist_matrix > min_dist))
        unique, counts = np.unique(keep_matrix_idxs[0], return_counts=True)
        keep_idxs = unique[counts == centers.shape[0]]
        return rois[keep_idxs], centers[keep_idxs]

    def cut_new_rois(self, centers, file_idxs, roi_size=None, min_border_dist=None):
        """
        Cuts new rois from images with specified centers.
        """
        # set default values
        if roi_size is None:
            roi_size = self.rois.shape[-2:]
        if min_border_dist is None:
            min_border_dist = self.min_border_dist
        
        if len(roi_size)==3:
            Nz = roi_size[0]
        else:
            Nz = self.images.shape[-3]

        if hasattr(self,'skew_const'):
            if self.skew_const:

                roisize_x = np.int32(roi_size[-1]+Nz*np.abs(self.skew_const[-1])+1)
                roisize_y = np.int32(roi_size[-2]+Nz*np.abs(self.skew_const[-2])+1)
                if len(roi_size)==3:
                    roi_shape = [roi_size[0],roisize_y,roisize_x]
                else:
                    roi_shape = [roisize_x,roisize_x]
            else:
                roi_shape = roi_size
        else:
            roi_shape = roi_size
        # checking border_dist not needed since we check this in psf class
        # nevertheless, I left it here just in case one does need it for another purpose
        '''
        # make sure rois are not too close to border
        # adapted from NanoImaginPack --> coordinates.py --> extractMuliPeaks()
        border_dist = np.array(min_border_dist)
        valid_idxs = np.all(centers - border_dist >= 0, axis=1) & np.all(self.images.shape[-2:] - centers - border_dist >= 0, axis=1)
        centers = centers[valid_idxs, :]
        '''

        # iterate over file_index to make sure that new roi is cut from correct file
        new_rois = []
        for i, file_idx in enumerate(file_idxs):
            new_rois.append(nip.multiROIExtract(self.images[file_idx], [centers[i]], roi_shape))

        # convert to numpy arrays and make sure everything has correct dtypes
        self.rois = np.concatenate(new_rois).astype(np.float32)
        self.centers = centers.astype(np.int32)
        self.file_idxs = file_idxs.astype(np.int32)
        self.rois_available = True

        return

    def get_image_data(self):
        """
        Provides the necessary image information (e.g., rois, centers, ...) for the psf class
        and the fitter class.
        """
        if self.rois_available:
            return self.images, self.rois, self.centers, self.file_idxs
        else:
            raise RuntimeError("Can't call 'get_image_data()' since 'rois_available' flag is False.\nThis is probably due to the fact that you did not call 'find_rois()' before using this ImageData.")

    def process(self,roi_size, gaus_sigma, min_border_dist, max_threshold, max_kernel,pixelsize_x,pixelsize_z,bead_radius, 
                min_center_dist=None,FOV=None, modulation_period=None,padPSF=True,plot=True, isVolume=True, pixelsize_y=None, skew_const=None,max_bead_number=None):

        if len(roi_size)==3:
            Nz = roi_size[0]
        else:
            Nz = self.images.shape[-3]
        if skew_const:

            roisize_x = np.int32(1+roi_size[-1]+Nz*np.abs(skew_const[-1]))
            roisize_y = np.int32(1+roi_size[-2]+Nz*np.abs(skew_const[-2]))
            if len(roi_size)==3:
                roiszL = [roi_size[0],roisize_y,roisize_x]
            else:
                roiszL = [roisize_x,roisize_x]
            min_border_dist=list(np.array(roiszL)//2+1)
            self.find_rois(roiszL, gaus_sigma, min_border_dist, max_threshold, max_kernel, FOV, min_center_dist,max_bead_number)
        else:
            self.find_rois(roi_size, gaus_sigma, min_border_dist, max_threshold, max_kernel, FOV, min_center_dist,max_bead_number)
        img, rois, cor, _ = self.get_image_data()
        self.centers_all = cor
        self.image_size = img.shape
        print(f"rois shape channel : {rois.shape}")

        self.pixelsize_z = pixelsize_z
        self.pixelsize_x = pixelsize_x
        self.bead_radius = bead_radius
        offset = np.min((np.quantile(rois,1e-3),0))
        self.rois = rois-offset
        if plot:
            plt.figure(figsize=[6,6])
            plt.plot(cor[:,-1],cor[:,-2],'o',markersize = 8,markerfacecolor='none')
            plt.show()
        # pad rois along z dimension
        if padPSF:
            _, rois, _, _ = self.get_image_data()
            value = np.empty((), dtype=object)
            value[()] = (0, 0)
            padsize = np.full((len(rois.shape), ), value, dtype=object)
            padsize[-3] = (np.int32(bead_radius//pixelsize_z),np.int32(bead_radius//pixelsize_z))
            roisL = np.pad(rois,tuple(padsize),mode='edge')
            self.rois = roisL
            print(f"padded rois shape channel : {roisL.shape}")

        # generate bead kernel
        if pixelsize_y is None:
            pixelsize_y = pixelsize_x
        self.pixelsize_y = pixelsize_y
 

        
        if modulation_period is not None:
            self.zT = modulation_period/pixelsize_z

        self.skew_const = skew_const
        if skew_const:
            self.deskew_roi(roi_size)
        return

    def deskew_roi(self,roi_size):
        _, rois, cor, _ = self.get_image_data()
        skew_const = self.skew_const
        Nz = rois.shape[-3]
        roisize_x = rois.shape[-1]
        roisize_y = rois.shape[-2]
        bxsz = roi_size
        rois1 = np.zeros(rois.shape[0:-2]+(bxsz[-2],bxsz[-1]),dtype = np.float32)
        for i in range(0,Nz):
            ccx = np.int32(np.round(roisize_x//2-skew_const[-1]*Nz/2 + i*skew_const[-1]))
            ccy = np.int32(np.round(roisize_y//2-skew_const[-2]*Nz/2 + i*skew_const[-2]))
            tmp = rois[...,i,ccy-bxsz[-2]//2:ccy+bxsz[-2]//2+bxsz[-2]%2,ccx-bxsz[-1]//2:ccx+bxsz[-1]//2+bxsz[-1]%2]
            rois1[...,i,:,:] = tmp
        self.rois = rois1
        self.skew_const = skew_const
        print(f"deskewed rois shape channel : {rois1.shape}")