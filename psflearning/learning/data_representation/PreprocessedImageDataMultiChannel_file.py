import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from typing import Type

from .PreprocessedImageDataInterface_file import PreprocessedImageDataInterface

class PreprocessedImageDataMultiChannel(PreprocessedImageDataInterface):
    """
    Class that handles preprocessed data for multi-channel case.
    Provides access to images data (rois, centers, etc.) for fitter and psf classes.
    Is basically a wrapper around multiple instance of the provided single-channel class.
    """
    def __init__(self, images, single_channel_dtype: Type[PreprocessedImageDataInterface], is4pi=None) -> None:
        if is4pi is None or is4pi is False:
            self.is4pi = False            
        elif is4pi is True:
            self.is4pi = True           
        else:
            raise ValueError("is4pi should be True or False.")

        self.single_channel_dtype = single_channel_dtype
        self.rois_available = False
        self.channels = [] # each element is an instance of single_channel_dtype

        for channel_data in images:
            new_single_channel_instance = self.single_channel_dtype(channel_data, self.is4pi)
            self.channels.append(new_single_channel_instance)

        self.min_border_dist = None # needed for get_min_border_dist()
        self.numofchannel = len(self.channels)
        self.shiftxy = None
        return

    def find_rois(self, roi_size, gaus_sigma, min_border_dist, max_threshold, max_kernel, FOV=None,min_center_dist=None,max_bead_number=None):
        """
        Cuts out rois around local maxima in all channels seperately.
        Just calls the 'find_rois' function for each channel.
        """
        self.min_border_dist = min_border_dist # needed for get_min_border_dist()

        for channel in self.channels:
            channel.find_rois(roi_size, gaus_sigma, min_border_dist, max_threshold, max_kernel, FOV, min_center_dist,max_bead_number)
        
        self.rois_available = True

        return
    

    def cut_new_rois(self, channel, centers, file_idxs, roi_shape=None, min_border_dist=None):
        """
        Cuts new rois from images with specified centers in specified channel.
        Just calls 'cut_new_rois' function of specified channel.
        """
        self.channels[channel].cut_new_rois(centers, file_idxs, roi_shape, min_border_dist)
        self.rois_available = True

        return

    def get_image_data(self):
        """
        Provides the necessary image information (e.g., rois, centers, ...) for the psf class
        and the fitter class. Just calls 'get_image_data' function of each channel and appends
        the results to list.
        """
        if self.rois_available:
            results = []
            for channel in self.channels:
                results.append(channel.get_image_data())

            return map(list, zip(*results)) # a way to tranpose a list of iterateables
                                            # needed to correct the order of the resuts without inferring how the results look like
                                            # see: https://stackoverflow.com/questions/6473679/transpose-list-of-lists
        else:
            raise RuntimeError("Can't call 'get_image_data()' since 'rois_available' flag is False.\nThis is probably due to the fact that you did not call 'find_rois()' before using this ImageData.")

    def get_channel(self, channel):
        """
        Returns the object holding the data for the channel with index 'channel'.
        """
        return self.channels[channel]

    def get_min_border_dist(self):
        """
        Returns the min_border_dist parameter from the find_rois() function.
        """
        return self.min_border_dist



    def pair_coordinates(self,delete_id=None):
        _, _, centers, file_idxs = self.get_image_data()
        mask = np.ones(centers[0].shape[0])
        if delete_id is not None:
            mask[delete_id]=0
        mask = mask==1        
        ref_pos = centers[0][mask,:]
        ref_fid = file_idxs[0][mask]
        pair_pos = [None]*self.numofchannel
        pair_file_id = [None]*self.numofchannel
        for i in range(0,self.numofchannel):
            tar_pos = centers[i]
            tar_fid = file_idxs[i]
            pairs_tar_pos_id = []
            pairs_ref_pos_id = []
            for ref_pos_idx in range(ref_pos.shape[0]):
                same_file_idxs = np.where(tar_fid == ref_fid[ref_pos_idx])[0]
                # only allow pairs when they are from same file and not already paired
                available = [i for i in same_file_idxs if i not in pairs_tar_pos_id]
                if not available:
                    continue
                tar_posi = tar_pos[available]
                ref_posi = ref_pos[ref_pos_idx]
                if self.shiftxy is None:
                    distances = np.sqrt(np.sum(np.square(tar_posi - ref_posi), axis=1))
                else:
                    distances = np.sqrt(np.sum(np.square(tar_posi-self.shiftxy[i] - ref_posi), axis=1))

                min_idx = np.argmin(distances)
                # TODO: is it necessary to add an additional hyperparameter for this
                if distances[min_idx] <= 5.:
                    pairs_tar_pos_id.append(available[min_idx])
                    pairs_ref_pos_id.append(ref_pos_idx)

            ref_fid = ref_fid[pairs_ref_pos_id]
            ref_pos = ref_pos[pairs_ref_pos_id]
            pair_pos[i] = tar_pos[pairs_tar_pos_id]
            pair_file_id[i] = tar_fid[pairs_tar_pos_id]
            for j in range(0,i):
                pair_pos[j] = pair_pos[j][pairs_ref_pos_id]
                pair_file_id[j] = pair_file_id[j][pairs_ref_pos_id]

        for i in range(0,self.numofchannel):
            self.cut_new_rois(i, pair_pos[i], pair_file_id[i])

    def process(self,roi_size, gaus_sigma, min_border_dist, max_threshold, max_kernel,pixelsize_x,pixelsize_z,bead_radius, 
                min_center_dist=None,FOV=None, modulation_period=None, padPSF=True, plot=True,pixelsize_y=None, isVolume = True,skew_const=None, max_bead_number=None):

        self.find_rois(roi_size, gaus_sigma, min_border_dist, max_threshold, max_kernel, FOV,min_center_dist, max_bead_number)
        _, rois, _, _ = self.get_image_data()
        for i in range(len(rois)):
            print(f"rois shape channel {i}: {rois[i].shape}")

        # find channel shift
        self.find_channel_shift_cor(plot=False)
        
        _, _, centers, _ = self.get_image_data()
        self.centers_all = centers
        # pair coordinates
        self.pair_coordinates()
        _, rois, centers, _ = self.get_image_data()
        cor0 = centers[0]
        pv = self.shiftxy[1:]
        if plot:
            for i, cor1 in enumerate(centers[1:]):
                plt.figure(figsize=[6,6])
                plt.plot(cor0[:,1],cor0[:,0],'o',markersize = 8,markerfacecolor='none')
                plt.plot(cor1[:,1]-pv[i,1],cor1[:,0]-pv[i,0],'x')
                plt.show()

        for i in range(len(rois)):
            print(f"rois shape channel {i}: {rois[i].shape}")

        self.pixelsize_z = pixelsize_z
        self.pixelsize_x = pixelsize_x
        self.bead_radius = bead_radius
        offset = np.min((np.quantile(rois,1e-3),0))
        for i in range(len(self.channels)):    
            self.channels[i].rois = rois[i]-offset

        # pad rois along z dimension
        _, rois, _, _ = self.get_image_data()
        if padPSF:
            rois = np.stack(rois)
            value = np.empty((), dtype=object)
            value[()] = (0, 0)
            padsize = np.full((len(rois.shape), ), value, dtype=object)
            padsize[-3] = (np.int(bead_radius//pixelsize_z),np.int(bead_radius//pixelsize_z))
            roisL = np.pad(rois,tuple(padsize),mode='edge')
            for i in range(len(self.channels)):    
                self.channels[i].rois = roisL[i]
            print(f"roisL shape channel {i}: {roisL.shape}")
        # generate bead kernel
        if pixelsize_y is None:
            pixelsize_y = pixelsize_x
        self.pixelsize_y = pixelsize_y

        if modulation_period is not None:
            for channel in self.channels:
                channel.zT = modulation_period/pixelsize_z

        for channel in self.channels:
            channel.pixelsize_x = pixelsize_x
            channel.pixelsize_y = pixelsize_y
            channel.pixelsize_z = pixelsize_z
            channel.bead_radius = bead_radius
        return

    def find_channel_shift_img(self):
        imgs, _, centers, _ = self.get_image_data()
        img0 = np.sum(np.max(imgs[0],axis = 1),axis=0)  
        shiftxy = []
        for img in imgs:
      
            img1 = np.sum(np.max(img,axis = 1),axis=0)
            cor_img_ft = np.fft.fft2(img0) * np.conj(np.fft.fft2(img1))
            cor_img_ft = sp.ndimage.fourier_gaussian(cor_img_ft, sigma=2.)
            cor_img =  np.real(np.fft.fftshift(np.fft.ifft2(cor_img_ft)))

            # find max and calculate dx, dy
            # TODO: is argmax okay or is there a better suited way to find maximum like some gaussian fitting?
            dy, dx = np.unravel_index(np.argmax(cor_img), shape=cor_img.shape)
            dy = (cor_img.shape[0]-1)/2 - dy
            dx = (cor_img.shape[1]-1)/2 - dx
            shiftxy.append([dy,dx])
        
        self.shiftxy = np.float32(shiftxy)
        return 

    def find_channel_shift_cor(self,plot=True):
        _, _, centers, _ = self.get_image_data()
        cor0 = centers[0]
        shiftxy = []
        for cor1 in centers:
            pv = (np.mean(cor1,axis=0)-np.mean(cor0,axis=0))
            N1 = cor1.shape[0]
            for k in range(0,5):
                dist = np.sqrt(np.sum((cor1.reshape((N1,1,2))-cor0-pv)**2,axis=-1))
                q = 1/(dist+1e-3)
                ind1,ind0 = np.where(q>=np.quantile(q.flatten(),0.975))
                pv = np.mean(cor1[ind1],axis=0)-np.mean(cor0[ind0],axis=0)
            if plot:
                plt.figure(figsize=[6,6])
                plt.plot(cor0[ind0,0],cor0[ind0,1],'o',markersize = 8,markerfacecolor='none')
                plt.plot(cor1[ind1,0]-pv[0],cor1[ind1,1]-pv[1],'x')
                plt.show()

            shiftxy.append(pv)

        self.shiftxy = np.float32(shiftxy)
        return