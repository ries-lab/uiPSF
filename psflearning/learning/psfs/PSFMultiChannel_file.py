
from typing import Type

import numpy as np
import scipy as sp
import tensorflow as tf

from .PSFInterface_file import PSFInterface
from ..data_representation.PreprocessedImageDataInterface_file import PreprocessedImageDataInterface
from ..fitters.Fitter_file import Fitter
from ..optimizers import OptimizerABC, L_BFGS_B

class PSFMultiChannel(PSFInterface):
    def __init__(self, psftype: Type[PSFInterface], init_optimizer: OptimizerABC=None, options = None,loss_weight=None) -> None:
        self.parameters = None
        self.updateflag = None        
        self.psftype = psftype
        self.PSFtype = 'scalar'
        self.sub_psfs = [] # each element is an instance of psftype
        self.data = None
        self.weight = None
        self.loss_weight = loss_weight
        self.options = options
        self.init_trafos = None
        if init_optimizer is None:
            self.init_optimizer = L_BFGS_B(100)
        else:
            self.init_optimizer = init_optimizer

    def calc_initials(self, data: PreprocessedImageDataInterface, start_time=None):
        """
        Provides initial values for the optimizable varibales for the fitter class.
        Since this is a multi-channel PSF, it performs an initial fitting for each
        channel first and then calculates an initial guess for the transformations.
        """
        
        self.data = data
        images, rois, centers, file_idxs = self.data.get_image_data()
        num_channels = len(images)
        self.sub_psfs = [None]*num_channels
        self.imgcenter = np.hstack((np.array(images[0].shape[-2:])/2,0)).astype(np.float32)
        # choose first channel as reference and run first round of optimization
        ref_psf = self.psftype(options = self.options)
        ref_psf.psftype = self.PSFtype
        if hasattr(self,'initpsf'):
            ref_psf.initpsf = self.initpsf[0]
        ref_psf.defocus = np.float32(self.options.multi.defocus[0]/self.data.pixelsize_z)
        self.sub_psfs[0] = ref_psf
        fitter_ref_channel = Fitter(self.data.get_channel(0), ref_psf,self.init_optimizer, ref_psf.default_loss_func,loss_weight=self.loss_weight) # TODO: redesign multiData        
        res_ref, toc = fitter_ref_channel.learn_psf(start_time=start_time)
        ref_pos = res_ref[0]        
        ref_pos_yx1 = np.concatenate((ref_pos[:, 1:], np.ones((ref_pos.shape[0], 1))), axis=1)
        self.ref_pos_yx = ref_pos_yx1
   
        # create empty initial guess lists
        # and fill in values for first channel
        

        init_trafos = []
        #ref_zpos = np.transpose(np.expand_dims(-ref_pos[:,0]+self.data.channels[0].rois.shape[-3]//2,axis=0))       
        #init_subpixel_pos_ref_channel = np.concatenate((ref_zpos, centers[0]-ref_pos[:, 1:]), axis=1)
  
        init_params = [res_ref[-1]]
        # do everything for the other channels and put initial values in corresponding array
        for i in range(1, num_channels):
            # run first round of optimization
            current_psf = self.psftype(options = self.options)
            current_psf.psftype = self.PSFtype
            if hasattr(self,'initpsf'):
                current_psf.initpsf = self.initpsf[i]
            current_psf.defocus = np.float32(self.options.multi.defocus[i]/self.data.pixelsize_z)
            self.sub_psfs[i] = current_psf
            fitter_current_channel = Fitter(self.data.get_channel(i), current_psf, self.init_optimizer,current_psf.default_loss_func,loss_weight=self.loss_weight)
            res_cur,toc = fitter_current_channel.learn_psf(start_time=toc)
            current_pos = res_cur[0]
            # calculate transformation
            current_pos_yx1 = np.concatenate((current_pos[:, 1:], np.ones((current_pos.shape[0], 1))), axis=1) 
            current_trafo = np.linalg.lstsq(ref_pos_yx1-self.imgcenter, current_pos_yx1-self.imgcenter, rcond=None)[0]

            #relative_shift = np.mean(centers[0],axis=0)-self.imgcenter[:-1]
            #current_trafo[-1][:-1] = self.data.shiftxy[i]-(np.matmul(relative_shift,current_trafo[:-1,:-1])-relative_shift)         
            # fill initial arrays
            self.sub_psfs[i].weight = self.sub_psfs[0].weight
            init_params.append(res_cur[-1])
            init_trafos.append(current_trafo)


        # get current status of image data
        images, _, centers, _ = self.data.get_image_data()
        num_channels = len(images)

        # stack centers of ref channel num_channels-1 times for easier calc in calc_forward_images
        cor_ref = np.concatenate((centers[0][:,-2:], np.ones((centers[0].shape[0], 1))), axis=1)
        self.cor_ref_channel = np.stack([cor_ref] * (num_channels-1)).astype(np.float32)
        # self.pos_ref_channel_yx1 = np.stack([ref_pos_yx1] * (num_channels-1)).astype(np.float32)
        # centers of other channels needed to calculate diffs in objective
        self.cor_other_channels = (np.stack(centers[1:])[...,-2:]).astype(np.float32)
           
        self.init_trafos = np.stack(init_trafos).astype(np.float32)

                    
        param = map(list, zip(*init_params)) # a way to tranpose the first two dimensions of a list of iterateables
        param = [np.stack(var) for var in param]
        param[0] = param[0][0]

        #param.insert(0,init_subpixel_pos_ref_channel.astype(np.float32))
        param.append(self.init_trafos)
        self.weight = np.ones((len(param)))
        self.weight[-1] = 1e-3
        if hasattr(self.sub_psfs[0],'pos_weight'):
            self.weight[0] = self.sub_psfs[0].pos_weight
        param[-1] = param[-1]/self.weight[-1]
        self.varinfo = self.sub_psfs[0].varinfo
        for k, vinfo in enumerate(self.varinfo[1:]):
            if vinfo['type'] == 'Nfit':
                self.varinfo[k+1]['id'] += 1
        self.varinfo.append(dict(type='shared'))
        return param, toc


    def calc_forward_images(self, variables):
        """
        Calculate forward images from the current guess of the variables.
        """        
        init_pos_ref = variables[0]*self.weight[0]
        trafos = variables[-1]*self.weight[-1]

        # calc positions from pos in ref channel and trafos
        positions = self.calc_positions_from_trafos(init_pos_ref, trafos)
       
        # use calc_forward_images of every sub_psf and stack at the end
        forward_images = [None] * len(self.sub_psfs) # needed since tf.function does not support .append()
        for i, sub_psf in enumerate(self.sub_psfs):
            pos = positions[i]/self.weight[0]       
            #link pos, intensity, phase
            #sub_variables = [pos, variables[1][i], variables[2][0], variables[3][i],variables[4][0]]
            sub_variables = [pos, variables[1][i], variables[2][0]]
            for k in range(3,len(variables)-2):
                sub_variables.append(variables[k][i])
            sub_variables.append(variables[-2][0])
            forward_images[i] = sub_psf.calc_forward_images(sub_variables)

        return tf.stack(forward_images)

    def calc_positions_from_trafos(self, init_subpixel_pos_ref_channel, trafos):
        # calculate positions from position in ref channel and transformation
        
        cor_target = tf.linalg.matmul(self.cor_ref_channel[:,self.ind[0]:self.ind[1]]-self.imgcenter, trafos)[..., :-1]

        diffs = tf.math.subtract(self.cor_other_channels[:,self.ind[0]:self.ind[1]]-self.imgcenter[:-1],cor_target)
        pos_other_channels = init_subpixel_pos_ref_channel + tf.concat((tf.zeros(diffs.shape[:-1] + (1,)), diffs), axis=2)
        positions = tf.concat((tf.expand_dims(init_subpixel_pos_ref_channel, axis=0), pos_other_channels), axis=0)

        return positions

    def postprocess(self, variables):
        """
        Applies postprocessing to the optimized variables. In this case calculates
        real positions in the image from the positions in the roi. Also, normalizes
        psf and adapts intensities and background accordingly.
        """
        res = variables.copy()
        res[-1] = variables[-1]*self.weight[-1]
        res[2] = variables[2]
        init_subpixel_pos_ref_channel = res[0]*self.weight[0]
        trafos = res[-1]
        self.ind = [0,res[0].shape[0]]
        # calc positions from pos in ref channel and trafos
        positions = self.calc_positions_from_trafos(init_subpixel_pos_ref_channel, trafos)
        
        # calc_positions_from_trafos is implemented using tf,
        # therefore convert to numpy here
        positions = positions.numpy()
        
        # just call postprocess of every sub_psf and stack results at the end
        results = []
        for i, sub_psf in enumerate(self.sub_psfs):
            sub_variables = [positions[i]/self.weight[0]]
            for k in range(1,len(variables)-1):
                sub_variables.append(res[k][i])
            results.append(sub_psf.postprocess(sub_variables)[:-1])
        
        results = map(list, zip(*results)) # a way to tranpose the first two dimensions of a list of iterateables
        results = [np.stack(variable) for variable in results]
        for k in range(-1,0):
            results.append(res[k])
        results.append(variables)
        return results


    def res2dict(self,res):
        res_dict = dict()
        for i,sub_psf in enumerate(self.sub_psfs):
            sub_res = []
            for k in range(0,len(res)-2):
                sub_res.append(res[k][i])
            res_dict['channel'+str(i)]=sub_psf.res2dict(sub_res)
        res_dict['T'] = np.squeeze(res[-2])
        res_dict['imgcenter'] = self.imgcenter
        res_dict['xyshift'] = self.data.shiftxy



        return res_dict
