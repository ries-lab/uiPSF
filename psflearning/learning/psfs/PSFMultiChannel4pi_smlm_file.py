
from typing import Type

import numpy as np
from requests import options
import scipy as sp
import tensorflow as tf
from psflearning.learning.psfs.PSFPupilBased4pi_file import PSFPupilBased4pi

from psflearning.learning.psfs.PSFVolumeBased4pi_file import PSFVolumeBased4pi
from psflearning.learning.psfs.PSFZernikeBased4pi_file import PSFZernikeBased4pi

from .PSFInterface_file import PSFInterface
from ..data_representation.PreprocessedImageDataInterface_file import PreprocessedImageDataInterface
from ..fitters.Fitter_file import Fitter
from ..optimizers import OptimizerABC, L_BFGS_B
from ..loclib import localizationlib

class PSFMultiChannel4pi_smlm(PSFInterface):
    def __init__(self, psftype: Type[PSFInterface], init_optimizer: OptimizerABC=None,options = None, loss_weight=None) -> None:
        self.parameters = None
        self.updateflag = None        
        self.psftype = psftype
        self.sub_psfs = [] # each element is an instance of psftype
        self.data = None
        self.weight = None
        self.loss_weight = loss_weight
        self.options = options
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
        options = self.options.copy()
        options.insitu.partition_data = False
        partition_data = self.options.insitu.partition_data

        ref_psf = self.psftype(options=options)
        if hasattr(self,'initpsf'):
            ref_psf.initpsf = self.initpsf[0]
            ref_psf.initA = self.initA[0]
        ref_psf.dphase = 0.0
        self.sub_psfs[0] = ref_psf

        #fitter_ref_channel = Fitter(self.data.get_channel(0), ref_psf,self.init_optimizer, ref_psf.default_loss_func,loss_weight=self.loss_weight) # TODO: redesign multiData        
        
        #res_ref, toc = fitter_ref_channel.learn_psf(start_time=start_time)
        ref_var, start_time = ref_psf.calc_initials(self.data.get_channel(0),start_time = start_time)

        #ref_pos = res_ref[0]        
        #ref_pos_yx1 = np.concatenate((ref_pos[:, 1:], np.ones((ref_pos.shape[0], 1))), axis=1)
        #self.ref_pos_yx = ref_pos_yx1
   
        # create empty initial guess lists
        # and fill in values for first channel

        
        #ref_zpos = np.transpose(np.expand_dims(-ref_pos[:,0]+self.data.channels[0].rois.shape[-3]//2,axis=0))       
        
  
        #init_params = [res_ref[-1]]
        #I_init = [res_ref[3]]
        #A_init = [res_ref[4]]
        #target_pos_yx1 = [None]*num_channels

        init_params = [ref_var]
        I_init = [ref_psf.I_model]
        A_init = [ref_psf.A_model]

        # do everything for the other channels and put initial values in corresponding array
        for i in range(1, num_channels):
            # run first round of optimization
            current_psf = self.psftype(options=options)
            if hasattr(self,'initpsf'):
                current_psf.initpsf = self.initpsf[i]
                current_psf.initA = self.initA[i]
            current_psf.dphase = -i*np.pi/2
            self.sub_psfs[i] = current_psf

            #fitter_current_channel = Fitter(self.data.get_channel(i), current_psf, self.init_optimizer,current_psf.default_loss_func,loss_weight=self.loss_weight)
            #res_cur, toc = fitter_current_channel.learn_psf(start_time=toc)
            #current_pos = res_cur[0]
            # calculate transformation
            #target_pos_yx1[i] = np.concatenate((current_pos[:, 1:], np.ones((current_pos.shape[0], 1))), axis=1)  
            cur_var, start_time = current_psf.calc_initials(self.data.get_channel(i),start_time = start_time)
            
            self.sub_psfs[i].weight = self.sub_psfs[0].weight
            # fill initial arrays
            init_params.append(cur_var)
            I_init.append(current_psf.I_model)
            A_init.append(current_psf.A_model)

        # pair_id = self.data.pair_coordinates()            

        # init_trafos = []
        # for i in range(1,num_channels):
        #     current_trafo = np.linalg.lstsq(ref_pos_yx1[pair_id[0]]-self.imgcenter, target_pos_yx1[i][pair_id[i]]-self.imgcenter, rcond=None)[0]
        #     init_trafos.append(current_trafo)

        # for j in range(0,num_channels):    
        #     for k in range(0,4):
        #         init_params[j][k] = init_params[j][k][pair_id[j]] # pos

        # get current status of image data
        images, rois, centers, frames = self.data.get_image_data()
        num_channels = len(images)
        dll = localizationlib(usecuda=True)
        cor = np.stack(centers)
        imgcenter = self.imgcenter
        #T = np.stack(init_trafos)
        T = np.eye(3)*np.ones((3,1,1))
        data = np.stack(rois)
        I_init = np.stack(I_init)
        A_init = np.stack(A_init)
        pixelsize_z = np.array(self.data.pixelsize_z)
        zT = np.array([self.sub_psfs[0].data.zT])

        locres = dll.loc_4pi(data,I_init,A_init,pixelsize_z,cor,imgcenter,T,zT,start_time=0, linkxy=False)
        toc = locres[-2]
        LL = locres[2]
        zp = locres[-1]['zast'].flatten()+self.sub_psfs[0].Zoffset
        phip = locres[-1]['z'].flatten()*2*np.pi/zT   
        ccx = rois[0].shape[-1]/2-1
        xp = locres[-1]['x']-ccx
        yp = locres[-1]['y']-ccx

        photon = locres[0][8]  
        bg = locres[0][9]
        a = 0.99
        a1 = options.insitu.min_photon
        mask = (np.prod(xp>np.quantile(xp,1-a),axis=0)==1) & (np.prod(xp<np.quantile(xp,a),axis=0)==1) & (np.prod(yp>np.quantile(yp,1-a),axis=0)==1) & (np.prod(yp<np.quantile(yp,a),axis=0)==1) & (zp>np.quantile(zp,1-a)) & (zp<np.quantile(zp,a))
        mask = mask.flatten() & (LL>np.quantile(LL,0.1)) & (photon>np.quantile(photon,a1))

        for k in range(0,num_channels):
            self.data.channels[k].rois = rois[k][mask]
            self.data.channels[k].centers = centers[k][mask]
            self.data.channels[k].frames = frames[k][mask]


        cor = cor[:,mask,:]
        xp = xp[:,mask]
        yp = yp[:,mask]
        zp = zp[mask]
        phip = phip[mask]
        photon = photon[mask]
        bg = bg[mask]
        LL = LL[mask]

        for j,vars in enumerate(init_params):
            for k,var in enumerate(vars):
                if self.sub_psfs[0].varinfo[k]['type'] == 'Nfit':
                    init_params[j][k] = var[mask]

        for i in range(0,num_channels):
            init_params[i][0][:,0] = zp/self.sub_psfs[0].weight[2]
            init_params[i][3][:,0,0] = phip/self.sub_psfs[0].weight[2]
            init_params[i][1][:,0,0] = bg/self.sub_psfs[0].weight[1]
            init_params[i][2][:,0,0] = photon/self.sub_psfs[0].weight[0]

        fitter_ref_channel = Fitter(self.data.get_channel(0), self.sub_psfs[0],self.init_optimizer, self.sub_psfs[0].default_loss_func,loss_weight=self.loss_weight) # TODO: redesign multiData        
        
        res_ref, toc = fitter_ref_channel.learn_psf(variables=init_params[0],start_time=start_time)

        ref_pos = res_ref[0]        
        ref_pos_yx1 = np.concatenate((ref_pos[:, 1:], np.ones((ref_pos.shape[0], 1))), axis=1)

        init_params1 = [res_ref[-1]]
        init_trafos = []
        for i in range(1,num_channels):
            fitter_current_channel = Fitter(self.data.get_channel(i), self.sub_psfs[i], self.init_optimizer,self.sub_psfs[i].default_loss_func,loss_weight=self.loss_weight)
            res_cur, toc = fitter_current_channel.learn_psf(variables=init_params[i],start_time=toc)
            current_pos = res_cur[0]
            # calculate transformation
            target_pos_yx1 = np.concatenate((current_pos[:, 1:], np.ones((current_pos.shape[0], 1))), axis=1)  
            current_trafo = np.linalg.lstsq(ref_pos_yx1-self.imgcenter, target_pos_yx1-self.imgcenter, rcond=None)[0]
            init_trafos.append(current_trafo)
            init_params1.append(res_cur[-1])

        # ref_pos_yx1 = np.vstack((cor[0][:,0]+yp[0],cor[0][:,1]+xp[0], np.ones((xp.shape[1])))).transpose()
        # init_trafos = []
        # for i in range(1,num_channels):
        #      target_pos_yx1 = np.vstack((cor[i][:,0]+yp[i],cor[i][:,1]+xp[i], np.ones((xp.shape[1])))).transpose()
        #      current_trafo = np.linalg.lstsq(ref_pos_yx1-self.imgcenter, target_pos_yx1-self.imgcenter, rcond=None)[0]
        #      init_trafos.append(current_trafo)

        if partition_data:
            initz, rois_id,_ = self.partitiondata(zp,LL)

        _, _, centers, _ = self.data.get_image_data()

        # stack centers of ref channel num_channels-1 times for easier calc in calc_forward_images
        cor_ref = np.concatenate((centers[0], np.ones((centers[0].shape[0], 1))), axis=1)
        self.cor_ref_channel = np.stack([cor_ref] * (num_channels-1)).astype(np.float32)
        # self.pos_ref_channel_yx1 = np.stack([ref_pos_yx1] * (num_channels-1)).astype(np.float32)
        # centers of other channels needed to calculate diffs in objective
        self.cor_other_channels = np.stack(centers[1:]).astype(np.float32)
                
        self.init_trafos = np.stack(init_trafos).astype(np.float32)
        if partition_data:
            for j,vars in enumerate(init_params1):
                for k,var in enumerate(vars):
                    if self.sub_psfs[0].varinfo[k]['type'] == 'Nfit':
                        init_params1[j][k] = var[rois_id]
        

        param = map(list, zip(*init_params1)) # a way to tranpose the first two dimensions of a list of iterateables
        param = [np.stack(var) for var in param]
        param[0] = param[0][0]
        #init_subpixel_pos_ref_channel = np.concatenate((param[0][:,0:1], centers[0]-ref_pos[:, 1:]), axis=1)

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
        if partition_data:
            param[0][:,0] = initz/self.weight[0]
            param[3][0,:,0,0] = phip[rois_id]/self.sub_psfs[0].weight[2]
            for i in range(0,num_channels):
                param[1][i,:,0,0] = bg[rois_id]/self.sub_psfs[0].weight[1]
                param[2][i,:,0,0] = photon[rois_id]/self.sub_psfs[0].weight[0]

        if self.options.fpi.link_zernikecoeff:
            param[4][0]=np.hstack((param[4][0][:,0:1],np.mean(param[4][:,:,1:],axis=0)))
            param[5][0]=np.hstack((param[5][0][:,0:4],np.mean(param[5][:,:,4:],axis=0)))


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
            sub_variables = [pos, variables[1][i], variables[2][0], variables[3][0]]
            if self.options.fpi.link_zernikecoeff:
                sub_variables.append(tf.concat((variables[4][i][:,0:1],variables[4][0][:,1:]),axis=1))
                sub_variables.append(tf.concat((variables[5][i][:,0:4],variables[5][0][:,4:]),axis=1))

                for k in range(6,len(variables)-1):
                    sub_variables.append(variables[k][i])
            else:
                for k in range(4,len(variables)-1):
                    sub_variables.append(variables[k][i])

            forward_images[i] = sub_psf.calc_forward_images(sub_variables)

        return tf.stack(forward_images)

    def calc_positions_from_trafos(self, init_subpixel_pos_ref_channel, trafos):
        # calculate positions from position in ref channel and transformation
        cor_target = tf.linalg.matmul(self.cor_ref_channel[:,self.ind[0]:self.ind[1]]-self.imgcenter, trafos)[..., :-1]

        diffs = tf.math.subtract(self.cor_other_channels[:,self.ind[0]:self.ind[1]]-self.imgcenter[:-1],cor_target)
        pos_other_channels = init_subpixel_pos_ref_channel + tf.concat((tf.zeros(diffs.shape[:-1] + (1,)), diffs), axis=2)
        positions = tf.concat((tf.expand_dims(init_subpixel_pos_ref_channel, axis=0), pos_other_channels), axis=0)

        return positions
    def partitiondata(self,zf,LL):
        _, rois, centers, frames = self.data.get_image_data()
        Nchannel = len(rois)
        Nfit = LL.shape[0]
        nbin = self.options.insitu.partition_size[0]
        count,edge = np.histogram(zf,nbin)
        ind = np.digitize(zf,edge)
        Npsf = self.options.insitu.partition_size[1]
        rois1_avg = []
        rois_id = np.array(range(0,Nfit))
        rois1_id = []
        
        for ii in range(1,nbin+1):
            mask = (ind==ii)
            im1 = rois[0][mask]
            Nslice = np.min((Npsf,np.sum(mask)))            
            #indsample = list(np.random.choice(im1.shape[0],Nslice,replace=False))
            indsample = np.argsort(-LL[mask])[0:Nslice]
            rois1 = im1[indsample]
            rois1_avg.append(np.mean(rois1,axis=0))
            rois1_id.append(rois_id[mask][indsample])
        rois1_avg = np.stack(rois1_avg)
        rois1_id = np.concatenate(rois1_id,axis=0)
        zf1 = zf[rois1_id]
        for k in range(0,Nchannel):
            self.data.channels[k].rois = rois[k][rois1_id]
            self.data.channels[k].centers = centers[k][rois1_id]
            self.data.channels[k].frames = frames[k][rois1_id]
        return zf1,rois1_id, rois1_avg

    def postprocess(self, variables):
        """
        Applies postprocessing to the optimized variables. In this case calculates
        real positions in the image from the positions in the roi. Also, normalizes
        psf and adapts intensities and background accordingly.
        """
        #init_subpixel_pos_ref_channel, backgrounds, intensities, psf_params, trafos = variables
        res = variables.copy()
        res[-1] = variables[-1]*self.weight[-1]
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
            
            sub_variables = [positions[i]/self.weight[0],res[1][i], res[2][0], res[3][0]]
            if self.options.fpi.link_zernikecoeff:
                sub_variables.append(np.hstack((res[4][i][:,0:1],res[4][0][:,1:])))
                sub_variables.append(np.hstack((res[5][i][:,0:4],res[5][0][:,4:])))
                for k in range(6,len(variables)-1):
                    sub_variables.append(res[k][i])
            else:
                for k in range(4,len(variables)-1):
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
