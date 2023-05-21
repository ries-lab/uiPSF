from ast import Raise
import numpy as np
from numpy.core.fromnumeric import var
import scipy as sp
import tensorflow as tf
#from tensorflow.keras import optimizers
import matplotlib.pyplot as plt

from psflearning.learning.psfs.PSFVolumeBased4pi_file import PSFVolumeBased4pi
from .FitterInterface_file import FitterInterface
from ..data_representation.PreprocessedImageDataSingleChannel_file import PreprocessedImageDataSingleChannel
from ..psfs.PSFInterface_file import PSFInterface
from ..optimizers import OptimizerABC
from ..loclib import localizationlib
from tqdm import tqdm


class Fitter(FitterInterface):
    """
    This class combines data, psf, optimizer and loss function and defines the actual learning process.
    """
    def __init__(self, data: PreprocessedImageDataSingleChannel, psf: PSFInterface, optimizer: OptimizerABC, loss_func, loss_func_single=None,loss_weight=None) -> None:
        self.data = data
        self.psf = psf
        self.loss_func = loss_func
        self.loss_func_single = loss_func_single
        self.optimizer = optimizer

        self.rois = None
        self.forward_images = None # just to have access to forward images from outside
        self.loc_FD= None
        self.mu = 1
        self.rate = 1.1
        self.loss_weight = loss_weight
        return

    def objective(self, variables,mu, ind = None):
        """
        Define sthe objective that should be optimized.
        Basically asks the psf to calculate forward_images and combines those with the data
        and the loss function to calculate the loss.
        """
        if ind is None:
            ind = [0,variables[0].shape[0]]
        #self.mu *= self.rate
        self.psf.ind = ind
        forward_images = self.psf.calc_forward_images(variables)
        if self.loss_func_single:
            loss = self.loss_func(forward_images, self.rois[:,ind[0]:ind[1]], self.loss_func_single,variables,mu,self.loss_weight)
        else:
            loss = self.loss_func(forward_images, self.rois[ind[0]:ind[1]], variables,mu,self.loss_weight)
        return loss

    def learn_psf(self, variables=None,start_time=None):
        """
        Defines the procedure of the psf learning. Just asks the psf to calculate initial
        values (if not provided), runs the optimization and uses the psf object to do postprocessing.
        """
        # calulate initial values if none are given
        #self.mu = 1
        if variables is None:
            variables, start_time = self.psf.calc_initials(self.data,start_time = start_time)

        # get real rois as target for loss calculation
        # do this after calc_init since rois can be changed there (e.g. in multiChannelPSF)
        _, rois, _, _ = self.data.get_image_data()

        # at this point rois could still be a list (e.g. in multiChannelPSF)
        # therefore make sure they are array
        # if they already are np.stack has no effect
        try:
            self.rois = np.stack(rois)
        except ValueError:
            raise RuntimeError("At this point each channel must have same number of rois and allow np.stack.")

        # run optimization itself
        #self.optimizer.weight = self.psf.weight
        pbar = tqdm(total=self.optimizer.maxiter+50,desc='3/6: learning',bar_format = "{desc}: {n_fmt}/{total_fmt} [{elapsed}s] {rate_fmt}, {postfix[0]}{postfix[2][loss]:>4.5f}, {postfix[1]}{postfix[2][time]:>4.2f}s",postfix=["current loss: ","total time: ", dict(loss=0,time=start_time)])
        
        variables = self.optimizer.minimize(self.objective, variables,self.psf.varinfo,pbar)
        toc = pbar.postfix[-1]['time']
        pbar.close()
        # save final state of forward images to access from easily from outside
        ind = [0,variables[0].shape[0]]
        self.psf.ind = ind
        self.forward_images = self.psf.calc_forward_images(variables).numpy()

        # run postprocessing
        variables = self.psf.postprocess(variables)

        return variables, toc

    def relearn(self,initres, channeltype, threshold,start_time=None):
        metric = self.reject_metric
        mask = metric[0]>0
        for i,val in enumerate(metric):
            mask = (val<threshold[i]) & mask
        mask = (self.minI>0) & mask
        #mask = mask & (1/metric[-1]<threshold[-1])
        delete_id = np.where(~mask)
        print('outlier id:',str(delete_id[0]))
        
        if (delete_id[0].size>0) & (delete_id[0].size<mask.size):
            if channeltype=='single':     
                _, rois, centers, file_idxs = self.data.get_image_data()
                cor = centers[mask,:]
                fid = file_idxs[mask]
                self.data.rois = rois[mask]
                self.data.centers = cor
                self.data.file_idxs = fid
                _, rois, _, _ = self.data.get_image_data()
                print(f"rois shape channel : {rois.shape}")
                var=initres[-1]
                var[0] = initres[-1][0][mask] # pos
                var[1] = initres[-1][1][mask] # bg
                var[2] = initres[-1][2][mask] # intensity
                var[-1] = initres[-1][-1][mask] # drift                
                res, toc = self.learn_psf(var,start_time=start_time)
            else:
                _, rois, centers, file_idxs = self.data.get_image_data() 
                for i in range(len(self.data.channels)):    
                    self.data.channels[i].rois = rois[i][mask]
                    self.data.channels[i].centers = centers[i][mask,:]
                    self.data.channels[i].file_idxs = file_idxs[i][mask]
                _, rois, centers, _ = self.data.get_image_data()
                num_channels = len(rois)
        
                cor_ref = np.concatenate((centers[0], np.ones((centers[0].shape[0], 1))), axis=1)
                self.psf.cor_ref_channel = np.stack([cor_ref] * (num_channels-1)).astype(np.float32)        
                self.psf.cor_other_channels = np.stack(centers[1:]).astype(np.float32)
                for i in range(len(rois)):
                    print(f"rois shape channel {i}: {rois[i].shape}")
                var=initres[-1]
                var[0] = initres[-1][0][mask] # pos
                var[1] = initres[-1][1][:,mask] # bg
                var[2] = initres[-1][2][:,mask] # intensity
                var[-2] = initres[-1][-2][:,mask] # drift
                if channeltype == '4pi':
                    var[3] = initres[-1][3][:,mask]# intensity (imag)
                    if self.psf.psftype != PSFVolumeBased4pi:
                        var[-4] = initres[-1][-4][:,mask] # pos_d

                res, toc = self.learn_psf(var,start_time=start_time)
        else:
            res = initres
            toc = start_time
        return res, toc
    
    def relearn_smlm(self,initres,channeltype, threshold,start_time=None):
        pos = initres[-1][0]
        intensity = np.squeeze(initres[-1][2])
        #cor = self.data.centers
        xp = pos[:,-1]
        yp = pos[:,-2]
        zp = pos[:,0]
        psf_data = self.rois
        psf_fit = self.forward_images
        mydiff = psf_fit-psf_data
        mse1 = np.mean(np.square(mydiff), axis = (-2,-1))/np.mean(psf_data, axis = (-2,-1))
        if channeltype == 'multi' or channeltype == '4pi':
            intensity = np.min(intensity,axis=0,keepdims=False)
            mse1 = np.sum(mse1,axis=0,keepdims=False)
        a = threshold[0]
        if self.psf.options.insitu.backgroundROI:
            mask = (xp>np.quantile(xp,1-a)) & (xp<np.quantile(xp,a)) & (yp>np.quantile(yp,1-a)) & (yp<np.quantile(yp,a)) 
        else:
            mask = (xp>np.quantile(xp,1-a)) & (xp<np.quantile(xp,a)) & (yp>np.quantile(yp,1-a)) & (yp<np.quantile(yp,a)) & (zp>np.quantile(zp,1-a)) & (zp<np.quantile(zp,a))

        mask = mask & (mse1<np.quantile(mse1,threshold[1]))
        mask = mask & (intensity>0)
        delete_id = np.where(~mask)
        print('outlier percentage:',1-np.sum(mask)/mask.size)
        
        if (delete_id[0].size>0) & (delete_id[0].size<mask.size):
            if channeltype=='single':     
                _, rois, centers, frames = self.data.get_image_data()
                cor = centers[mask,:]
                fid = frames[mask]
                self.data.rois = rois[mask]
                self.data.centers = cor
                self.data.frames = fid
                _, rois, _, _ = self.data.get_image_data()
                print(f"rois shape channel : {rois.shape}")
                var=initres[-1]
                var[0] = initres[-1][0][mask] # pos
                var[1] = initres[-1][1][mask] # bg
                var[2] = initres[-1][2][mask] # intensity
                self.psf.zweight = self.psf.zweight[mask]
                res,toc = self.learn_psf(var,start_time=start_time)
            else:
                _, rois, centers, frames = self.data.get_image_data() 
                for i in range(len(self.data.channels)):    
                    self.data.channels[i].rois = rois[i][mask]
                    self.data.channels[i].centers = centers[i][mask,:]
                    self.data.channels[i].frames = frames[i][mask]
                _, rois, centers, _ = self.data.get_image_data()
                num_channels = len(rois)
        
                cor_ref = np.concatenate((centers[0], np.ones((centers[0].shape[0], 1))), axis=1)
                self.psf.cor_ref_channel = np.stack([cor_ref] * (num_channels-1)).astype(np.float32)        
                self.psf.cor_other_channels = np.stack(centers[1:]).astype(np.float32)

                for i in range(len(rois)):
                    print(f"rois shape channel {i}: {rois[i].shape}")
                var=initres[-1]
                var[0] = initres[-1][0][mask] # pos
                var[1] = initres[-1][1][:,mask] # bg
                var[2] = initres[-1][2][:,mask] # intensity
                if channeltype == '4pi':
                    var[3] = initres[-1][3][:,mask]# intensity (phase)

                res,toc = self.learn_psf(var,start_time=start_time)
        else:
            res = initres
            toc = start_time
        return res, toc

    def localize(self,res,channeltype,usecuda=True,initz=None, plot=True,start_time=None):
        intensity = np.abs(np.squeeze(res[2],axis=(-1,-2)))
        if res[2].dtype == 'complex64':
            intensityR = intensity
        else:
            intensityR = np.real(np.squeeze(res[2],axis=(-1,-2)))
        I_model = res[3]
        psf_fit = self.forward_images
        psf_data = self.rois
        pz = self.data.pixelsize_z

        dll = localizationlib(usecuda=usecuda)
        if channeltype=='single':
            locres = dll.loc_ast(psf_data,I_model,pz,initz=initz,plot=plot,start_time=start_time)
            mydiff = psf_fit[:,1:-1]-psf_data[:,1:-1]
            mse1 = np.mean(np.square(mydiff), axis = (-3,-2,-1))/np.mean(psf_data, axis = (-3,-2,-1))

        elif channeltype=='multi':
            _, _, centers, _ = self.data.get_image_data()
            cor = np.stack(centers)
            imgcenter = self.psf.imgcenter
            T = res[-2]
            locres = dll.loc_ast_dual(psf_data,I_model,pz,cor,imgcenter,T,initz=initz,plot=plot,start_time=start_time)
            mydiff = psf_fit[:,:,1:-1]-psf_data[:,:,1:-1]
            mse1 = np.mean(np.mean(np.square(mydiff), axis = (-3,-2,-1))/np.mean(psf_data, axis = (-3,-2,-1)),axis=0)

        elif channeltype=='4pi':
            _, _, centers, _ = self.data.get_image_data()
            A_model = res[4]
            cor = np.stack(centers)
            imgcenter = self.psf.imgcenter
            T = np.squeeze(res[-2])
            zT = np.array([self.psf.sub_psfs[0].zT])
            locres = dll.loc_4pi(psf_data,I_model,A_model,pz,cor,imgcenter,T,zT,initz=initz,plot=plot,start_time=start_time)
            mydiff = psf_fit[:,:,:,1:-1]-psf_data[:,:,:,1:-1]
            mse1 = np.mean(np.mean(np.square(mydiff), axis = (-4,-3,-2,-1))/np.mean(psf_data, axis = (-4,-3,-2,-1)),axis=0)
        else:
            raise TypeError('supported channeltype is:',str(['single','multi','4pi']))

        if channeltype=='single':
            if len(intensity.shape)<2:
                avgI = intensity      
                minI = intensityR      
            else:
                avgI = np.median(intensity,axis=1)    
                minI = np.min(intensityR,axis=1)
        else:
            if len(intensity.shape)<3:
                avgI = intensity[0]
                minI = intensityR[0]
            else:
                avgI = np.median(intensity[0],axis=1)
                minI = np.min(intensityR[0],axis=1)

        if psf_data.shape[0]==1:
            intRatio = np.array([1.0])
            mseRatio = np.array([1.0])
        else:
            intRatio = np.square(avgI-np.median(avgI))/np.median(avgI)/avgI
            mseRatio = mse1/np.median(mse1)
        msezRatio = locres[4]        
        metric = [msezRatio,mseRatio,intRatio]
        label = ['relative MSE in z','relative MSE']
        if plot & (mseRatio.size>1):
            fig = plt.figure(figsize=[8,8])
            for i,val in enumerate(metric[:-1]):
                plt.plot(intRatio,val,'.')
            plt.xlabel('relative intensity')
            plt.ylabel('relative MSE')
            plt.xlim([0,3])
            plt.ylim([0,5])
            plt.legend(label)
            plt.grid(True)
            plt.show()
        self.reject_metric = metric
        self.minI = minI
        return locres

    def localize_smlm(self,res,channeltype,usecuda=True,initz=None, plot=True):

        I_model = res[3]

        psf_data = self.rois
        pz = self.data.pixelsize_z

        dll = localizationlib(usecuda=usecuda)
        if channeltype=='single':
            locres = dll.loc_ast(psf_data,I_model,pz,initz=initz,plot=plot)
 
        elif channeltype=='multi':
            _, _, centers, _ = self.data.get_image_data()
            cor = np.stack(centers)
            imgcenter = self.psf.imgcenter
            T = res[-2]
            locres = dll.loc_ast_dual(psf_data,I_model,pz,cor,imgcenter,T,initz=initz,plot=plot)

        elif channeltype=='4pi':
            _, _, centers, _ = self.data.get_image_data()
            A_model = res[4]
            cor = np.stack(centers)
            imgcenter = self.psf.imgcenter
            T = np.squeeze(res[-2])
            zT = np.array([self.data.channels[0].zT])
            locres = dll.loc_4pi(psf_data,I_model,A_model,pz,cor,imgcenter,T,zT,initz=initz,plot=plot)

        else:
            raise TypeError('supported psftype is:',str(['single','multi','4pi']))

        return locres

    
    def localize_FD(self,res, channeltype,usecuda=True,initz=None, plot=True):
        #res_dict = self.psf.res2dict(res)
        #I_model_all = res_dict['I_model_all']
        I_model_all = self.forward_images
        psf_data = self.rois
        pz = self.data.pixelsize_z
        if len(psf_data.shape)>3:
            Nz = psf_data.shape[-3]
        else:
            Nz = 1
        _, _, centers, _ = self.data.get_image_data()
        cor = np.stack(centers)
        dll = localizationlib(usecuda=usecuda)
        x = []
        y = []
        z = []
        for i in range(psf_data.shape[-4]):
            if channeltype=='single':
                loci = dll.loc_ast(psf_data[i],I_model_all[i],pz,initz=initz,start_time=0)
            elif channeltype=='multi':

                imgcenter = self.psf.imgcenter
                T = res[-2]
                loci = dll.loc_ast_dual(psf_data[:,i:i+1],I_model_all[:,i],pz,cor[:,i:i+1],imgcenter,T,initz=initz,start_time=0)

            x.append(np.squeeze(loci[-1]['x']))
            y.append(np.squeeze(loci[-1]['y']))
            z.append(np.squeeze(loci[-1]['z']))

        xf = np.stack(x)
        yf = np.stack(y)
        zf = np.stack(z)

        zg = np.linspace(0,Nz-1,Nz)
        if Nz>1:
            zf = zf-np.median(zf-zg,axis=1,keepdims=True)
            zdiff = zf-zg        
            xf = xf-np.median(xf,axis=1,keepdims=True)        
            yf = yf-np.median(yf,axis=1,keepdims=True)
            if Nz>4:
                zind = range(2,Nz-2,1)
            else:
                zind = range(0,Nz,1)
        
            zdiff = zdiff-np.mean(zdiff[:,zind],axis=1,keepdims=True)            
        else:
            zdiff = zf
        if plot & (Nz>1):
            fig = plt.figure(figsize=[12,6])
            ax = fig.add_subplot(1,2,1)
            plt.plot(zf.transpose(),color=(0.6,0.6,0.6))
            plt.plot(zg)
            ax = fig.add_subplot(1,2,2)
            plt.plot((zdiff).transpose(),'k',alpha=0.1)
            plt.plot(np.median(zdiff,axis=0),color='r')
            plt.plot(zg-zg,color='k')
            ax.set_ylabel('z bias')
            ax.set_ylim([-0.1,0.1]/np.array(pz))

        
        loc_FD = dict(x=xf,y=yf,z=zf)
        return loc_FD
