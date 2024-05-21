"""
Copyright (c) 2022      Ries Lab, EMBL, Heidelberg, Germany
All rights reserved     

@author: Sheng Liu
"""
#%%
#from pickle import FALSE
import h5py as h5
#import czifile as czi
import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import gridspec
#from skimage import io
# append the path of the parent directory as long as it's not a real package
#import sys
#import glob
#import scipy.io as sio
import scipy.signal as sig
#import tensorflow as tf
import json
from tqdm import tqdm
#from PIL import Image
from omegaconf import OmegaConf
#import os
#from tkinter import EXCEPTION, messagebox as mbox
from dotted_dict import DottedDict
from .dataloader import dataloader
#sys.path.append("..")
from .learning import utilities as util
from .learning import ( PreprocessedImageDataSingleChannel,
                        PreprocessedImageDataMultiChannel,
                        PreprocessedImageDataSingleChannel_smlm,
                        PreprocessedImageDataMultiChannel_smlm,
                        Fitter,
                        PSFVolumeBased,
                        PSFPupilBased,
                        PSFZernikeBased,
                        PSFZernikeBased_FD,
                        PSFVolumeBased4pi,
                        PSFPupilBased4pi,
                        PSFZernikeBased4pi,
                        PSFMultiChannel,
                        PSFMultiChannel_smlm,
                        PSFMultiChannel4pi,
                        PSFZernikeBased_vector_smlm,
                        PSFPupilBased_vector_smlm,
                        PSFZernikeBased_FD_smlm,
                        PSFMultiChannel4pi_smlm,
                        PSFZernikeBased4pi_smlm,
                        L_BFGS_B,
                        localizationlib,
                        psf2cspline_np,
                        mse_real,
                        mse_real_zernike,
                        mse_real_zernike_FD,
                        mse_real_zernike_smlm,
                        mse_real_pupil_smlm,
                        mse_real_zernike_FD_smlm,
                        mse_real_4pi,
                        mse_zernike_4pi,
                        mse_zernike_4pi_smlm,
                        mse_real_pupil,
                        mse_pupil_4pi,
                        mse_real_All,
                        mse_real_4pi_All)


#%%
PSF_DICT = dict(voxel=PSFVolumeBased, 
                pupil=PSFPupilBased,
                pupil_vector=PSFPupilBased,
                zernike=PSFZernikeBased,
                zernike_vector=PSFZernikeBased,
                zernike_FD=PSFZernikeBased_FD,
                zernike_vector_FD=PSFZernikeBased_FD,
                insitu_zernike = PSFZernikeBased_vector_smlm,
                insitu_pupil = PSFPupilBased_vector_smlm,
                insitu_FD = PSFZernikeBased_FD_smlm)

LOSSFUN_DICT = dict(voxel=mse_real, 
                pupil=mse_real_pupil,
                pupil_vector=mse_real_pupil,
                zernike=mse_real_zernike,
                zernike_vector=mse_real_zernike,
                zernike_FD=mse_real_zernike_FD,
                zernike_vector_FD=mse_real_zernike_FD,
                insitu_zernike = mse_real_zernike_smlm,
                insitu_pupil = mse_real_pupil_smlm,
                insitu_FD = mse_real_zernike_FD_smlm)


PSF_DICT_4pi = dict(voxel=PSFVolumeBased4pi, 
                    pupil=PSFPupilBased4pi,
                    zernike=PSFZernikeBased4pi,
                    insitu_zernike = PSFZernikeBased4pi_smlm)

LOSSFUN_DICT_4pi = dict(voxel=mse_real_4pi, 
                pupil=mse_pupil_4pi,
                zernike=mse_zernike_4pi,
                insitu_zernike=mse_zernike_4pi_smlm)

            

class psflearninglib:
    def __init__(self,param=None):
        self.param = param
        self.loc_FD = None



    def getpsfclass(self):
        param = self.param
        PSFtype = param.PSFtype
        channeltype = param.channeltype
        lossfun = LOSSFUN_DICT[PSFtype]
        lossfunmulti = None

        if channeltype == 'single':
            psfclass = PSF_DICT[PSFtype]
            psfmulticlass = None
        elif channeltype == 'multi':
            psfclass = PSF_DICT[PSFtype]
            if 'insitu' in PSFtype:
                psfmulticlass = PSFMultiChannel_smlm
            else:
                psfmulticlass = PSFMultiChannel
            lossfunmulti = mse_real_All
        elif channeltype == '4pi':
            psfclass = PSF_DICT_4pi[PSFtype]
            lossfun = LOSSFUN_DICT_4pi[PSFtype]
            if 'insitu' in PSFtype:
                psfmulticlass = PSFMultiChannel4pi_smlm
            else:
                psfmulticlass = PSFMultiChannel4pi
            lossfunmulti = mse_real_4pi_All

        self.psf_class = psfclass
        self.psf_class_multi = psfmulticlass
        self.loss_fun = lossfun
        self.loss_fun_multi = lossfunmulti
        return

    def load_data(self,frange=None):
        param = self.param
        varname = param.varname
        format = param.format
        channeltype = param.channeltype
        PSFtype = param.PSFtype
        ref_channel = param.ref_channel
        filelist = param.filelist

        loader = dataloader(param)
        if not filelist:
            filelist = loader.getfilelist()

        if frange:
            filelist = filelist[frange[0]:frange[1]]
        
        if format == '.mat':
            imagesall = loader.loadmat(filelist)
        elif (format == '.tif') or (format == '.tiff'):
            imagesall = loader.loadtiff(filelist)
        elif format == '.czi':
            imagesall = loader.loadczi(filelist)
        elif format == '.h5':
            imagesall = loader.loadh5(filelist)
        else:
            raise TypeError('supported data format is '+'.mat,'+'.tif,'+'.czi,'+'.h5.')

        
        

        if channeltype == '4pi':
            if 'insitu' in PSFtype:
                images = np.transpose(imagesall,(1,0,2,3,4)) 
            else:
                if varname:
                    images = np.transpose(imagesall,(1,0,2,3,4,5))
                else:
                    images = np.transpose(imagesall,(1,0,3,2,4,5))
        elif channeltype == 'multi':
            images = np.transpose(imagesall,(1,0,2,3,4))
            Nchannel = images.shape[0]
            defocus = []
            for i in range(0,Nchannel):
                defocus.append(param.option.multi.defocus_offset+i*param.option.multi.defocus_delay)
            
            defocus[0],defocus[ref_channel] = defocus[ref_channel],defocus[0]
            self.param.option.multi.defocus = defocus
            id = list(range(images.shape[0]))
            id[0],id[ref_channel] = id[ref_channel],id[0]
            images = images[id]
        else:
            images = imagesall

        if 'insitu' in PSFtype:
            if channeltype == 'single':
                images = images.reshape(-1,images.shape[-2],images.shape[-1])
            elif channeltype == 'multi':
                images = images.reshape(images.shape[0],-1,images.shape[-2],images.shape[-1])
            elif channeltype == '4pi':
                images = images.reshape(images.shape[0],-1,images.shape[-2],images.shape[-1])
        
        
        if param.swapxy:
            #if format == '.tif':
            tmp = np.zeros(images.shape[:-2]+(images.shape[-1],images.shape[-2]),dtype=np.float32)            
            tmp[0:] = np.swapaxes(images[0:],-1,-2)
            images = tmp

        

        if (param.stage_mov_dir=='reverse') & (param.datatype == 'bead'):
            images = np.flip(images,axis=-3)
        
        print(images.shape)
        return images

    def prep_data(self,images):
        param = self.param
        peak_height = param.roi.peak_height
        roi_size = param.roi.roi_size
        gaus_sigma = param.roi.gauss_sigma
        kernel = param.roi.max_kernel
        pixelsize_x = param.pixel_size.x
        pixelsize_y = param.pixel_size.y
        pixelsize_z = param.pixel_size.z
        bead_radius = param.roi.bead_radius
        showplot = param.plotall
        zT = param.fpi.modulation_period
        PSFtype = param.PSFtype
        channeltype = param.channeltype
        fov = list(param.FOV.values())
        skew_const = param.LLS.skew_const
        maxNobead = param.roi.max_bead_number


        zstart = fov[-3]
        zend = images.shape[-3]+fov[-2]
        zstep = fov[-1]
        zind = range(zstart,zend,zstep)
        ims = np.swapaxes(images,0,-3)

        ims = ims[zind]
        images = np.swapaxes(ims,0,-3)

        if PSFtype == 'voxel':
            isvolume = True
            padpsf = False
        else:
            isvolume = False
            padpsf = False

        if channeltype == 'single':
            if 'insitu' in PSFtype:
                dataobj = PreprocessedImageDataSingleChannel_smlm(images)
            else:
                dataobj = PreprocessedImageDataSingleChannel(images)
        elif channeltype == '4pi':
            if 'insitu' in PSFtype:
                dataobj = PreprocessedImageDataMultiChannel_smlm(images, PreprocessedImageDataSingleChannel_smlm, is4pi=True)
            else:
                dataobj = PreprocessedImageDataMultiChannel(images, PreprocessedImageDataSingleChannel, is4pi=True)        
        elif channeltype == 'multi':
            if 'insitu' in PSFtype:
                dataobj = PreprocessedImageDataMultiChannel_smlm(images, PreprocessedImageDataSingleChannel_smlm)
            else:
                dataobj = PreprocessedImageDataMultiChannel(images, PreprocessedImageDataSingleChannel)
        
        if fov[2]==0:
            fov = None
        if (skew_const[0]==0.0) & (skew_const[1]==0.0):
            skew_const = None
            
        #dataobj.shiftxy = np.array([[0,0],[0.3,1.6]])
        dataobj.process( roi_size = roi_size,
                        gaus_sigma=gaus_sigma,
                        min_border_dist= list(np.array(roi_size)//2+1),
                        min_center_dist = np.max(roi_size),
                        FOV=fov,
                        max_threshold= peak_height,
                        max_kernel=kernel,
                        pixelsize_x = pixelsize_x,
                        pixelsize_y = pixelsize_y,
                        pixelsize_z = pixelsize_z,
                        bead_radius = bead_radius,
                        modulation_period=zT,
                        plot = showplot,
                        padPSF = padpsf,
                        isVolume = isvolume,
                        skew_const=skew_const,
                        max_bead_number=maxNobead)
        
        return dataobj

    def initializepsf(self):
        param = self.param
        w = list(param.loss_weight.values())
        optionparam = param.option
        batchsize = param.batch_size

        if self.psf_class_multi is None:
            psfobj = self.psf_class(options=optionparam)
            if 'vector' in param.PSFtype:
                psfobj.psftype = 'vector'
        else:
            optimizer_single = L_BFGS_B(maxiter=50)
            optimizer_single.batch_size = batchsize
            psfobj = self.psf_class_multi(self.psf_class,optimizer_single,options=optionparam,loss_weight=w)
            if 'vector' in param.PSFtype:
                psfobj.PSFtype = 'vector'

        return psfobj
    
    def learn_psf(self,dataobj,time=None):
        param = self.param
        rej_threshold = list(param.rej_threshold.values())
        maxiter = param.iteration
        w = list(param.loss_weight.values())
        usecuda = param.usecuda
        showplot = param.plotall
        optionparam = param.option
        channeltype = param.channeltype
        PSFtype = param.PSFtype
        roi_size = param.roi.roi_size
        batchsize = param.batch_size
        pupilfile = optionparam.model.init_pupil_file
        psfobj = self.initializepsf()

        if pupilfile:
            f = h5.File(pupilfile, 'r')
            if channeltype == 'single':
                try:
                    psfobj.initpupil = np.array(f['res']['pupil'])
                except:
                    pass

                try:
                    psfobj.Zoffset = np.array(f['res']['zoffset'])
                except:
                    pass

                try:
                    psfobj.initpsf = np.array(f['res']['I_model_reverse']).astype(np.float32)
                except:
                    psfobj.initpsf = np.array(f['res']['I_model']).astype(np.float32)

                try:
                    psfobj.initzcoeff = np.array(f['res']['zernike_coeff']).astype(np.float32)
                except:
                    pass

                
            else:
                Nchannels = len(dataobj.channels)
                psfobj.initpupil = [None]*Nchannels
                psfobj.initpsf = [None]*Nchannels
                if channeltype == '4pi':
                    psfobj.initA = [None]*Nchannels
                for k in range(0,Nchannels):
                    try:
                        psfobj.initpupil[k] = np.array(f['res']['channel'+str(k)]['pupil'])
                    except:
                        pass
                    try:
                        psfobj.Zoffset = np.array(f['res']['channel'+str(k)]['zoffset'])
                    except:
                        pass
                    psfobj.initpsf[k] = np.array(f['res']['channel'+str(k)]['I_model']).astype(np.float32)
                    if channeltype == '4pi':
                        psfobj.initA[k] = np.array(f['res']['channel'+str(k)]['A_model']).astype(np.complex64)


        optimizer = L_BFGS_B(maxiter=maxiter)
        optimizer.batch_size = batchsize
        if self.loss_fun_multi:
            fitter = Fitter(dataobj, psfobj,optimizer,self.loss_fun_multi,loss_func_single=self.loss_fun,loss_weight=w)
        else:
            fitter = Fitter(dataobj, psfobj,optimizer,self.loss_fun,loss_weight=w)
        _, _, centers, file_idxs = dataobj.get_image_data()
        centers = np.stack(centers)
        res, toc = fitter.learn_psf(start_time=time)

        pos = res[-1][0]
        zpos = pos[:,0:1]
        zpos = zpos-np.mean(zpos)
        if (centers.shape[-1]==3) & (np.max(np.abs(zpos))>2) & (PSFtype=='voxel'):
            cor = dataobj.centers

            if dataobj.skew_const:
                sk = dataobj.skew_const
                centers1 = np.int32(np.round(np.hstack((cor[:,0:1]-zpos,cor[:,1:2]-sk[0]*zpos,cor[:,2:]-sk[1]*zpos))))
            else:
                centers1 = np.int32(np.round(np.hstack((cor[:,0:1]-zpos,cor[:,1:2],cor[:,2:]))))
            dataobj.cut_new_rois(centers1, file_idxs, roi_size=roi_size)
            offset = np.min((np.quantile(dataobj.rois,1e-3),0))
            dataobj.rois = dataobj.rois-offset
            if dataobj.skew_const:
                dataobj.deskew_roi(roi_size)

            fitter.dataobj=dataobj
            res, toc = fitter.learn_psf(start_time=time)
        
        if len(file_idxs)==1:
            locres = fitter.localize(res,channeltype,usecuda=usecuda,plot=showplot,start_time=toc)
            res1 = res
        else:
             
            # %%  remove ourlier
            if 'insitu' in PSFtype:
                #th = [0.99,0.9] # quantile
                res1,toc = fitter.relearn_smlm(res,channeltype,rej_threshold,start_time=toc)
                locres = fitter.localize_smlm(res1,channeltype,plot=showplot)
            else:
                locres = fitter.localize(res,channeltype,usecuda=usecuda,plot=showplot,start_time=toc)
                toc = locres[-2]
                res1, toc = fitter.relearn(res,channeltype,rej_threshold,start_time=toc)
                if res1[0].shape[-2] < res[0].shape[-2]:
                    locres = fitter.localize(res1,channeltype,usecuda=usecuda,plot=showplot,start_time=toc)
            
        self.learning_result = res1
        self.loc_result = locres
        return psfobj, fitter

    def localize_FD(self,fitter, initz=None):
        res = self.learning_result
        usecuda = self.param.usecuda
        showplot = self.param.plotall
        channeltype = self.param.channeltype
        loc_FD = fitter.localize_FD(res, channeltype, usecuda=usecuda, initz=initz,plot=showplot)
        self.loc_FD = loc_FD
        return loc_FD

    def iterlearn_psf(self,dataobj,time=None):
        min_photon = self.param.option.insitu.min_photon
        iterN = self.param.option.insitu.repeat
        pz = self.param.pixel_size.z
        channeltype = self.param.channeltype
        savename = self.param.savename
        for nn in range(0,iterN):
            if nn >0:
                dataobj.resetdata()
            psfobj,fitter = self.learn_psf(dataobj,time=time)
            self.param.savename = savename + str(nn)
            resfile = self.save_result(psfobj,dataobj,fitter)
            self.param.option.model.init_pupil_file = resfile
            self.param.option.insitu.min_photon = max([min_photon-nn*0.1,0.2])
            res = psfobj.res2dict(self.learning_result)
            
            if channeltype == 'single':
                self.param.option.insitu.stage_pos = float(res['stagepos'])
                I_model = res['I_model']
                Nz = I_model.shape[-3]
                zind = range(0,Nz,4)
                if self.param.plotall:
                    fig = plt.figure(figsize=[3*len(zind),3])
                    for i,id in enumerate(zind):
                        ax = fig.add_subplot(1,len(zind),i+1)
                        plt.imshow(I_model[id],cmap='twilight')
                        plt.axis('off')
                    plt.show()
            else:
                try:
                    self.param.option.insitu.stage_pos = float(res['channel0']['stagepos'])
                except:
                    pass
                if self.param.plotall:
                    for j in range(0,len(dataobj.channels)):
                        if channeltype == '4pi':
                            I_model = res['channel'+str(j)]['psf_model']
                        else:
                            I_model = res['channel'+str(j)]['I_model']
                        Nz = I_model.shape[-3]
                        zind = range(0,Nz,4)

                        fig = plt.figure(figsize=[3*len(zind),3])
                        for i,id in enumerate(zind):
                            ax = fig.add_subplot(1,len(zind),i+1)
                            plt.imshow(I_model[id],cmap='twilight')
                            plt.axis('off')
                    plt.show()

        
        return resfile


    def save_result(self,psfobj,dataobj,fitter):
        
        param = self.param
        res = self.learning_result
        locres = self.loc_result
        toc = locres[-2]
        pbar = tqdm(desc='6/6: saving results',bar_format = "{desc}: [{elapsed}s] {postfix[0]}{postfix[1][time]:>4.2f}s",postfix=["total time: ", dict(time=toc)])
        
        folder = param.datapath
        savename = param.savename+'_'+param.PSFtype+'_'+param.channeltype
        res_dict = psfobj.res2dict(res)
        coeff_reverse = self.gencspline(res_dict,psfobj,keyname='I_model_reverse')
        coeff = self.gencspline(res_dict,psfobj)

        if self.loc_FD is not None:
            locres_dict = dict(P=locres[0],CRLB = locres[1],LL=locres[2],coeff=coeff,coeff_bead=locres[3],loc=locres[-1],loc_FD=self.loc_FD,coeff_reverse=coeff_reverse)
        else:
            locres_dict = dict(P=locres[0],CRLB = locres[1],LL=locres[2],coeff=coeff,coeff_bead=locres[3],loc=locres[-1],coeff_reverse=coeff_reverse)
        img, _, centers, file_idxs = dataobj.get_image_data()
        img = np.stack(img)
        rois_dict = dict(cor=np.stack(centers),fileID=np.stack(file_idxs),psf_data=fitter.rois,
                        psf_fit=fitter.forward_images,image_size=img.shape)
        resfile = savename+'.h5'
        self.writeh5file(resfile,res_dict,locres_dict,rois_dict)

        self.result_file = resfile
        pbar.postfix[1]['time'] = toc +pbar._time()-pbar.start_t
        pbar.update()
        pbar.close
        return resfile
    
    def writeh5file(self,filename,res_dict,locres_dict,rois_dict):
        with h5.File(filename, "w") as f:
            f.attrs["params"] = json.dumps(OmegaConf.to_container(self.param))
            g3 = f.create_group("rois")
            g1 = f.create_group("res")
            g2 = f.create_group("locres")

            for k, v in locres_dict.items():
                if isinstance(v,dict):
                    gi = g2.create_group(k)
                    for ki,vi in v.items():
                        gi[ki] = vi
                else:
                    g2[k] = v
            for k, v in res_dict.items():
                if isinstance(v,dict):
                    gi = g1.create_group(k)
                    for ki,vi in v.items():
                        gi[ki] = vi
                else:
                    g1[k] = v
            for k, v in rois_dict.items():
                g3[k] = v
        
        return


    def gencspline(self, res_dict,psfobj,keyname='I_model'):
        param = self.param
        coeff = []
        if param.channeltype == 'single':
            if keyname in res_dict:
                I_model = res_dict[keyname]
                offset = np.min(I_model)
                Imd = I_model-offset
                normf = np.median(np.sum(Imd,axis = (-1,-2)))
                Imd = Imd/normf
                coeff = psf2cspline_np(Imd)
                coeff = coeff.astype(np.float32)
        if param.channeltype == 'multi':
            if keyname in res_dict['channel0']:
                Nchannel = len(psfobj.sub_psfs)
                I_model = []
                for i in range(Nchannel):
                    I_model.append(res_dict['channel'+str(i)][keyname])       
                I_model = np.stack(I_model)
                offset = np.min(I_model)
                Iall = []
                Imd = I_model-offset
                normf = np.max(np.median(np.sum(Imd,axis = (-1,-2)),axis=-1))
                Imd = Imd/normf
                for i in range(Nchannel):                                             
                    coeff = psf2cspline_np(Imd[i])
                    Iall.append(coeff)
                coeff = np.stack(Iall).astype(np.float32)
        if param.channeltype == '4pi':
            if keyname in res_dict['channel0']:
                Nchannel = len(psfobj.sub_psfs)
                I_model = []
                A_model = []
                for i in range(Nchannel):
                    I_model.append(res_dict['channel'+str(i)][keyname])   
                    if keyname=='I_model':
                        A_model.append(res_dict['channel'+str(i)]['A_model'])      
                    else: 
                        A_model.append(res_dict['channel'+str(i)]['A_model_reverse'])       
                I_model = np.stack(I_model)
                A_model = np.stack(A_model)
                offset = np.min(I_model-2*np.abs(A_model))
                Imd = I_model-offset
                normf = np.max(np.median(np.sum(Imd[:,1:-1],axis = (-1,-2)),axis=-1))*2.0
                Imd = Imd/normf
                Amd = A_model/normf
                IABall = []
                for i in range(Nchannel):                         
                    Ii = Imd[i]
                    Ai = 2*np.real(Amd[i])
                    Bi = -2*np.imag(Amd[i]) 
                    IAB = [psf2cspline_np(Ai),psf2cspline_np(Bi),psf2cspline_np(Ii)]  
                    IAB = np.stack(IAB)
                    IABall.append(IAB)
                coeff = np.stack(IABall).astype(np.float32)

        return coeff
    
    def genpsf(self,f,Nz=21,xsz=21,stagepos=1.0):
        p = self.param
        dataobj = DottedDict(pixelsize_x = p.pixel_size.x,
                            pixelsize_y = p.pixel_size.y,
                            pixelsize_z = p.pixel_size.z,
                            image_size = list(f.rois.image_size),
                            rois = np.zeros((Nz,xsz,xsz)))
        self.getpsfclass()
        psfobj = self.initializepsf()
        if p.channeltype == 'single':
            sigma = f.res.sigma
            Zcoeff = f.res.zernike_coeff
            Zcoeff = Zcoeff.reshape((Zcoeff.shape+(1,1)))
            psfobj.data = dataobj
            if 'insitu' in p.PSFtype:
                psfobj.stagepos = stagepos/p.pixel_size.z
                psfobj.estzoffset(Nz=Nz) 
            else:
                if psfobj.psftype=='vector':
                    psfobj.calpupilfield('vector',Nz=Nz)
                else:
                    psfobj.calpupilfield('scalar',Nz=Nz)

            if 'FD' in p.PSFtype:
                #dx = f.rois.image_size[-1]/f.res.zernike_map.shape[-1]/2
                #dy = f.rois.image_size[-2]/f.res.zernike_map.shape[-2]/2
                xrange = np.linspace(0,f.rois.image_size[-1],f.res.zernike_map.shape[-1]+1,dtype=np.float32)
                yrange = np.linspace(0,f.rois.image_size[-2],f.res.zernike_map.shape[-2]+1,dtype=np.float32)
                [xx,yy] = np.meshgrid(xrange,yrange)
                cor = np.vstack((xx.flatten(),yy.flatten())).transpose()
                Zmap = f.res.zernike_map
                batchsize = 200
                ind = list(np.int32(np.linspace(0,cor.shape[0],cor.shape[0]//batchsize+2)))
                for i in range(len(ind)-1):
                    I0,_,_ = psfobj.genpsfmodel(sigma,Zmap=Zmap,cor = cor[ind[i]:ind[i+1]])
                    if i == 0:
                        I_model = I0
                    else:
                        I_model = np.vstack((I_model,I0))
                #I_model, _, _ = psfobj.genpsfmodel(sigma,Zmap,cor)
            else:
                I_model,_ = psfobj.genpsfmodel(sigma,Zcoeff)
            f.res.I_model = I_model
        elif p.channeltype == 'multi':
            Nchannel = f.rois.cor.shape[0]
            psfobj.sub_psfs = [None]*Nchannel
            for i in range(Nchannel):
                psf = psfobj.psftype(options = psfobj.options)
                psf.psftype = psfobj.PSFtype
                psfobj.sub_psfs[i] = psf
                sigma = f.res['channel'+str(i)].sigma
                Zcoeff = f.res['channel'+str(i)].zernike_coeff
                Zcoeff = Zcoeff.reshape((Zcoeff.shape+(1,1)))
                psf.data = dataobj
                if 'insitu' in p.PSFtype:
                    psf.stagepos = stagepos/p.pixel_size.z
                    psf.estzoffset(Nz=Nz)
                else:
                    if psf.psftype=='vector':
                        psf.calpupilfield('vector',Nz=Nz)
                    else:
                        psf.calpupilfield('scalar',Nz=Nz)

                I_model,_ = psf.genpsfmodel(sigma,Zcoeff)
                f.res['channel'+str(i)].I_model = I_model

        return f, psfobj
    
    def calstrehlratio(self,f,xsz = 31):
        f1 = f.copy()
        p = self.param
        if p.channeltype == 'single':
            if 'FD' in p.PSFtype:
                f1.res.zernike_map = f.res.zernike_map.copy()
                f1.res.zernike_map[1,0:4] = 0.0
                f1,psfobj = self.genpsf(f1,Nz=1,xsz=xsz)
                I_model = f1.res.I_model/np.sum(f1.res.I_model,axis=(-1,-2),keepdims=True)
                I1 = I_model[:,0,xsz//2,xsz//2]

                f1.res.zernike_map = np.zeros(f1.res.zernike_map.shape,dtype=np.float32)
                f1.res.zernike_map[0,0] = 1
                f1,psfobj = self.genpsf(f1,Nz=1,xsz=xsz)
                I_model = f1.res.I_model/np.sum(f1.res.I_model,axis=(-1,-2),keepdims=True)
                I0 = I_model[:,0,xsz//2,xsz//2]
                strehlratio = np.float32(I1/I0)
                strehlratio_map = np.reshape(strehlratio,(f.res.zernike_map.shape[-2]+1,f.res.zernike_map.shape[-1]+1))
                plt.imshow(strehlratio_map)
                plt.colorbar()
                plt.title('Strehl ratio map',fontsize=15)
            else:
                f1.res.zernike_coeff[1,0:4] = 0.0
                f1,psfobj = self.genpsf(f1,Nz=1,xsz=xsz)
                I_model = f1.res.I_model/np.sum(f1.res.I_model)
                I1 = I_model[0,xsz//2,xsz//2]

                f1.res.zernike_coeff = np.zeros(f1.res.zernike_coeff.shape,dtype=np.float32)
                f1.res.zernike_coeff[0,0] = 1
                f1,psfobj = self.genpsf(f1,Nz=1,xsz=xsz)
                I_model = f1.res.I_model/np.sum(f1.res.I_model)
                I0 = I_model[0,xsz//2,xsz//2]
                strehlratio = np.float32(I1/I0)
                print('Strehl ratio: ',strehlratio)
        elif p.channeltype == 'multi':
            Nchannel = f1.rois.cor.shape[0]
            I1 = []
            I0 = []
            for i in range(Nchannel):
                f1.res['channel'+str(i)].zernike_coeff[1,0:4] = 0.0
            f1,psfobj = self.genpsf(f1,Nz=1,xsz=xsz)
            coeff = np.zeros(f1.res.channel0.zernike_coeff.shape,dtype=np.float32)
            coeff[0,0] = 1

            for i in range(Nchannel):
                I_model = f1.res['channel'+str(i)].I_model/np.sum(f1.res['channel'+str(i)].I_model)
                I1.append(I_model[0,xsz//2,xsz//2])
                f1.res['channel'+str(i)].zernike_coeff = coeff


            f1,psfobj = self.genpsf(f1,Nz=1,xsz=31)
            for i in range(Nchannel):
                I_model = f1.res['channel'+str(i)].I_model/np.sum(f1.res['channel'+str(i)].I_model)
                I0.append(I_model[0,xsz//2,xsz//2])

            I1 = np.stack(I1)
            I0 = np.stack(I0)
            strehlratio = np.float32(I1/I0)
            print('Strehl ratio: ', strehlratio)
        elif p.channeltype == '4pi':
            Nchannel = f.rois.cor.shape[0]
            mdepth = np.zeros(Nchannel)
            for i in range(0,Nchannel):
                mdepth[i] = f.res['channel'+str(i)].modulation_depth
            print('modulation depth: ',np.round(mdepth,2))
            strehlratio = mdepth
        return strehlratio
    
    def calfwhm(self,f):
        p = self.param
        f1 = f.copy()
        if p.channeltype == 'single':
            if 'FD' in p.PSFtype:
                psfsize = f.res.I_model_bead.shape
                f1.res.zernike_map = f.res.zernike_map.copy()
                f1.res.zernike_map[1,0:4] = 0.0
                f1,psfobj = self.genpsf(f1,Nz=psfsize[-3],xsz=psfsize[-1])
                I_model = f1.res.I_model
                fwhmx = np.zeros(I_model.shape[0])
                fwhmy = np.zeros(I_model.shape[0])
                fwhmz = np.zeros(I_model.shape[0])
                for i,psfi in enumerate(I_model):
                    Ix, xh, Iy, yh, Iz, zh = self.getfwhm(psfi)
                    fwhmx[i] = np.diff(xh)*p.pixel_size.x*1e3
                    fwhmy[i] = np.diff(yh)*p.pixel_size.y*1e3
                    fwhmz[i] = np.diff(zh)*p.pixel_size.z*1e3

                fig = plt.figure(figsize=[12,5])
                fwhmx_map = np.reshape(fwhmx,(f.res.zernike_map.shape[-2],f.res.zernike_map.shape[-1]))
                fwhmy_map = np.reshape(fwhmy,(f.res.zernike_map.shape[-2],f.res.zernike_map.shape[-1]))
                fwhmz_map = np.reshape(fwhmz,(f.res.zernike_map.shape[-2],f.res.zernike_map.shape[-1]))
                ax = fig.add_subplot(121)
                plt.imshow((fwhmx_map+fwhmy_map)/2)
                clb = plt.colorbar()
                clb.ax.set_title('nm')
                plt.title('FWHMxy map',fontsize=15)
                ax = fig.add_subplot(122)
                plt.imshow(fwhmz_map)
                clb = plt.colorbar()
                clb.ax.set_title('nm')
                plt.title('FWHMz map',fontsize=15)
                fwhmx = fwhmx_map
                fwhmy = fwhmy_map
                fwhmz = fwhmz_map

            else:
                I_model = f.res.I_model
                Imaxh = np.max(I_model)/2
                Ix, xh, Iy, yh, Iz, zh = self.getfwhm(I_model)
                fwhmx = np.diff(xh)*p.pixel_size.x*1e3
                fwhmy = np.diff(yh)*p.pixel_size.y*1e3
                fwhmz = np.diff(zh)*p.pixel_size.z*1e3
                fig = plt.figure(figsize=[12,4])
                ax = fig.add_subplot(121)
                plt.plot(Ix,'o-')
                plt.plot(xh,[Imaxh,Imaxh],'-')
                plt.plot(Iy,'o-')
                plt.plot(yh,[Imaxh,Imaxh],'-')
                plt.title('FWHMxy: '+str(np.round((fwhmx[0]+fwhmy[0])/2,2))+' nm',fontsize=15)
                plt.xlabel('x (pixel)')
                plt.ylabel('intensity')

                ax = fig.add_subplot(122)
                plt.plot(Iz,'o-')
                plt.plot(zh,[Imaxh,Imaxh],'-')
                plt.title('FWHMz: '+str(np.round(fwhmz[0],2))+' nm',fontsize=15)
                plt.xlabel('z (pixel)')
                plt.ylabel('intensity')
                
        elif p.channeltype == 'multi':
            Nchannel = f.rois.cor.shape[0]
            fig = plt.figure(figsize=[4*Nchannel,8])
            spec = gridspec.GridSpec(ncols=Nchannel, nrows=2,
                        width_ratios=list(np.ones(Nchannel)), wspace=0.4,
                        hspace=0.3, height_ratios=[1, 1])

            fwhmx = []
            fwhmy = []
            fwhmz = []
            for i in range(0,Nchannel):
                I_model = f.res['channel'+str(i)].I_model
                Imaxh = np.max(I_model)/2
                Ix, xh, Iy, yh, Iz, zh = self.getfwhm(I_model)
                fwhmxi = np.diff(xh)*p.pixel_size.x*1e3
                fwhmyi = np.diff(yh)*p.pixel_size.y*1e3
                fwhmzi = np.diff(zh)*p.pixel_size.z*1e3
                
                ax = fig.add_subplot(spec[i])
                plt.plot(Ix,'o-')
                plt.plot(xh,[Imaxh,Imaxh],'-')
                plt.plot(Iy,'o-')
                plt.plot(yh,[Imaxh,Imaxh],'-')
                plt.title('ch'+str(i)+' FWHMxy: '+str(np.round((fwhmxi[0]+fwhmyi[0])/2,2))+' nm',fontsize=15)
                plt.xlabel('x (pixel)')
                plt.ylabel('intensity')

                ax = fig.add_subplot(spec[Nchannel+i])
                plt.plot(Iz,'o-')
                plt.plot(zh,[Imaxh,Imaxh],'-')
                plt.title('ch'+str(i)+' FWHMz: '+str(np.round(fwhmzi[0],2))+' nm',fontsize=15)
                plt.xlabel('z (pixel)')
                plt.ylabel('intensity')
                
                fwhmx.append(fwhmxi)
                fwhmz.append(fwhmzi)
            fwhmx = np.stack(fwhmx)
            fwhmz = np.stack(fwhmz)

        plt.show()
        return fwhmx, fwhmy, fwhmz

    def getfwhm(self,I_model):
        cor = np.unravel_index(np.argmax(I_model),I_model.shape)
        # lateral
        Ix = I_model[cor[0],cor[1]]
        xh = self.get1dfwhm(Ix,cor[2])

        Iy = I_model[cor[0],:,cor[2]]
        yh = self.get1dfwhm(Iy,cor[1])

        # axial
        Iz = I_model[:,cor[1],cor[2]]
        zh = self.get1dfwhm(Iz,cor[0])

        return Ix, xh, Iy, yh, Iz,zh
    
    def get1dfwhm(self,I,cor):
        Imaxh = np.max(I)/2

        x1 = np.argsort(np.abs(I[:cor]-Imaxh))[0]
        if I[x1]>Imaxh:
            x1=[x1,x1-1]
        else:
            x1=[x1,x1+1]
        x2 = np.argsort(np.abs(I[cor:]-Imaxh))[0]+cor
        if I[x2]>Imaxh:
            x2=[x2,x2+1]
        else:
            x2=[x2,x2-1]
        g = np.diff(x1)/np.diff(I[x1])
        xh1 = g*(Imaxh-I[x1[0]])+x1[0]
        x1 = np.array(x1,dtype = np.float64)
        xh1 = np.minimum(np.maximum(xh1,np.min(x1)),np.max(x1))
        g = np.diff(x2)/np.diff(I[x2])
        xh2 = g*(Imaxh-I[x2[0]])+x2[0]
        x2 = np.array(x2,dtype = np.float64)
        xh2 = np.minimum(np.maximum(xh2,np.min(x2)),np.max(x2))
        return np.hstack([xh1, xh2])

    def localize(self,f,datarange=[0,5]):
        # currently tested for .h5 data, single channel system
        #% build layers for segmentation
        sigma = self.param.roi.gauss_sigma
        poolsize = self.param.roi.max_kernel
        roisize = self.param.roi.roi_size
        thresh = self.param.roi.peak_height

        conv2d,max_pool_2d,conv_uniform = util.gen_layers(f.rois.image_size,sigma,poolsize,roisize)
        x = []
        y = []
        z = []
        LL = []
        batchsize = 1000
        for id in range(datarange[0],datarange[1]):
            # load data
            self.param.insitu.dataId = id
            images = self.load_data()
            # batch process
            Nf = images.shape[0]
            ind = list(range(0,Nf,batchsize))+[Nf]
            for i in range(len(ind)-1):
                # segment
                rois,coords = util.crop_rois(images[ind[i]:ind[i+1]],conv2d,max_pool_2d,conv_uniform,thresh,roisize)
                #% remove negative values
                offset = np.min((np.quantile(rois,1e-3),0))
                rois = rois - offset
                #% localize
                dll = localizationlib(usecuda=True)
                pz = self.param.pixel_size.z
                locres = dll.loc_ast(rois,f.res.I_model,pz)
                # collect results
                x.append(coords[:,-1]+locres[-1]['x'].flatten())
                y.append(coords[:,-2]+locres[-1]['y'].flatten())
                z.append(locres[-1]['z'].flatten())
                LL.append(locres[2])
        # combine results
        x = np.hstack(x)
        y = np.hstack(y)
        z = np.hstack(z)
        LL = np.hstack(LL)

        return x,y,z,LL
    
    def identify_background(self,xf,yf,zf,nfovbin=6,zbinsize=0.1,zpeak=0.07):
        #%% select background region from localizations
        nbin_x = nfovbin
        nbin_y = nfovbin
        count_x, edge_x = np.histogram(xf, nbin_x)
        count_y, edge_y = np.histogram(yf, nbin_y)
        ind_x = np.digitize(xf, edge_x)
        ind_y = np.digitize(yf, edge_y)
        x0 = []
        y0 = []
        z0 = []
        for xx in range(1,nbin_x+1):
            for yy in range(1,nbin_y+1):
                maskid = np.where((ind_x==xx) & (ind_y==yy))
                if maskid[0].size>0:
                    zi = zf[maskid]                                        
                    out = np.histogram(zi,bins=np.arange(0,25,zbinsize))
                    g = out[0]
                    bins = (out[1][1:]+out[1][:-1])/2
                    ind, _ = sig.find_peaks(g/np.max(g),height=zpeak,width=None)
                    zcutoff = bins[ind[0]]+1
                    mask = zi<zcutoff
                    x0.append(xf[maskid][mask])
                    y0.append(yf[maskid][mask])
                    z0.append(zf[maskid][mask])

        x0 = np.hstack(x0)
        y0 = np.hstack(y0)
        z0 = np.hstack(z0)

        return x0,y0,z0