"""
Copyright (c) 2022      Ries Lab, EMBL, Heidelberg, Germany
All rights reserved     

@author: Sheng Liu
"""
#%%
from pickle import FALSE
import h5py as h5
import czifile as czi
import numpy as np
from skimage import io
# append the path of the parent directory as long as it's not a real package
import glob
import json
from PIL import Image


class dataloader:
    def __init__(self,param=None):
        self.param = param

    def getfilelist(self):
        param = self.param
        if not param.subfolder:
            filelist = glob.glob(param.datapath+'/*'+param.keyword+'*'+param.format)
        else:
            filelist = []
            folderlist = glob.glob(param.datapath+'/*'+param.subfolder+'*/')
            for f in folderlist:
                filelist.append(glob.glob(f+'/*'+param.keyword+'*'+param.format)[0])
        
        return filelist


    def loadtiff(self,filelist):
        param = self.param
        imageraw = []
        for filename in filelist:
            print(filename)
            if param.datatype == 'smlm':
                dat = []
                fID = Image.open(filename)
                
                for ii in range(param.insitu.frame_range[0],param.insitu.frame_range[1]):
                    fID.seek(ii)
                    dat.append(np.asarray(fID))
                dat = np.stack(dat).astype(np.float32)
            else:
                dat = np.squeeze(io.imread(filename).astype(np.float32))
            if param.channeltype == 'multi':
                dat = self.splitchannel(dat)

            dat = (dat-param.ccd_offset)*param.gain
            imageraw.append(dat)
        imagesall = np.stack(imageraw)

        return imagesall

    def loadmat(self,filelist):
        param = self.param
        imageraw = []
        for filename in filelist:
            print(filename)
            fdata = h5.File(filename,'r')
            if param.varname:
                name = [param.varname]
            else:
                name = list(fdata.keys())       
            try:
                name.remove('metadata')
            except:
                pass
            try:
                name.remove('#refs#')
            except:
                pass
                        
            if param.channeltype == 'single':
                dat = np.squeeze(np.array(fdata.get(name[0])).astype(np.float32))
            else:
                if len(name)>1:
                    dat = []
                    for ch in name:            
                        datai = np.squeeze(np.array(fdata.get(ch)).astype(np.float32))
                        dat.append(datai)
                    dat = np.squeeze(np.stack(dat))
                else:
                    dat = np.squeeze(np.array(fdata.get(name[0])).astype(np.float32))
                    dat = self.splitchannel(dat)

            dat = (dat-param.ccd_offset)*param.gain
            imageraw.append(dat)
        imagesall = np.stack(imageraw)

        return imagesall
        
    def loadh5(self,filelist):
        # currently only for smlm data
        param = self.param
        imageraw = []

        for filename in filelist:
            f = h5.File(filename,'r')
            k = list(f.keys())
            gname = ''
            while len(k)==1:
                gname += k[0]+'/'
                k = list(f[gname].keys())
            datalist = list(f[gname].keys())
            try:
                dat = np.squeeze(np.array(f.get(gname+datalist[0])).astype(np.float32))
            except:
                dat = np.squeeze(np.array(f.get(gname+datalist[0]+'/'+datalist[0])).astype(np.float32))
            dat = dat[param.insitu.frame_range[0]:param.insitu.frame_range[1]]
            dat = (dat-param.ccd_offset)*param.gain
            imageraw.append(dat)
        imagesall = np.stack(imageraw)

        return imagesall


    def loadczi(self,filelist):
        param = self.param
        imageraw = []
        for filename in filelist:
            dat = np.squeeze(czi.imread(filename).astype(np.float32))
            dat = (dat-param.ccd_offset)*param.gain
            imageraw.append(dat)
        imagesall = np.stack(imageraw)

        return imagesall

    def splitchannel(self,dat):
        param = self.param
        if param.dual.channel_arrange:
            if param.dual.channel_arrange == 'up-down':
                cc = dat.shape[-2]//2
                if param.dual.mirrortype == 'up-down':
                    dat = np.stack([dat[:,:-cc],np.flip(dat[:,cc:],axis=-2)])
                elif param.dual.mirrortype == 'left-right':
                    dat = np.stack([dat[:,:-cc],np.flip(dat[:,cc:],axis=-1)])
                else:
                    dat = np.stack([dat[:,:-cc],dat[:,cc:]])
            else:
                cc = dat.shape[-1]//2
                if param.dual.mirrortype == 'up-down':
                    dat = np.stack([dat[...,:-cc],np.flip(dat[...,cc:],axis=-2)])
                elif param.dual.mirrortype == 'left-right':
                    dat = np.stack([dat[...,:-cc],np.flip(dat[...,cc:],axis=-1)])  
                else:
                    dat = np.stack([dat[...,:-cc],dat[...,cc:]])    
        if param.multi.channel_size:
            roisz = param.multi.channel_size
            xdiv = list(range(0,dat.shape[-1],roisz[-1]))
            ydiv = list(range(0,dat.shape[-2],roisz[-2]))
            im = []
            for yd in ydiv[:-1]:
                for xd in xdiv[:-1]:
                    im.append(dat[...,yd:yd+roisz[-2],xd:xd+roisz[-1]])

            dat = np.stack(im)


        return dat
