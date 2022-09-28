
"""
Copyright (c) 2022      Ries Lab, EMBL, Heidelberg, Germany
All rights reserved     

@author: Sheng Liu
"""
#%%
import ctypes
import numpy.ctypeslib as ctl
import numpy as np
import h5py as h5
from .utilities import psf2cspline_np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import tensorflow as tf
#%%
class localizationlib:
    def __init__(self,usecuda=False):
        thispath = os.path.dirname(os.path.abspath(__file__))
        if sys.platform.startswith('win'):
            dllpath_cpu_astM = thispath+'/source/mleFit_LM_DLL/x64/Release/CPUmleFit_LM_MultiChannel.dll'
            dllpath_gpu_astM = thispath+'/source/mleFit_LM_DLL/x64/Release/GPUmleFit_LM_MultiChannel.dll'
            dllpath_cpu_4pi = thispath+'/source/mleFit_LM_DLL/x64/Release/CPUmleFit_LM_4Pi.dll'
            dllpath_gpu_4pi = thispath+'/source/mleFit_LM_DLL/x64/Release/GPUmleFit_LM_4Pi.dll'
            dllpath_cpu_ast = thispath+'/source/mleFit_LM_DLL/x64/Release/CPUmleFit_LM.dll'
            dllpath_gpu_ast = thispath+'/source/mleFit_LM_DLL/x64/Release/GPUmleFit_LM.dll'
            
            if tf.test.gpu_device_name():
                lib_gpu_astM = ctypes.CDLL(dllpath_gpu_astM)            
                lib_gpu_4pi = ctypes.CDLL(dllpath_gpu_4pi)            
                lib_gpu_ast = ctypes.CDLL(dllpath_gpu_ast)
            else:
                usecuda = False
        elif sys.platform.startswith('darwin'):
            usecuda = False
            dllpath_cpu_ast = thispath+'/source/mleFit_LM_dylib/mac/Build/Products/Release/libCPUmleFit_LM.dylib'
            dllpath_cpu_astM = thispath+'/source/mleFit_LM_dylib/mac/Build/Products/Release/libCPUmleFit_LM_MultiChannel.dylib'
            dllpath_cpu_4pi = thispath+'/source/mleFit_LM_dylib/mac/Build/Products/Release/libCPUmleFit_LM_4Pi.dylib'

        lib_cpu_astM = ctypes.CDLL(dllpath_cpu_astM)        
        lib_cpu_4pi = ctypes.CDLL(dllpath_cpu_4pi)        
        lib_cpu_ast = ctypes.CDLL(dllpath_cpu_ast)
       
    

        if usecuda:
            self._mleFit_MultiChannel = lib_gpu_astM.GPUmleFit_MultiChannel
            self._mleFit_4Pi = lib_gpu_4pi.GPUmleFit_LM_4Pi
            self._mleFit = lib_gpu_ast.GPUmleFit_LM
        else:
            self._mleFit_MultiChannel = lib_cpu_astM.CPUmleFit_MultiChannel
            self._mleFit_4Pi = lib_cpu_4pi.CPUmleFit_LM_4Pi
            self._mleFit = lib_cpu_ast.CPUmleFit_LM
        
        
        self._mleFit_4Pi.argtypes = [
            ctl.ndpointer(np.float32), # data
            ctl.ndpointer(np.int32),   # shared
            ctypes.c_int32,          # iterations
            ctl.ndpointer(np.float32), # spline_coeff
            ctl.ndpointer(np.float32), # dTAll
            ctl.ndpointer(np.float32), # phiA
            ctl.ndpointer(np.float32), # init_z
            ctl.ndpointer(np.float32), # initphase
            ctl.ndpointer(np.int32),   # datasize
            ctl.ndpointer(np.int32),    # spline_size
            ctl.ndpointer(np.float32), # P
            ctl.ndpointer(np.float32), # CRLB
            ctl.ndpointer(np.float32) # LL

        ]

        self._mleFit_MultiChannel.argtypes = [
            ctl.ndpointer(np.float32), # data
            ctypes.c_int32,          # fittype
            ctl.ndpointer(np.int32),   # shared
            ctypes.c_int32,          # iterations
            ctl.ndpointer(np.float32), # spline_coeff
            ctl.ndpointer(np.float32), # dTAll
            ctl.ndpointer(np.float32), # varim
            ctl.ndpointer(np.float32), # init_z
            ctl.ndpointer(np.int32),   # datasize
            ctl.ndpointer(np.int32),    # spline_size
            ctl.ndpointer(np.float32), # P
            ctl.ndpointer(np.float32), # CRLB
            ctl.ndpointer(np.float32) # LL
        ]

        self._mleFit.argtypes = [
            ctl.ndpointer(np.float32), # data
            ctypes.c_int32,          # fittype
            ctypes.c_int32,          # iterations
            ctl.ndpointer(np.float32), # spline_coeff
            ctl.ndpointer(np.float32), # varim
            ctypes.c_float,             # init_z
            ctl.ndpointer(np.int32),   # datasize
            ctl.ndpointer(np.int32),    # spline_size
            ctl.ndpointer(np.float32), # P
            ctl.ndpointer(np.float32), # CRLB
            ctl.ndpointer(np.float32) # LL
        ]



    def loc_ast_dual(self,psf_data,I_model,pixelsize_z,cor,imgcenter,T,initz=None,plot=True, start_time = None):

        rsz = psf_data.shape[-1]
        Nbead = cor.shape[1]
        Nchannel = cor.shape[0]
        Nz = psf_data.shape[-3]
        Nfit = Nbead*Nz
        Nparam = 5
        offset = np.min(I_model)
        Iall = []
        Imd = I_model-offset
        normf = np.max(np.median(np.sum(Imd,axis = (-1,-2)),axis=-1))
        Imd = Imd/normf
        pbar = tqdm(total=Nchannel,desc='4/6: calculating spline coefficients',bar_format = "{desc}: {n_fmt}/{total_fmt} [{elapsed}s] {rate_fmt} {postfix[0]}{postfix[1][time]:>4.2f}s",postfix=["total time: ", dict(time=start_time)])
        
       
        for i in range(Nchannel):     
                                      
            coeff = psf2cspline_np(Imd[i])
            Iall.append(coeff)
            
            pbar.postfix[1]['time'] = start_time+pbar._time()-pbar.start_t    
            pbar.update(1)
            
        toc = pbar.postfix[1]['time']    
        pbar.close  
        Iall = np.stack(Iall).astype(np.float32)
        data = psf_data.reshape((Nchannel,Nfit,rsz,rsz))
        bxsz = np.min((rsz,20))
        data = data[:,:,rsz//2-bxsz//2:rsz//2+bxsz//2,rsz//2-bxsz//2:rsz//2+bxsz//2].astype(np.float32)
        data = np.maximum(data,0.0)

        cor1 = np.concatenate((cor[0],np.ones((Nbead,1))),axis=1)
        T1 = np.concatenate((np.expand_dims(np.eye(3),axis=0),T),axis=0)
        dx1 = np.zeros((Nbead,Nchannel))
        dy1 = np.zeros((Nbead,Nchannel))
        for i in range(Nchannel):
            cor2 = np.matmul(cor1-imgcenter,T1[i])+imgcenter
            dy1[:,i] = cor2[:,0]-cor[i][:,0]
            dx1[:,i] = cor2[:,1]-cor[i][:,1]
        dTS = np.zeros((Nbead,Nz,Nchannel*2,Nparam))
        dTS[:,:,0:Nchannel,0] = np.expand_dims(dx1,axis=1)
        dTS[:,:,0:Nchannel,1] = np.expand_dims(dy1,axis=1)
        dTS[:,:,Nchannel:]=1
        dTS = dTS.reshape((Nfit,Nchannel*2,Nparam)).astype(np.float32)
        shared = np.array([1,1,1,1,0])
        sharedA = np.repeat(np.expand_dims(shared,axis=0),Nfit,axis = 0).astype(np.int32)
        
        ccz = Iall.shape[-3]//2
        if initz is None:
            initz = np.linspace(-Nz*pixelsize_z/2,Nz*pixelsize_z/2,np.int32(Nz*pixelsize_z/0.5))*0.8/pixelsize_z+ccz
        else:
            initz = np.array(initz)*0.5/pixelsize_z+ccz
        zstart = np.repeat(np.expand_dims(initz,axis=1),Nfit,axis=1).astype(np.float32)

        datasize = np.array(np.flip(data.shape)).astype(np.int32)
        splinesize = np.array(np.flip(Iall.shape)).astype(np.int32)
        varim = np.array((0)).astype(np.float32)
        Pk = np.zeros((Nparam+1+(Nchannel-1)*(Nparam-np.sum(shared)),Nfit)).astype(np.float32)
        CRLBk = np.zeros((Nparam+(Nchannel-1)*(Nparam-np.sum(shared)),Nfit)).astype(np.float32)
        LLk = np.zeros((Nfit)).astype(np.float32)
        fittype = np.int32(2)
        iterations = np.int32(100)
        P = np.zeros((Nparam+1+(Nchannel-1)*(Nparam-np.sum(shared)),Nfit)).astype(np.float32)
        CRLB = np.zeros((Nparam+(Nchannel-1)*(Nparam-np.sum(shared)),Nfit)).astype(np.float32)
        LL = np.zeros((Nfit)).astype(np.float32)-1e10
        pbar = tqdm(total=len(zstart),desc='5/6: localization',bar_format = "{desc}: {n_fmt}/{total_fmt} [{elapsed}s] {rate_fmt} {postfix[0]}{postfix[1][time]:>4.2f}s",postfix=["total time: ", dict(time=toc)])

        for z0 in zstart:
            
            self._mleFit_MultiChannel(data,fittype,sharedA,iterations,Iall,dTS,
                                         varim,z0,datasize,splinesize,Pk,CRLBk,LLk)
            mask = (LLk-LL)>1e-4
            LL[mask] = LLk[mask]
            P[:,mask] = Pk[:,mask]
            CRLB[:,mask] = CRLBk[:,mask]
            
            pbar.postfix[1]['time'] = toc+pbar._time()-pbar.start_t    
            pbar.update(1)

        toc = pbar.postfix[1]['time']  
        pbar.close()

        zf = P[2].reshape((Nbead,Nz))
        zg = np.linspace(0,Nz-1,Nz)
        zf = zf-np.median(zf-zg,axis=1,keepdims=True)
        zdiff = zf-zg

        xf = P[1].reshape((Nbead,Nz))
        xf = xf-np.median(xf,axis=1,keepdims=True)

        yf = P[0].reshape((Nbead,Nz))
        yf = yf-np.median(yf,axis=1,keepdims=True)


        if Nz>4:
            zind = range(2,Nz-2,1)
        else:
            zind = range(0,Nz,1)
      
        zdiff = zdiff-np.mean(zdiff[:,zind],axis=1,keepdims=True)
        msez = np.mean(np.square((np.median(zf-zg,axis=0)-(zf-zg))[:,zind]),axis=1)
        msezRatio =msez/np.median(msez)
        if plot:
            fig = plt.figure(figsize=[12,6])
            ax = fig.add_subplot(1,2,1)
            plt.plot(zf.transpose(),color=(0.6,0.6,0.6))
            plt.plot(np.linspace(0,Nz-1,Nz))
            ax = fig.add_subplot(1,2,2)
            plt.plot((zdiff).transpose(),color=(0.6,0.6,0.6))
            plt.plot(np.median(zdiff,axis=0),color='r')
            plt.plot(zg-zg,color='k')
            ax.set_ylim([-0.1,0.1]/np.array([pixelsize_z]))
            plt.show()

        loc_dict = dict(x=xf,y=yf,z=zf)

        return P, CRLB, LL, Iall, msezRatio,toc, loc_dict


    def loc_4pi(self,psf_data,I_model,A_model,pixelsize_z,cor,imgcenter,T,zT,initz=None,initphi=None,plot=True,start_time=None):
        rsz = psf_data.shape[-1]
        Nbead = cor.shape[1]
        Nchannel = cor.shape[0]
        Nz = psf_data.shape[-3]
        if len(psf_data.shape)>5:
            Nphase = psf_data.shape[-4]
        else:
            Nphase = 1
        Nfit = Nbead*Nz*Nphase
        Nparam = 6
        offset = np.min(I_model-2*np.abs(A_model))
        Imd = I_model-offset
        normf = np.max(np.median(np.sum(Imd[:,1:-1],axis = (-1,-2)),axis=-1))*2.0
        Imd = Imd/normf
        Amd = A_model/normf
        pbar = tqdm(total=Nchannel,desc='4/6: calculating spline coefficients',bar_format = "{desc}: {n_fmt}/{total_fmt} [{elapsed}s] {rate_fmt} {postfix[0]}{postfix[1][time]:>4.2f}s",postfix=["total time: ", dict(time=start_time)])

       
        IABall = []
        for i in range(Nchannel):     
            
            Ii = Imd[i]
            Ai = 2*np.real(Amd[i])
            Bi = -2*np.imag(Amd[i]) 
            IAB = [psf2cspline_np(Ai),psf2cspline_np(Bi),psf2cspline_np(Ii)]  
            IAB = np.stack(IAB)
            IABall.append(IAB)
            
            pbar.postfix[1]['time'] = start_time+pbar._time()-pbar.start_t    
            pbar.update(1)
            
        toc = pbar.postfix[1]['time']
        pbar.close()
        IABall = np.stack(IABall).astype(np.float32)
        data = psf_data.reshape((Nchannel,Nfit,rsz,rsz))
        bxsz = np.min((rsz,20))
        data = data[:,:,rsz//2-bxsz//2:rsz//2+bxsz//2,rsz//2-bxsz//2:rsz//2+bxsz//2].astype(np.float32)
        data = np.maximum(data,0.0)

        cor1 = np.concatenate((cor[0],np.ones((Nbead,1))),axis=1)
        T1 = np.concatenate((np.expand_dims(np.eye(3),axis=0),T),axis=0)
        dx1 = np.zeros((Nbead,Nchannel))
        dy1 = np.zeros((Nbead,Nchannel))
        for i in range(Nchannel):
            cor2 = np.matmul(cor1-imgcenter,T1[i])+imgcenter
            dy1[:,i] = cor2[:,0]-cor[i][:,0]
            dx1[:,i] = cor2[:,1]-cor[i][:,1]
        dTS = np.zeros((Nbead,Nz*Nphase,Nchannel,Nparam))
        dTS[:,:,:,0] = np.expand_dims(dx1,axis=1)
        dTS[:,:,:,1] = np.expand_dims(dy1,axis=1)
        dTS = dTS.reshape((Nfit,Nchannel,Nparam)).astype(np.float32)
        shared = np.array([1,1,1,1,1,1])
        sharedA = np.repeat(np.expand_dims(shared,axis=0),Nfit,axis = 0).astype(np.int32)

        phic = np.array([0,0,0,0])
        phiA = np.repeat(np.expand_dims(phic,axis=0),Nfit,axis = 0).astype(np.float32)
        
        ccz = IABall.shape[-3]//2
        if initz is None:
            initz = np.array([-1,1])*0.15/pixelsize_z+ccz
        else:
            initz = np.array(initz)*0.15/pixelsize_z+ccz
        zstart = np.repeat(np.expand_dims(initz,axis=1),Nfit,axis=1).astype(np.float32)

        if initphi is None:
            initphi = np.array([0,np.pi])
        else:
            initphi = np.array(initphi)
        phi_start = np.repeat(np.expand_dims(initphi,axis=1),Nfit,axis=1).astype(np.float32)

        datasize = np.array(np.flip(data.shape)).astype(np.int32)
        splinesize = np.array(np.flip(IABall.shape)).astype(np.int32)
  
        Pk = np.zeros((Nparam+1+(Nchannel-1)*(Nparam-np.sum(shared)),Nfit)).astype(np.float32)
        CRLBk = np.zeros((Nparam+(Nchannel-1)*(Nparam-np.sum(shared)),Nfit)).astype(np.float32)
        LLk = np.zeros((Nfit)).astype(np.float32)
        iterations = np.int32(100)
        P = np.zeros((Nparam+1+(Nchannel-1)*(Nparam-np.sum(shared)),Nfit)).astype(np.float32)
        CRLB = np.zeros((Nparam+(Nchannel-1)*(Nparam-np.sum(shared)),Nfit)).astype(np.float32)
        LL = np.zeros((Nfit)).astype(np.float32)-1e10

        maxN = 3000
        Nf = np.ceil(Nfit/maxN).astype(np.int32)

        vec = np.linspace(0,Nf*maxN,Nf+1).astype(np.int32)
        vec[-1] = Nfit
        pbar = tqdm(total=len(zstart)*len(phi_start),desc='5/6: localization',bar_format = "{desc}: {n_fmt}/{total_fmt} [{elapsed}s] {rate_fmt} {postfix[0]}{postfix[1][time]:>4.2f}s",postfix=["total time: ", dict(time=toc)])

        for z0 in zstart:
            for phi0 in phi_start:
                
                for i in range(Nf):

                    nfit = vec[i+1]-vec[i]
                    ph = np.zeros((Nparam+1+(Nchannel-1)*(Nparam-np.sum(shared)),nfit)).astype(np.float32)
                    ch = np.zeros((Nparam+(Nchannel-1)*(Nparam-np.sum(shared)),nfit)).astype(np.float32)
                    Lh = np.zeros((nfit)).astype(np.float32)
                    datai = np.copy(data[:,vec[i]:vec[i+1]])
                    sharedi = np.copy(sharedA[vec[i]:vec[i+1]])
                    dts = np.copy(dTS[vec[i]:vec[i+1]])
                    phiAi = np.copy(phiA[vec[i]:vec[i+1]])
                    z0i = np.copy(z0[vec[i]:vec[i+1]])
                    phi0i = np.copy(phi0[vec[i]:vec[i+1]])
                    datsz = np.array(np.flip(datai.shape)).astype(np.int32)
                    self._mleFit_4Pi(datai,sharedi,iterations,IABall,dts,phiAi,z0i,phi0i,datsz,splinesize,ph,ch,Lh)
                    #self._mleFit_4Pi(data,sharedA,iterations,IABall,dTS,phiA,z0,phi0,datasize,splinesize,Pk,CRLBk,LLk)
                    Pk[:,vec[i]:vec[i+1]] = ph
                    CRLBk[:,vec[i]:vec[i+1]] = ch
                    LLk[vec[i]:vec[i+1]] = Lh
                mask = (LLk-LL)>1e-4
                LL[mask] = LLk[mask]
                P[:,mask] = Pk[:,mask]
                CRLB[:,mask] = CRLBk[:,mask]
                
                pbar.postfix[1]['time'] = toc+pbar._time()-pbar.start_t    
                pbar.update(1)
            
        toc = pbar.postfix[1]['time']
        pbar.close()

        
        zf = np.mean(P[4].reshape((Nbead,Nphase,Nz)),axis=1)
        
        zg = np.linspace(0,Nz-1,Nz)
        zf = zf-np.median(zf-zg,axis=1,keepdims=True)
        phif = np.mean(np.unwrap(P[5].reshape((Nbead,Nphase*Nz)),axis=1).reshape((Nbead,Nphase,Nz)),axis=1)*zT/2/np.pi
        phif = phif-np.median(phif-zg,axis=1,keepdims=True)
        zdiff = zf-zg
        phidiff = phif-zg

        xf = np.mean(P[1].reshape((Nbead,Nphase,Nz)),axis=1)
        xf = xf-np.median(xf,axis=1,keepdims=True)

        yf = np.mean(P[0].reshape((Nbead,Nphase,Nz)),axis=1)
        yf = yf-np.median(yf,axis=1,keepdims=True)

        if Nz>4:
            zind = range(2,Nz-2,1)
        else:
            zind = range(0,Nz,1)
    
        zdiff = zdiff-np.mean(zdiff[:,zind],axis=1,keepdims=True)        
        phidiff = phidiff-np.mean(phidiff[:,zind],axis=1,keepdims=True)
        msez = np.mean(np.square((np.median(zf-zg,axis=0)-(zf-zg))[:,zind]),axis=1)
   

        msezRatio =msez/np.median(msez)
        if plot:
            fig = plt.figure(figsize=[12,6])
            ax = fig.add_subplot(2,2,1)
            plt.plot(zf.transpose(),color=(0.6,0.6,0.6))
            plt.plot(np.linspace(0,Nz-1,Nz))
            ax = fig.add_subplot(2,2,2)
            plt.plot(phif.transpose(),color=(0.6,0.6,0.6))
            plt.plot(np.linspace(0,Nz-1,Nz))    
            ax = fig.add_subplot(2,2,3)
            plt.plot((zdiff).transpose(),color=(0.6,0.6,0.6))
            plt.plot(np.median(zdiff,axis=0),color='r')
            plt.plot(zg-zg,color='k')
            ax.set_ylim([-0.1,0.1]/np.array([pixelsize_z]))
            ax.set_title('z')
            ax = fig.add_subplot(2,2,4)
            plt.plot((phidiff).transpose(),color=(0.6,0.6,0.6))
            plt.plot(np.median(phidiff,axis=0),color='r')
            plt.plot(zg-zg,color='k')
            ax.set_ylim([-0.01,0.01]/np.array([pixelsize_z]))
            ax.set_title('phi')
            plt.show()
    

        loc_dict = dict(x=xf,y=yf,z=phif,zast=zf)
        return P, CRLB, LL, IABall, msezRatio, toc, loc_dict


    def loc_ast(self,psf_data,I_model,pixelsize_z,initz=None,plot=True,start_time=0):
        rsz = psf_data.shape[-1]
        Nbead = psf_data.shape[0]
        if len(psf_data.shape)>3:
            Nz = psf_data.shape[-3]
        else:
            Nz = 1
        Nfit = Nbead*Nz
        Nparam = 5
        offset = np.min(I_model)
        Imd = I_model-offset
        normf = np.median(np.sum(Imd,axis = (-1,-2)))
        Imd = Imd/normf
        pbar = tqdm(total=1,desc='4/6: calculating spline coefficients',bar_format = "{desc}: {n_fmt}/{total_fmt} [{elapsed}s] {rate_fmt} {postfix[0]}{postfix[1][time]:>4.2f}s",postfix=["total time: ", dict(time=start_time)])
                  
    
        coeff = psf2cspline_np(Imd)
                         
        pbar.postfix[1]['time'] = start_time+pbar._time()-pbar.start_t    
        pbar.update(1)
        toc = pbar.postfix[1]['time']    
        pbar.close
        
        coeff = coeff.astype(np.float32)
        data = psf_data.reshape((Nfit,rsz,rsz))
        bxsz = np.min((rsz,20))
        data = data[:,rsz//2-bxsz//2:rsz//2+bxsz//2,rsz//2-bxsz//2:rsz//2+bxsz//2].astype(np.float32)
        data = np.maximum(data,0.0)
        
        ccz = coeff.shape[-3]//2
        if initz is None:
            Nzm = Imd.shape[0]
            initz = np.linspace(-Nzm*pixelsize_z/2,Nzm*pixelsize_z/2,np.int32(Nzm*pixelsize_z/0.5))*0.8/pixelsize_z+ccz
            #else:
            #    initz = np.array([-1,0,1])*0.5/pixelsize_z+ccz
        else:
            initz = np.array(initz)*0.5/pixelsize_z+ccz
        zstart = initz.astype(np.float32)

        datasize = np.array(np.flip(data.shape)).astype(np.int32)
        splinesize = np.array(np.flip(coeff.shape)).astype(np.int32)
        varim = np.array((0)).astype(np.float32)
        Pk = np.zeros((Nparam+1,Nfit)).astype(np.float32)
        CRLBk = np.zeros((Nparam,Nfit)).astype(np.float32)
        LLk = np.zeros((Nfit)).astype(np.float32)
        fittype = np.int32(5)
        iterations = np.int32(100)
        P = np.zeros((Nparam+1,Nfit)).astype(np.float32)
        CRLB = np.zeros((Nparam,Nfit)).astype(np.float32)
        LL = np.zeros((Nfit)).astype(np.float32)-1e10
        
        pbar = tqdm(total=len(zstart),desc='5/6: localization',bar_format = "{desc}: {n_fmt}/{total_fmt} [{elapsed}s] {rate_fmt} {postfix[0]}{postfix[1][time]:>4.2f}s",postfix=["total time: ", dict(time=start_time)])

        for z0 in zstart:
            
            self._mleFit(data,fittype,iterations,coeff,varim,z0,datasize,splinesize,Pk,CRLBk,LLk)
            mask = (LLk-LL)>1e-4
            LL[mask] = LLk[mask]
            P[:,mask] = Pk[:,mask]
            CRLB[:,mask] = CRLBk[:,mask]
           
            pbar.postfix[1]['time'] = toc+pbar._time()-pbar.start_t    
            pbar.update(1)

        toc = pbar.postfix[1]['time']     
        pbar.close()


        zf = P[4].reshape((Nbead,Nz))
        xf = P[1].reshape((Nbead,Nz))
        yf = P[0].reshape((Nbead,Nz))

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
            msez = np.mean(np.square((np.median(zf-zg,axis=0)-(zf-zg))[:,zind]),axis=1)
        else:
            zdiff = zf
            msez = np.array([1.0])


        if Nbead == 1:
            msezRatio = np.array([1.0])
        else:
            msezRatio =msez/np.median(msez)
        if plot & (Nz>1):
            fig = plt.figure(figsize=[12,6])
            ax = fig.add_subplot(1,2,1)
            plt.plot(zf.transpose(),color=(0.6,0.6,0.6))
            plt.plot(zg)
            ax.set_title('z')
            ax = fig.add_subplot(1,2,2)
            plt.plot((zdiff).transpose(),color=(0.6,0.6,0.6))
            plt.plot(np.median(zdiff,axis=0),color='r')
            plt.plot(zg-zg,color='k')
            ax.set_ylim([-0.1,0.1]/np.array([pixelsize_z]))
            plt.show()

        loc_dict = dict(x=xf,y=yf,z=zf)

        return P, CRLB, LL, coeff, msezRatio, toc, loc_dict