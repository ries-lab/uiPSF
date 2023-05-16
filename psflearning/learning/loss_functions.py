"""
Copyright (c) 2022      Ries Lab, EMBL, Heidelberg, Germany
All rights reserved     

@author: Sheng Liu, Jonas Hellgoth
"""

import tensorflow as tf

import numpy as np


def mse_real(model,data,variables=None,mu=None,w=None):
    mydiff = model-data
    mydiff = mydiff[:,1:-1]
    data = data[:,1:-1]
    model = model[:,1:-1]
    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data) 
    mse_norm2 = tf.reduce_mean(tf.reduce_sum(tf.square(mydiff),axis=(-3,-2,-1)) / tf.math.reduce_max(tf.square(data),axis=(-3,-2,-1)))/data.shape[-3]*200

    LL = (model-data-data*tf.math.log(model)+data*tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    f = variables[3]  
    gxymean = tf.reduce_mean(tf.abs(variables[4]))   
    bg = variables[1]
    intensity = variables[2]
    s = tf.math.reduce_sum(tf.math.square(f[0]-f[1])+tf.math.square(f[-1]-f[-2]))
   
    dfz = tf.math.square(tf.experimental.numpy.diff(f, n = 1, axis = -3))
    dfz = tf.reduce_sum(dfz)
   
    Imin = tf.reduce_sum(tf.math.square(tf.math.minimum(f,0)))
    bgmin = tf.reduce_sum(tf.math.square(tf.math.minimum(bg,0)))
    intensitymin = tf.reduce_sum(tf.math.square(tf.math.minimum(intensity,0)))
    fsz = f.shape
    ccz = fsz[0]//2
    #wd = tf.math.minimum(cc,10)
    # g0 = f
    # g = g0[:,1:-1,1:-1]
    # Imin1 = tf.reduce_sum(tf.math.square(g0))-tf.reduce_sum(tf.math.square(g))
    
    # Inorm = tf.math.abs(tf.math.reduce_sum(f))
    Inorm = tf.reduce_mean(tf.math.square(tf.math.reduce_sum(f,axis=(-1,-2))-tf.math.reduce_sum(f)/fsz[0]))

    loss = mse_norm1*w[0] + mse_norm2*w[1] + w[2]*dfz + s*w[3] + w[4]*Imin*mu + bgmin*w[5]*mu  + intensitymin*w[6]*mu + Inorm*w[7]*mu + gxymean*w[8]
    #loss = LL*w[0] + w[2]*dfz + s*w[3] + w[4]*Imin*mu + bgmin*w[5]*mu  + intensitymin*w[6]*mu + Inorm*w[7]*mu + gxymean*w[8]

    return loss

    
def mse_real_4pi(model,data,variables=None,mu=None,w=None):
    mydiff = model-data
    mydiff = mydiff[:,:,1:-1]
    data = data[:,:,1:-1]
    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data) 
    #mse_norm = tf.reduce_mean(tf.reduce_sum(tf.square(mydiff),axis=(-3,-2,-1)) / tf.math.reduce_max(tf.square(data),axis=(-3,-2,-1)))*10
    mse_norm2 = tf.reduce_mean(tf.reduce_sum(tf.square(mydiff),axis=(-3,-2,-1)) / tf.math.reduce_max(tf.square(data),axis=(-3,-2,-1)))/data.shape[-3]*200

    f = variables[4]    
    bg = variables[1] 
    intensity = variables[2]
    gxymean = tf.reduce_mean(tf.abs(variables[-1]))   

    s = tf.math.reduce_sum(tf.math.square(f[0]-f[1])+tf.math.square(f[-1]-f[-2]))
    fsz = f.shape
    Areal = variables[5]
    Aimg = variables[6]
    A = tf.complex(Areal,Aimg)

    dfz = tf.math.square(tf.experimental.numpy.diff(f, n = 1, axis = -3))+tf.math.square(tf.experimental.numpy.diff(Areal, n = 1, axis = -3))+tf.math.square(tf.experimental.numpy.diff(Aimg, n = 1, axis = -3))
    dfz = tf.reduce_sum(dfz)

    s1 = tf.math.reduce_sum(tf.math.square(Areal[0]-Areal[1])+tf.math.square(Areal[-1]-Areal[-2]))
    s2 = tf.math.reduce_sum(tf.math.square(Aimg[0]-Aimg[1])+tf.math.square(Aimg[-1]-Aimg[-2]))
    Imin = tf.reduce_sum(tf.math.square(tf.math.minimum(f-2*tf.math.abs(A),0)))
    bgmin = tf.reduce_sum(tf.math.square(tf.math.minimum(bg,0)))
    intensitymin = tf.reduce_sum(tf.math.square(tf.math.minimum(intensity,0)))
    Inorm = tf.reduce_mean(tf.math.square(tf.math.reduce_sum(f,axis=(-1,-2))-tf.math.reduce_sum(f)/fsz[0]))
    loss = mse_norm1*w[0] + mse_norm2*w[1] + w[2]*dfz + (s+s1+s2)*w[3] + w[4]*Imin*mu + bgmin*mu*w[5] + intensitymin*w[6]*mu + Inorm*w[7]*mu + gxymean*w[8]
  
    return loss


def mse_real_4pi_All(model,data,loss_func,variables=None,mu=None,w=None):
    varsize = len(variables)
    var = [None]*(varsize-1)
    loss = 0.0
    for i in range(0,model.shape[0]):
        for j in range(1,varsize-1):
            var[j] = variables[j][i]
        var[0] = variables[0]
        loss += loss_func(model[i],data[i],var,mu,w)
    
    return loss

def mse_real_All(model,data,loss_func,variables=None,mu=None,w=None):
    varsize = len(variables)
    var = [None]*(varsize-1)
    loss = 0.0
    for i in range(0,model.shape[0]):
        for j in range(1,varsize-1):
            var[j] = variables[j][i]
        var[0] = variables[0]
        loss += loss_func(model[i],data[i],var,mu,w)
    
    return loss



def mse_real_pupil(model,data,variables=None,mu=None,w=None):
    mydiff = model-data

    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data) 
    
    mse_norm2 = tf.reduce_mean(tf.reduce_sum(tf.square(mydiff),axis=(-3,-2,-1)) / tf.math.reduce_max(tf.square(data),axis=(-3,-2,-1)))/data.shape[-3]*200


    #LL = (model-data*tf.math.log(model))
    LL = (model-data-data*tf.math.log(model)+data*tf.math.log(data))


    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)]) 
    pupilR = variables[3]    
    pupilI = variables[4] 
    bg = variables[1]
    intensity = variables[2]
    gxymean = tf.reduce_mean(tf.abs(variables[-1]))   


    dfxy1 = tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilI, n = 1, axis = -1)))+ tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilI, n = 1, axis = -2)))
    dfxy2 = tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilR, n = 1, axis = -1)))+ tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilR, n = 1, axis = -2)))
    dfxy = dfxy1+dfxy2

 
    bgmin = tf.reduce_sum(tf.math.square(tf.math.minimum(bg,0)))
    intensitymin = tf.reduce_sum(tf.math.square(tf.math.minimum(intensity,0)))
   

    #loss = mse_norm1*w[0] + mse_norm2*w[1] + bgmin*w[5]*mu  + intensitymin*w[6]*mu + dfxy*w[2] 
    loss = LL*w[0] + bgmin*w[5]*mu  + intensitymin*w[6]*mu + dfxy*w[2] + gxymean*w[8]

    return loss



def mse_pupil_4pi(model,data,variables=None,mu=None,w=None):
    mydiff = model-data
    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)     
    mse_norm2 = tf.reduce_mean(tf.reduce_sum(tf.square(mydiff),axis=(-3,-2,-1)) / tf.math.reduce_max(tf.square(data),axis=(-3,-2,-1)))/data.shape[-3]*200

    #LL = (model-data*tf.math.log(model))
    LL = (model-data-data*tf.math.log(model)+data*tf.math.log(data))

    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)]) 

    pupilR1 = variables[4]    
    pupilI1 = variables[5] 
    pupilR2 = variables[6]    
    pupilI2 = variables[7] 
    bg = variables[1]
    intensity = variables[2]
    alpha = variables[9]
    wavelength = variables[10]
    gxymean = tf.reduce_mean(tf.abs(variables[-1]))   

    dfxy1 = tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilI1, n = 1, axis = -1)))+ tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilI1, n = 1, axis = -2)))
    dfxy2 = tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilR1, n = 1, axis = -1)))+ tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilR1, n = 1, axis = -2)))
    dfxy3 = tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilI2, n = 1, axis = -1)))+ tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilI2, n = 1, axis = -2)))
    dfxy4 = tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilR2, n = 1, axis = -1)))+ tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilR2, n = 1, axis = -2)))
    dfxy = dfxy1+dfxy2+dfxy3+dfxy4
 
    bgmin = tf.reduce_sum(tf.math.square(tf.math.minimum(bg,0)))
    intensitymin = tf.reduce_sum(tf.math.square(tf.math.minimum(intensity,0)))
    alphamin = tf.reduce_sum(tf.math.square(tf.math.minimum(alpha,0)))

    #loss = mse_norm1*w[0] + mse_norm2*w[1] + bgmin*w[5]*mu  + intensitymin*w[6]*mu + dfxy*w[2] + alphamin*w[4]*mu
    loss = LL*w[0] + bgmin*w[5]*mu  + intensitymin*w[6]*mu + dfxy*w[2] + alphamin*w[4]*mu + gxymean*w[8]

    return loss



def mse_real_zernike(model,data,variables=None,mu=None,w=None):
    mydiff = model-data

    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)     
    mse_norm2 = tf.reduce_mean(tf.reduce_sum(tf.square(mydiff),axis=(-3,-2,-1)) / tf.math.reduce_max(tf.square(data),axis=(-3,-2,-1)))/data.shape[-3]*200

    #LL = (model-data*tf.math.log(model))
    LL = (model-data-data*tf.math.log(model)+data*tf.math.log(data))

    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    bg = variables[1]
    intensity = variables[2]
    zcoeff = variables[3]
    wavelength = variables[5]
    gxymean = tf.reduce_mean(tf.abs(variables[-1]))   
    
    bgmin = tf.reduce_sum(tf.math.square(tf.math.minimum(bg,0)))
    intensitymin = tf.reduce_sum(tf.math.square(tf.math.minimum(intensity,0)))
    g1 = tf.reduce_sum(tf.square(zcoeff[0][1:]))
    g2 = tf.reduce_sum(tf.square(zcoeff[1]))

    #loss = mse_norm1*w[0] + mse_norm2*w[1] + bgmin*w[5]*mu  + intensitymin*w[6]*mu + (g1+g2)*w[2]
    loss = LL*w[0] +  bgmin*w[5]*mu  + intensitymin*w[6]*mu + gxymean*w[8]

    return loss


def mse_zernike_4pi(model,data,variables=None,mu=None,w=None):
    mydiff = model-data

    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)     
    mse_norm2 = tf.reduce_mean(tf.reduce_sum(tf.square(mydiff),axis=(-3,-2,-1)) / tf.math.reduce_max(tf.square(data),axis=(-3,-2,-1)))/data.shape[-3]*200

    #LL = (model-data*tf.math.log(model))
    LL = (model-data-data*tf.math.log(model)+data*tf.math.log(data))

    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)]) 
    bg = variables[1]
    intensity = variables[2]
    intensity_phase = variables[3]
    zcoeff1 = variables[4]
    zcoeff2 = variables[5]
    alpha = variables[7]
    wavelength = variables[8]
    posd = variables[9]
    gxymean = tf.reduce_mean(tf.abs(variables[-1]))   

    bgmin = tf.reduce_sum(tf.math.square(tf.math.minimum(bg,0)))
    intensitymin = tf.reduce_sum(tf.math.square(tf.math.minimum(intensity,0)))
    alphamin = tf.reduce_sum(tf.math.square(tf.math.minimum(alpha,0)))

    g1 = tf.reduce_sum(tf.abs(zcoeff1[1][1:]))
    g2 = tf.reduce_sum(tf.abs(zcoeff1[0][1:]))*2 + tf.reduce_sum(tf.abs(zcoeff2[0][1:]))*2
    g3 = tf.reduce_sum(tf.abs(zcoeff2[1][1:]))
    g4 = tf.reduce_sum(tf.square(posd))*2

    #loss = mse_norm1*w[0] + mse_norm2*w[1] + bgmin*w[5]*mu  + intensitymin*w[6]*mu + alphamin*w[4]*mu + (g1+g2)*w[2]
    loss = LL*w[0] + bgmin*w[5]*mu  + intensitymin*w[6]*mu + alphamin*w[4]*mu + gxymean*w[8]

    return loss

def mse_zernike_4pi_smlm(model,data,variables=None,mu=None,w=None):
    mydiff = model-data

    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)     
    mse_norm2 = tf.reduce_mean(tf.reduce_sum(tf.square(mydiff),axis=(-2,-1)) / tf.math.reduce_max(tf.square(data),axis=(-2,-1)))*200

    #LL = (model-data*tf.math.log(model))
    LL = (model-data-data*tf.math.log(model)+data*tf.math.log(data))

    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)]) 
    bg = variables[1]
    intensity = variables[2]
    intensity_phase = variables[3]
    zcoeff1 = variables[4]
    zcoeff2 = variables[5]
    alpha = variables[7]

    bgmin = tf.reduce_sum(tf.math.square(tf.math.minimum(bg,0)))
    intensitymin = tf.reduce_sum(tf.math.square(tf.math.minimum(intensity,0)))
    alphamin = tf.reduce_sum(tf.math.square(tf.math.minimum(alpha,0)))

    g1 = tf.reduce_sum(tf.abs(zcoeff1[1][1:]))
    g2 = tf.reduce_sum(tf.abs(zcoeff1[0][1:]))*2 + tf.reduce_sum(tf.abs(zcoeff2[0][1:]))*2
    g3 = tf.reduce_sum(tf.abs(zcoeff2[1][1:]))
    g4 = tf.reduce_sum(tf.square(posd))*2

    #loss = mse_norm1*w[0] + mse_norm2*w[1] + bgmin*w[5]*mu  + intensitymin*w[6]*mu + alphamin*w[4]*mu + (g1+g2)*w[2]
    loss = LL*w[0] + bgmin*w[5]*mu  + intensitymin*w[6]*mu + alphamin*w[4]*mu 

    return loss

def mse_real_zernike_FD(model,data,variables=None,mu=None,w=None):
    mydiff = model-data

    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)     
    mse_norm2 = tf.reduce_mean(tf.reduce_sum(tf.square(mydiff),axis=(-3,-2,-1)) / tf.math.reduce_max(tf.square(data),axis=(-3,-2,-1)))/data.shape[-3]*200

    #LL = (model-data*tf.math.log(model))
    LL = (model-data-data*tf.math.log(model)+data*tf.math.log(data))

    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    bg = variables[1]
    intensity = variables[2]
    gxymean = tf.reduce_mean(tf.abs(variables[-1]))   

    bgmin = tf.reduce_sum(tf.math.square(tf.math.minimum(bg,0)))
    intensitymin = tf.reduce_sum(tf.math.square(tf.math.minimum(intensity,0)))

    Zmap = variables[3]
    dfxy = tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(Zmap, n = 1, axis = -1)))+ tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(Zmap, n = 1, axis = -2)))

    #loss = mse_norm1*w[0] + mse_norm2*w[1] + bgmin*w[5]*mu  + intensitymin*w[6]*mu + dfxy*w[2] 
    loss = LL*w[0] + bgmin*w[5]*mu  + intensitymin*w[6]*mu + dfxy*w[2] + gxymean*w[8]

    return loss

def mse_real_zernike_FD_smlm(model,data,variables=None,mu=None,w=None):
    mydiff = model-data

    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)     
    mse_norm2 = tf.reduce_mean(tf.reduce_mean(tf.square(mydiff),axis=(-2,-1)) / tf.math.reduce_max(tf.square(data),axis=(-2,-1)))*200
    LL = (model-data-data*tf.math.log(model)+data*tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    bg = variables[1]
    intensity = variables[2]
    stagepos = variables[5]
    zpos = variables[0][:,0,...]
    bgmin = tf.reduce_mean(tf.math.square(tf.math.minimum(bg,0)))
    zmin = tf.reduce_mean(tf.math.square(tf.math.minimum(zpos,0))) + tf.reduce_mean(tf.math.square(tf.math.minimum(stagepos,0)))
    intensitymin = tf.reduce_mean(tf.math.square(tf.math.minimum(intensity,0)))

    Zmap = variables[3]
    dfxy = tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(Zmap, n = 1, axis = -1)))+ tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(Zmap, n = 1, axis = -2)))

    #loss = mse_norm1*w[0] + mse_norm2*w[1] + bgmin*w[5]*mu  + intensitymin*w[6]*mu
    loss = LL*w[0]  + bgmin*w[5]*mu  + intensitymin*w[6]*mu + dfxy*w[2] + zmin*w[4]*mu
    
    return loss

def mse_real_zernike_smlm(model,data,variables=None,mu=None,w=None):
    mydiff = model-data

    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)     
    mse_norm2 = tf.reduce_mean(tf.reduce_mean(tf.square(mydiff),axis=(-2,-1)) / tf.math.reduce_max(tf.square(data),axis=(-2,-1)))*200
    LL = (model-data-data*tf.math.log(model)+data*tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    bg = variables[1]
    intensity = variables[2]
    zcoeff = variables[3]
    stagepos = variables[5]
    zpos = variables[0][:,0,...]
    bgmin = tf.reduce_mean(tf.math.square(tf.math.minimum(bg,0)))
    zmin = tf.reduce_mean(tf.math.square(tf.math.minimum(zpos,0))) + tf.reduce_mean(tf.math.square(tf.math.minimum(stagepos,0)))
    intensitymin = tf.reduce_mean(tf.math.square(tf.math.minimum(intensity,0)))

    g1 = tf.reduce_sum(tf.square(zcoeff[0][1:]))
    g2 = tf.reduce_sum(tf.square(zcoeff[1]))

    #loss = mse_norm1*w[0] + mse_norm2*w[1] + bgmin*w[5]*mu  + intensitymin*w[6]*mu
    loss = LL*w[0]  + bgmin*w[5]*mu  + intensitymin*w[6]*mu + (g1+g2)*w[2] + zmin*w[4]*mu
    
    return loss

def mse_real_pupil_smlm(model,data,variables=None,mu=None,w=None):
    mydiff = model-data

    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)     
    mse_norm2 = tf.reduce_mean(tf.reduce_mean(tf.square(mydiff),axis=(-2,-1)) / tf.math.reduce_max(tf.square(data),axis=(-2,-1)))*200
    LL = (model-data-data*tf.math.log(model)+data*tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    bg = variables[1]
    intensity = variables[2]
    pupilR = variables[3]    
    pupilI = variables[4] 
    stagepos = variables[6]
    zpos = variables[0][:,0,...]
    bgmin = tf.reduce_mean(tf.math.square(tf.math.minimum(bg,0)))
    zmin = tf.reduce_mean(tf.math.square(tf.math.minimum(zpos,0))) + tf.reduce_mean(tf.math.square(tf.math.minimum(stagepos,0)))
    intensitymin = tf.reduce_mean(tf.math.square(tf.math.minimum(intensity,0)))

    dfxy1 = tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilI, n = 1, axis = -1)))+ tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilI, n = 1, axis = -2)))
    dfxy2 = tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilR, n = 1, axis = -1)))+ tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilR, n = 1, axis = -2)))
    dfxy = dfxy1+dfxy2

    mu = np.min([mu,1.0e30])
    #loss = mse_norm1*w[0] + mse_norm2*w[1] + bgmin*w[5]*mu  + intensitymin*w[6]*mu
    loss = LL*w[0]  + bgmin*w[5]*mu  + intensitymin*w[6]*mu + dfxy*w[2] + zmin*w[4]*mu
    
    return loss