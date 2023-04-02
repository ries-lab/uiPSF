import matplotlib.pyplot as plt
import numpy as np

def showlearnedparam(f,p):
    if p.channeltype == 'single':
        cor = f.rois.cor
        pos = f.res.pos
        photon = f.res.intensity.transpose()
        bg = f.res.bg
        drift = f.res.drift_rate
    else:
        cor = f.rois.cor[0]
        pos = f.res.channel0.pos
        photon = f.res.channel0.intensity.transpose()
        bg = f.res.channel0.bg
        drift = f.res.channel0.drift_rate
    if p.channeltype == '4pi':
        phi = np.angle(f.res.channel0.intensity)
        photon = np.abs(f.res.channel0.intensity)

    fig = plt.figure(figsize=[16,8])
    ax = fig.add_subplot(2,4,1)
    plt.plot(pos[:,2]-cor[:,1])
    plt.title('x')
    ax = fig.add_subplot(2,4,2)
    plt.plot(pos[:,1]-cor[:,0])
    plt.title('y')
    ax = fig.add_subplot(2,4,3)
    plt.plot(pos[:,0])
    plt.title('z')
    if p.channeltype == '4pi':
        ax = fig.add_subplot(2,4,4)
        plt.plot(phi)
        ax.set_title('phi')
    ax = fig.add_subplot(2,4,5)
    plt.plot(photon)
    plt.title('photon')
    ax = fig.add_subplot(2,4,6)
    plt.plot(bg)
    plt.title('background')
    ax = fig.add_subplot(2,4,7)
    plt.plot(drift)
    plt.title('drift')

    return

def showlearnedparam_insitu(f,p):
    if p.channeltype == 'single':
        cor = f.rois.cor
        pos = f.res.pos
        photon = f.res.intensity
        bg = f.res.bg

    else:
        cor = f.rois.cor[0]
        pos = f.res.channel0.pos
        photon = f.res.channel0.intensity
        bg = f.res.channel0.bg

    fig = plt.figure(figsize=[16,8])
    ax = fig.add_subplot(2,4,1)
    plt.plot(pos[:,1]-cor[:,0],'.')
    plt.title('y')
    ax = fig.add_subplot(2,4,2)
    plt.plot(pos[:,2]-cor[:,1],'.')
    plt.title('x')
    ax = fig.add_subplot(2,4,3)
    plt.plot(pos[:,0],'.')
    plt.title('z')
    ax = fig.add_subplot(2,4,5)
    plt.plot(photon,'.')
    plt.title('photon')
    ax = fig.add_subplot(2,4,6)
    plt.plot(bg,'.')
    plt.title('background')

    return

def showpupil(f,p):
    if p.channeltype == 'single':
        fig = plt.figure(figsize=[12,6])
        pupil = f.res.pupil

        ax = fig.add_subplot(1,2,1)

        plt.imshow(np.abs(pupil))
        plt.title('pupil magnitude')
        ax = fig.add_subplot(1,2,2)

        plt.imshow(np.angle(pupil))
        plt.title('pupil phase')
    elif p.channeltype == 'multi':
        Nchannel = f.rois.psf_data.shape[0]
        fig = plt.figure(figsize=[4*Nchannel,4])
        for i in range(0,Nchannel):
            ax = fig.add_subplot(1,Nchannel,i+1)
            pupil_mag = np.abs(f.res['channel'+str(i)].pupil)
            plt.imshow(pupil_mag)
            plt.axis('off')
            plt.title('pupil magnitude ' + str(i))
        fig = plt.figure(figsize=[4*Nchannel,4])
        for i in range(0,Nchannel):
            ax = fig.add_subplot(1,Nchannel,i+1)
            pupil_phase = np.angle(f.res['channel'+str(i)].pupil)
            plt.imshow(pupil_phase)
            plt.axis('off')
            plt.title('pupil phase ' + str(i))
    elif p.channeltype == '4pi':
        Nchannel = f.rois.psf_data.shape[0]
        fig = plt.figure(figsize=[16,8])
        for i in range(0,Nchannel):
            ax = fig.add_subplot(2,4,i+1)
            pupil_mag = np.abs(f.res['channel'+str(i)].pupil1)
            plt.imshow(pupil_mag)
            plt.axis('off')
            plt.title('top pupil magnitude ' + str(i))
            ax = fig.add_subplot(2,4,i+5)
            pupil_mag = np.abs(f.res['channel'+str(i)].pupil2)
            plt.imshow(pupil_mag)
            plt.axis('off')
            plt.title('bottom pupil magnitude ' + str(i))
        fig = plt.figure(figsize=[16,8])
        for i in range(0,Nchannel):
            ax = fig.add_subplot(2,4,i+1)
            pupil_phase = np.angle(f.res['channel'+str(i)].pupil1)
            plt.imshow(pupil_phase)
            plt.axis('off')
            plt.title('top pupil phase ' + str(i))
            ax = fig.add_subplot(2,4,i+5)
            pupil_phase = np.angle(f.res['channel'+str(i)].pupil2)
            plt.imshow(pupil_phase)
            plt.axis('off')
            plt.title('bottom pupil phase ' + str(i))
    return

def showzernike(f,p):
    if p.channeltype == 'single':
        fig = plt.figure(figsize=[12,6])

        plt.plot(f.res.zernike_coeff.transpose(),'.-')
        plt.xlabel('zernike polynomial')
        plt.ylabel('coefficient')
        plt.legend(['pupil magnitude','pupil phase'])
        
        aperture=np.float32(np.abs(f.res.pupil)>0.0)
        Zk = f.res.zernike_polynomial

        pupil_mag = np.sum(Zk*f.res.zernike_coeff[0].reshape((-1,1,1)),axis=0)*aperture
        pupil_phase = np.sum(Zk[4:]*f.res.zernike_coeff[1][4:].reshape((-1,1,1)),axis=0)*aperture

        fig = plt.figure(figsize=[12,6])
        ax = fig.add_subplot(1,2,1)
        plt.imshow(pupil_mag)
        ax = fig.add_subplot(1,2,2)
        plt.imshow(pupil_phase)
    elif p.channeltype == 'multi':
        Nchannel = f.rois.psf_data.shape[0]
        fig = plt.figure(figsize=[6*Nchannel,6])
        ax1 = fig.add_subplot(2,2,1)
        ax2 = fig.add_subplot(2,2,2)
        for i in range(0,Nchannel):
            line, = ax1.plot(f.res['channel'+str(i)].zernike_coeff[0],'.-')    
            ax2.plot(f.res['channel'+str(i)].zernike_coeff[1],'.-')
            ax1.set_xlabel('zernike polynomial')
            ax1.set_ylabel('coefficient')
            ax1.set_title('pupil magnitude')
            ax2.set_title('pupil phase')
            line.set_label('channel '+str(i))
            ax1.legend()
        aperture=np.float32(np.abs(f.res.channel0.pupil)>0.0)
        Zk = f.res.channel0.zernike_polynomial

        fig = plt.figure(figsize=[4*Nchannel,4])
        for i in range(0,Nchannel):
            ax = fig.add_subplot(1,Nchannel,i+1)
            pupil_mag = np.sum(Zk*f.res['channel'+str(i)].zernike_coeff[0].reshape((-1,1,1)),axis=0)*aperture
            plt.imshow(pupil_mag,)
            plt.axis('off')
            plt.title('pupil magnitude ' + str(i))

        fig = plt.figure(figsize=[4*Nchannel,4])
        for i in range(0,Nchannel):
            ax = fig.add_subplot(1,Nchannel,i+1)
            pupil_phase = np.sum(Zk[4:]*f.res['channel'+str(i)].zernike_coeff[1][4:].reshape((-1,1,1)),axis=0)*aperture
            plt.imshow(pupil_phase)
            plt.axis('off')
            plt.title('pupil phase ' + str(i))
    elif p.channeltype == '4pi':
        Nchannel = f.rois.psf_data.shape[0]
        fig = plt.figure(figsize=[16,8])
        ax1 = fig.add_subplot(2,2,1)
        ax2 = fig.add_subplot(2,2,2)
        ax3 = fig.add_subplot(2,2,3)
        ax4 = fig.add_subplot(2,2,4)
        for i in range(0,Nchannel):
        
            line, = ax1.plot(f.res['channel'+str(i)].zernike_coeff_mag[0],'.-')    
            ax2.plot(f.res['channel'+str(i)].zernike_coeff_phase[0],'.-')
            ax2.set_ylim((-0.6,0.6))
            ax3.plot(f.res['channel'+str(i)].zernike_coeff_mag[1],'.-')
            ax4.plot(f.res['channel'+str(i)].zernike_coeff_phase[1],'.-')
            ax4.set_ylim((-0.6,0.6))
            ax3.set_xlabel('zernike polynomial')
            ax3.set_ylabel('coefficient')
            ax1.set_title('top pupil magnitude')
            ax2.set_title('top pupil phase')
            ax3.set_title('bottom pupil magnitude')
            ax4.set_title('bottom pupil phase')
            line.set_label('channel '+str(i))
            ax1.legend()

        aperture=np.float32(np.abs(f.res.channel0.pupil1)>0.0)
        Zk = f.res.channel0.zernike_polynomial
        fig = plt.figure(figsize=[16,8])
        for i in range(0,Nchannel):
            ax = fig.add_subplot(2,4,i+1)
            pupil_mag = np.sum(Zk*f.res['channel'+str(i)].zernike_coeff_mag[0].reshape((-1,1,1)),axis=0)*aperture
            plt.imshow(pupil_mag)
            plt.axis('off')
            plt.title('top pupil magnitude ' + str(i))
            ax = fig.add_subplot(2,4,i+5)
            pupil_mag = np.sum(Zk*f.res['channel'+str(i)].zernike_coeff_mag[1].reshape((-1,1,1)),axis=0)*aperture
            plt.imshow(pupil_mag)
            plt.axis('off')
            plt.title('bottom pupil magnitude ' + str(i))

        fig = plt.figure(figsize=[16,8])
        for i in range(0,Nchannel):
            ax = fig.add_subplot(2,4,i+1)
            pupil_phase = np.sum(Zk[4:]*f.res['channel'+str(i)].zernike_coeff_phase[0][4:].reshape((-1,1,1)),axis=0)*aperture
            plt.imshow(pupil_phase)
            plt.axis('off')
            plt.title('top pupil phase ' + str(i))
            ax = fig.add_subplot(2,4,i+5)
            pupil_phase = np.sum(Zk[4:]*f.res['channel'+str(i)].zernike_coeff_phase[1][4:].reshape((-1,1,1)),axis=0)*aperture
            plt.imshow(pupil_phase)
            plt.axis('off')
            plt.title('bottom pupil phase ' + str(i))
        plt.show()


    return


def showpsfvsdata(f,p,index):
    psf_data = f.rois.psf_data
    psf_fit = f.rois.psf_fit
    if p.channeltype == 'single':
        im1 = psf_data[index]
        im2 = psf_fit[index]
        psfcompare(im1,im2)
    else:
        Nchannel = psf_data.shape[0]
        for ch in range(0,Nchannel):
            if p.channeltype == '4pi':
                im1 = psf_data[ch,index,0]
                im2 = psf_fit[ch,index,0]
            else:
                im1 = psf_data[ch,index]
                im2 = psf_fit[ch,index]
            print('channel '+str(ch))
            psfcompare(im1,im2)
    return

def psfcompare(im1,im2):
    Nz = im1.shape[0]
    zind = range(0,Nz,4)
    fig = plt.figure(figsize=[3*len(zind),6])
    for i,id in enumerate(zind):
        ax = fig.add_subplot(2,len(zind),i+1)
        plt.imshow(im1[id],cmap='twilight')
        plt.axis('off')
        ax = fig.add_subplot(2,len(zind),i+1+len(zind))
        plt.imshow(im2[id],cmap='twilight')
        plt.axis('off')
    plt.show()
    return


def showpsfvsdata_insitu(f,p):
    if p.channeltype == 'single':
        rois = f.rois.psf_data
        I_model = f.res.I_model
        zf = f.res.pos[:,0]
        Nz = I_model.shape[0]
        edge = np.real(f.res.zoffset)+range(0,Nz+1)
        ind = np.digitize(zf,edge.flatten())
        rois_avg = np.zeros(I_model.shape)
        for ii in range(1,Nz+1):
            mask = (ind==ii)
            if sum(mask)>0:
                rois_avg[ii-1] = np.mean(rois[mask],axis=0)
        
        psfcompare(rois_avg,I_model)

    else:
        Nchannel = f.rois.psf_data.shape[0]    
        zoffset = f.res.channel0.zoffset
        for ch in range(0,Nchannel):
            rois = f.rois.psf_data[ch]
            I_model = f.res['channel'+str(ch)].I_model
            zf = f.res.channel0.pos[:,0]
            Nz = I_model.shape[0]
            edge = np.real(zoffset)+range(0,Nz+1)
            ind = np.digitize(zf,edge.flatten())
            rois_avg = np.zeros(I_model.shape)
            for ii in range(1,Nz+1):
                mask = (ind==ii)
                if sum(mask)>0:
                    rois_avg[ii-1] = np.mean(rois[mask],axis=0)
            print('channel '+str(ch))
            psfcompare(rois_avg,I_model)
    return

def showlocalization(f,p):
    Nz = f.locres.loc.z.shape[1]
    fig = plt.figure(figsize=[16,4])
    ax = fig.add_subplot(1,3,1)
    plt.plot(f.locres.loc.x.transpose()*p.pixel_size.x*1e3,'k',alpha=0.1)
    plt.plot(f.locres.loc.x[0]*0.0,'r')
    ax.set_ylabel('x bias (nm)')
    ax = fig.add_subplot(1,3,2)
    plt.plot(f.locres.loc.y.transpose()*p.pixel_size.y*1e3,'k',alpha=0.1)
    plt.plot(f.locres.loc.y[0]*0.0,'r')
    ax.set_ylabel('y bias (nm)')
    ax = fig.add_subplot(1,3,3)
    plt.plot(np.transpose(f.locres.loc.z-np.linspace(0,Nz-1,Nz))*p.pixel_size.z*1e3,'k',alpha=0.1)
    plt.plot(f.locres.loc.z[0]*0.0,'r')
    ax.set_ylabel('z bias (nm)')
    ax.set_ylim([-40,40])

    return

def showtransform(f):
    Nchannel = f.rois.psf_data.shape[0]
    ref_pos = f.res.channel0.pos
    dxy = f.res.xyshift 
    fig = plt.figure(figsize=[6*Nchannel,6])

    for i in range(1,Nchannel):
        pos = f.res['channel'+str(i)].pos
        ax = fig.add_subplot(1,Nchannel,i+1)
        plt.plot(ref_pos[:,1],ref_pos[:,2],'.')
        plt.plot(pos[:,1]-dxy[i][0],pos[:,2]-dxy[i][1],'o',markersize = 8,mfc='none')
        plt.plot(f.res.imgcenter[0],f.res.imgcenter[1],'*')
    
    ax.legend(['ref','target','center'])