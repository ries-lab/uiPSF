import matplotlib.pyplot as plt
from matplotlib import gridspec
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
        photon = np.abs(f.res.channel0.intensity.transpose())

    fig = plt.figure(figsize=[16,8])
    spec = gridspec.GridSpec(ncols=4, nrows=2,
                         width_ratios=[3, 3,3,3], wspace=0.4,
                         hspace=0.3, height_ratios=[4, 4])
    ax = fig.add_subplot(spec[0])
    plt.plot(pos[:,2]-cor[:,1])
    plt.xlabel('bead number')
    plt.ylabel('x (pixel)')
    ax = fig.add_subplot(spec[1])
    plt.plot(pos[:,1]-cor[:,0])
    plt.xlabel('bead number')
    plt.ylabel('y (pixel)')
    ax = fig.add_subplot(spec[2])
    plt.plot(pos[:,0])
    plt.xlabel('bead number')
    plt.ylabel('z (pixel)')
    if p.channeltype == '4pi':
        ax = fig.add_subplot(spec[3])
        plt.plot(phi)
        ax.set_xlabel('bead number')
        ax.set_ylabel('phi (radian)')
    ax = fig.add_subplot(spec[4])
    plt.plot(photon)
    if len(photon.shape)>1:
        plt.xlabel('z slice')
        plt.legend(['bead 1'])
    else:
        plt.xlabel('bead number')
    plt.ylabel('photon')
    ax = fig.add_subplot(spec[5])
    plt.plot(bg)
    plt.xlabel('bead number')
    plt.ylabel('background')
    ax = fig.add_subplot(spec[6])
    plt.plot(drift)
    plt.xlabel('bead number')
    plt.ylabel('drift per z slice (pixel)')
    plt.legend(['x','y'])
    plt.show()

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
    if p.channeltype == '4pi':
        phi = np.angle(f.res.channel0.intensity)
        photon = np.abs(f.res.channel0.intensity)

    fig = plt.figure(figsize=[16,8])
    spec = gridspec.GridSpec(ncols=4, nrows=2,
                         width_ratios=[3, 3,3,3], wspace=0.4,
                         hspace=0.3, height_ratios=[4, 4])
    ax = fig.add_subplot(spec[0])
    plt.plot(pos[:,2]-cor[:,1],'.')
    plt.xlabel('emitter number')
    plt.ylabel('x (pixel)')

    ax = fig.add_subplot(spec[1])
    plt.plot(pos[:,1]-cor[:,0],'.')
    plt.xlabel('emitter number')
    plt.ylabel('y (pixel)')

    ax = fig.add_subplot(spec[2])
    plt.plot(pos[:,0],'.')
    plt.xlabel('emitter number')
    plt.ylabel('z (pixel)')
    if p.channeltype == '4pi':
        ax = fig.add_subplot(spec[3])
        plt.plot(phi,'.')
        ax.set_xlabel('emitter number')
        ax.set_ylabel('phi (radian)')
        ax = fig.add_subplot(spec[6])
        plt.plot(pos[:,0],phi,'.')
        ax.set_xlabel('z (pixel)')
        ax.set_ylabel('phi (radian)')
    ax = fig.add_subplot(spec[4])
    plt.plot(photon,'.')
    plt.xlabel('emitter number')
    plt.ylabel('photon')
    ax = fig.add_subplot(spec[5])
    plt.plot(bg,'.')
    plt.xlabel('emitter number')
    plt.ylabel('background')

    return

def showpupil(f,p, index=None):
    if p.channeltype == 'single':
        fig = plt.figure(figsize=[12,5])
        if index is None:
            pupil = f.res.pupil
        else:
            pupil = f.res.pupil[index]

        ax = fig.add_subplot(1,2,1)
        plt.imshow(np.abs(pupil),interpolation='nearest')
        plt.title('pupil magnitude')
        plt.colorbar()
        ax = fig.add_subplot(1,2,2)
        plt.imshow(np.angle(pupil),interpolation='nearest')
        plt.title('pupil phase')
        plt.colorbar()
    elif p.channeltype == 'multi':
        Nchannel = f.rois.psf_data.shape[0]
        fig = plt.figure(figsize=[5*Nchannel,4])
        fig1 = plt.figure(figsize=[5*Nchannel,4])
        for i in range(0,Nchannel):
            if index is None:
                pupil = f.res['channel'+str(i)].pupil
            else:
                pupil = f.res['channel'+str(i)].pupil[index]

            ax = fig.add_subplot(1,Nchannel,i+1)
            pupil_mag = np.abs(pupil)
            h=ax.imshow(pupil_mag,interpolation='nearest')
            ax.axis('off')
            ax.set_title('pupil magnitude ' + str(i))
            fig.colorbar(h,ax=ax)
            ax1 = fig1.add_subplot(1,Nchannel,i+1)
            pupil_phase = np.angle(pupil)
            h1=ax1.imshow(pupil_phase,interpolation='nearest')
            ax1.axis('off')
            ax1.set_title('pupil phase ' + str(i))
            fig1.colorbar(h1,ax=ax1)
    elif p.channeltype == '4pi':
        Nchannel = f.rois.psf_data.shape[0]
        fig = plt.figure(figsize=[20,8])
        for i in range(0,Nchannel):
            ax = fig.add_subplot(2,4,i+1)
            pupil_mag = np.abs(f.res['channel'+str(i)].pupil1)
            plt.imshow(pupil_mag,interpolation='nearest')
            plt.axis('off')
            plt.title('top pupil magnitude ' + str(i))
            plt.colorbar()
            ax = fig.add_subplot(2,4,i+5)
            pupil_mag = np.abs(f.res['channel'+str(i)].pupil2)
            plt.imshow(pupil_mag,interpolation='nearest')
            plt.axis('off')
            plt.title('bottom pupil magnitude ' + str(i))
            plt.colorbar()
        fig = plt.figure(figsize=[20,8])
        for i in range(0,Nchannel):
            ax = fig.add_subplot(2,4,i+1)
            pupil_phase = np.angle(f.res['channel'+str(i)].pupil1)
            plt.imshow(pupil_phase,interpolation='nearest')
            plt.axis('off')
            plt.title('top pupil phase ' + str(i))
            plt.colorbar()
            ax = fig.add_subplot(2,4,i+5)
            pupil_phase = np.angle(f.res['channel'+str(i)].pupil2)
            plt.imshow(pupil_phase,interpolation='nearest')
            plt.axis('off')
            plt.title('bottom pupil phase ' + str(i))
            plt.colorbar()
    return

def showzernike(f,p,index=None):
    n_max = p.option.model.n_max
    Nk = (n_max+1)*(n_max+2)//2
    
    indz = np.array([4,5,6,7,10,21])
    textstr = [None]*6
    textstr[0] = r'$\mathrm{D \ astigmatism}$'
    textstr[1] = r'$\mathrm{astigmatism}$'
    textstr[2] = r'$\mathrm{V \ coma}$'
    textstr[3] = r'$\mathrm{H \ coma}$'
    textstr[4] = r'$\mathrm{spherical}$'
    textstr[5] = r'$\mathrm{2nd \ spherical}$'

    mask = indz<Nk
    Nzk = np.sum(mask)
    if p.channeltype == 'single':
        fig = plt.figure(figsize=[10,8])
        if index is None:
            zcoeff = f.res.zernike_coeff
            
        else:
            zcoeff = f.res.zernike_coeff[:,index]

        if len(f.res.pupil.shape)>2:
            aperture=np.float32(np.abs(f.res.pupil[0])>0.0)
        else:
            aperture=np.float32(np.abs(f.res.pupil)>0.0)
        ax = fig.add_subplot(2,1,1)
        ax.plot(zcoeff.transpose(),'.-')
        ax.plot(indz[mask],zcoeff[1,indz[mask]],'ko',markersize = 6,mfc='none')
        plt.xlabel('zernike polynomial',fontsize=16)
        plt.ylabel('coefficient')
        plt.legend(['pupil magnitude','pupil phase'])
        
        ax1 = fig.add_subplot(2,1,2)
        tstr = ''
        for i in range(0,Nzk):
            tstr = '\n'.join((tstr, textstr[i]+'=%.2f' % (zcoeff[1][indz[i]], )))
        tstr = tstr[1:]
        bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
        ax1.text(0.03, 0.9, tstr, fontsize=12, bbox=bbox,
            transform=ax1.transAxes, horizontalalignment='left',verticalalignment='top')
        ax1.set_axis_off()

        Zk = f.res.zernike_polynomial

        pupil_mag = np.sum(Zk*zcoeff[0].reshape((-1,1,1)),axis=0)*aperture
        pupil_phase = np.sum(Zk[4:]*zcoeff[1][4:].reshape((-1,1,1)),axis=0)*aperture

        fig = plt.figure(figsize=[12,5])
        ax = fig.add_subplot(1,2,1)
        plt.imshow(pupil_mag,interpolation='nearest')
        plt.colorbar()
        plt.title('pupil magnitude')
        ax = fig.add_subplot(1,2,2)
        plt.imshow(pupil_phase,interpolation='nearest')
        plt.colorbar()
        plt.title('pupil phase')
        plt.suptitle('Zernike expanded pupil function, no apodization',fontsize=16)
    elif p.channeltype == 'multi':
        Nchannel = f.rois.psf_data.shape[0]
        fig = plt.figure(figsize=[12,6])
        ax1 = fig.add_subplot(2,2,1)
        ax2 = fig.add_subplot(2,2,2)
        ax5 = fig.add_subplot(2,2,3)
        fig1 = plt.figure(figsize=[5*Nchannel,4])
        fig2 = plt.figure(figsize=[5*Nchannel,4])
        Zk = f.res.channel0.zernike_polynomial
        
        for i in range(0,Nchannel):
            if index is None:
                zcoeff = f.res['channel'+str(i)].zernike_coeff
            else:
                zcoeff = f.res['channel'+str(i)].zernike_coeff[:,index]
            
            if len(f.res['channel'+str(i)].pupil.shape)>2:
                aperture=np.float32(np.abs(f.res['channel'+str(i)].pupil[0])>0.0)
            else:
                aperture=np.float32(np.abs(f.res['channel'+str(i)].pupil)>0.0)


            line, = ax1.plot(zcoeff[0],'.-')    
            ax2.plot(zcoeff[1],'.-')
            ax1.set_xlabel('zernike polynomial')
            ax1.set_ylabel('coefficient')
            ax1.set_title('pupil magnitude')
            ax2.set_title('pupil phase')
            line.set_label('channel '+str(i))
            ax1.legend()
            if i==Nchannel-1:
                ax1.plot(indz[mask],zcoeff[0][indz[mask]],'ko',markersize = 6,mfc='none')
                ax2.plot(indz[mask],zcoeff[1][indz[mask]],'ko',markersize = 6,mfc='none')

            for k in range(0,Nzk):

                textstr[k] = '\n'.join((textstr[k],r'$\mathrm{ch}%d=%.2f$' % (i,zcoeff[1][indz[k]], ),))

            ax3 = fig1.add_subplot(1,Nchannel,i+1)
            pupil_mag = np.sum(Zk*zcoeff[0].reshape((-1,1,1)),axis=0)*aperture
            h = ax3.imshow(pupil_mag,interpolation='nearest')
            ax3.axis('off')
            ax3.set_title('pupil magnitude ' + str(i),fontsize=20)
            fig1.colorbar(h,ax=ax3)
            ax4 = fig2.add_subplot(1,Nchannel,i+1)
            pupil_phase = np.sum(Zk[4:]*zcoeff[1][4:].reshape((-1,1,1)),axis=0)*aperture
            h1=ax4.imshow(pupil_phase,interpolation='nearest')
            ax4.axis('off')
            ax4.set_title('pupil phase ' + str(i),fontsize=20)
            fig2.colorbar(h1,ax=ax4)
        
        bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
        for k in range(0,len(textstr)):
            ax5.text(0.01+k*0.35, 0.9, textstr[k], fontsize=12, bbox=bbox,
                    transform=ax5.transAxes, horizontalalignment='left',verticalalignment='top')
        ax5.set_axis_off()
    elif p.channeltype == '4pi':
        Nchannel = f.rois.psf_data.shape[0]
        fig = plt.figure(figsize=[12,10])
        ax1 = fig.add_subplot(3,2,1)
        ax2 = fig.add_subplot(3,2,2)
        ax3 = fig.add_subplot(3,2,3)
        ax4 = fig.add_subplot(3,2,4)
        ax5 = fig.add_subplot(3,2,5)

        for k in range(0,Nzk):
            textstr[k] = '\n'.join((textstr[k],r'$\mathrm{upper}=%.2f$' % (f.res.channel0.zernike_coeff_phase[0][indz[k]], ),
                                    r'$\mathrm{lower}=%.2f$' % (f.res.channel0.zernike_coeff_phase[1][indz[k]], )))

        for i in range(0,Nchannel):
            zcoeff_mag = f.res['channel'+str(i)].zernike_coeff_mag
            zcoeff_phase = f.res['channel'+str(i)].zernike_coeff_phase
            line, = ax1.plot(zcoeff_mag[0],'.-')    
            ax2.plot(zcoeff_phase[0],'.-')
            ax2.set_ylim((-0.6,0.6))
            ax3.plot(zcoeff_mag[1],'.-')
            ax4.plot(zcoeff_phase[1],'.-')
            ax4.set_ylim((-0.6,0.6))
            ax3.set_xlabel('zernike polynomial')
            ax3.set_ylabel('coefficient')
            ax1.set_title('upper pupil magnitude')
            ax2.set_title('upper pupil phase')
            ax3.set_title('lower pupil magnitude')
            ax4.set_title('lower pupil phase')
            line.set_label('channel '+str(i))
            ax1.legend()
            if i==Nchannel-1:
                ax1.plot(indz[mask],zcoeff_mag[0][indz[mask]],'ko',markersize = 6,mfc='none')
                ax2.plot(indz[mask],zcoeff_phase[0][indz[mask]],'ko',markersize = 6,mfc='none')
                ax3.plot(indz[mask],zcoeff_mag[1][indz[mask]],'ko',markersize = 6,mfc='none')
                ax4.plot(indz[mask],zcoeff_phase[1][indz[mask]],'ko',markersize = 6,mfc='none')

        bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
        for k in range(0,len(textstr)):
            ax5.text(0.01+k*0.35, 0.9, textstr[k], fontsize=12, bbox=bbox,
                    transform=ax5.transAxes, horizontalalignment='left',verticalalignment='top')
        ax5.set_axis_off()

        aperture=np.float32(np.abs(f.res.channel0.pupil1)>0.0)
        Zk = f.res.channel0.zernike_polynomial
        fig = plt.figure(figsize=[20,8])
        for i in range(0,Nchannel):
            ax = fig.add_subplot(2,4,i+1)
            pupil_mag = np.sum(Zk*f.res['channel'+str(i)].zernike_coeff_mag[0].reshape((-1,1,1)),axis=0)*aperture
            plt.imshow(pupil_mag,interpolation='nearest')
            plt.axis('off')
            plt.title('upper pupil magnitude ' + str(i),fontsize=20)
            plt.colorbar()
            ax = fig.add_subplot(2,4,i+5)
            pupil_mag = np.sum(Zk*f.res['channel'+str(i)].zernike_coeff_mag[1].reshape((-1,1,1)),axis=0)*aperture
            plt.imshow(pupil_mag,interpolation='nearest')
            plt.axis('off')
            plt.title('lower pupil magnitude ' + str(i),fontsize=20)
            plt.colorbar()
        fig = plt.figure(figsize=[20,8])
        for i in range(0,Nchannel):
            ax = fig.add_subplot(2,4,i+1)
            pupil_phase = np.sum(Zk[4:]*f.res['channel'+str(i)].zernike_coeff_phase[0][4:].reshape((-1,1,1)),axis=0)*aperture
            plt.imshow(pupil_phase,interpolation='nearest')
            plt.axis('off')
            plt.title('upper pupil phase ' + str(i),fontsize=20)
            plt.colorbar()
            ax = fig.add_subplot(2,4,i+5)
            pupil_phase = np.sum(Zk[4:]*f.res['channel'+str(i)].zernike_coeff_phase[1][4:].reshape((-1,1,1)),axis=0)*aperture
            plt.imshow(pupil_phase,interpolation='nearest')
            plt.axis('off')
            plt.title('lower pupil phase ' + str(i),fontsize=20)
            plt.colorbar()
        plt.show()
    return


def showzernikemap(f,p,index=None):
    if p.channeltype == 'single':
        zmap = f.res.zernike_map
        zcoeff = f.res.zernike_coeff
        pupil = f.res.pupil
        Zk = f.res.zernike_polynomial
        zernikemap(f,index,zmap,zcoeff,pupil,Zk)
    if p.channeltype == 'multi':
        Nchannel = f.rois.psf_data.shape[0]
        for i in range(0,Nchannel):
            print('channel '+str(i))
            zmap = f.res['channel'+str(i)].zernike_map
            zcoeff = f.res['channel'+str(i)].zernike_coeff
            pupil = f.res['channel'+str(i)].pupil
            Zk = f.res['channel'+str(i)].zernike_polynomial
            zernikemap(f,index,zmap,zcoeff,pupil,Zk)

def zernikemap(f,index,zmap,zcoeff,pupil,Zk):

    if index is None:
        index = [4,5,6,7,10,11,12,15,16,21]
        mask = np.array(index) < (zcoeff.shape[-1]-1)
        index = np.array(index)[mask]

    fig = plt.figure(figsize=[16,4])
    ax = fig.add_subplot(1,2,1)
    plt.plot(zcoeff[0].transpose(),'k',alpha=0.1)
    plt.plot(index,zcoeff[0,0,index],'ro')
    plt.xlabel('zernike polynomial')
    plt.ylabel('coefficient')
    plt.title('pupil magnitude')
    plt.legend(['bead 1'])
    ax = fig.add_subplot(1,2,2)
    plt.plot(zcoeff[1].transpose(),'k',alpha=0.1)
    plt.plot(index,zcoeff[1,0,index],'ro')
    plt.xlabel('zernike polynomial')
    plt.ylabel('coefficient')
    plt.title('pupil phase')
    plt.legend(['bead 1'])

    if len(pupil.shape)>2:
        aperture=np.float32(np.abs(pupil[0])>0.0)
    else:
        aperture=np.float32(np.abs(pupil)>0.0)
    imsz = np.array(f.rois.image_size)
    

    scale = (imsz[-2:]-1)/(np.array(zmap.shape[-2:])-1)

    N = len(index)
    Nx = 4
    Ny = N//Nx+1
    fig = plt.figure(figsize=[4.5*Nx,7*Ny])
    spec = gridspec.GridSpec(ncols=Nx, nrows=2*Ny,
                        width_ratios=list(np.ones(Nx)), wspace=0.1,
                        hspace=0.2, height_ratios=list(np.ones(2*Ny)))

    abername = ['']*np.max([zcoeff.shape[-1],22])
    abername[3] = 'defocus'
    abername[4] = 'D astigmatism'
    abername[5] = 'astigmatism'
    abername[6] = 'V coma'
    abername[7] = 'H coma'
    abername[10] = 'spherical'
    abername[11] = '2nd ast.'
    abername[12] = '2nd D ast.'
    abername[15] = '2nd H coma'
    abername[16] = '2nd V coma'
    abername[21] = '2nd spherical'
    for i,id in enumerate(index):
        j = i//Nx
        ax = fig.add_subplot(spec[i+j*Nx])
        #plt.imshow(Zmap[1,id],cmap='twilight',vmin=-0.05,vmax=0.5)
        plt.imshow(zmap[1,id],cmap='twilight',interpolation='nearest')
        #plt.plot(cor[:,-1]/scale[-1],cor[:,-2]/scale[-2],'ro',markersize=5)
        plt.axis('off')
        plt.title('('+str(id)+') '+abername[id],fontsize=16)
        plt.colorbar()
        ax = fig.add_subplot(spec[i+(j+1)*Nx])
        plt.imshow(Zk[id]*aperture,cmap='viridis',interpolation='nearest')
        plt.axis('off')
        plt.colorbar()
    plt.show()

def showpsfvsdata(f,p,index):
    psf_data = f.rois.psf_data
    psf_fit = f.rois.psf_fit
    if p.channeltype == 'single':
        im1 = psf_data[index]
        im2 = psf_fit[index]
        psfcompare(im1,im2,p.pixel_size.z)
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
            psfcompare(im1,im2,p.pixel_size.z)
    try:
        cor = f.res.cor
    except:
        cor = f.res.channel1.cor
    imsz = f.rois.image_size
    fig = plt.figure(figsize=[4,4])
    plt.plot(cor[index,-1],cor[index,-2],'ro')
    plt.xlim(0,imsz[-1])
    plt.ylim(0,imsz[-2])
    plt.xlabel('x (pixel)')
    plt.ylabel('y (pixel)')
    plt.legend(['bead'+str(index)])
    plt.show()
    return

def psfcompare(im1,im2,pz):
    Nz = im1.shape[0]
    zrange = np.linspace(-Nz/2+0.5,Nz/2-0.5,Nz)*pz
    zind = range(0,Nz,4)
    cc = im1.shape[-1]//2
    N = len(zind)+1
    fig = plt.figure(figsize=[3*N,6])
    for i,id in enumerate(zind):
        ax = fig.add_subplot(2,N,i+1)
        plt.imshow(im1[id],cmap='twilight')
        plt.axis('off')
        plt.title(str(np.round(zrange[id],2))+r'$\ \mu$m',fontsize=30)
        ax = fig.add_subplot(2,N,i+1+N)
        plt.imshow(im2[id],cmap='twilight')
        plt.axis('off')
    ax = fig.add_subplot(2,N,N)
    plt.imshow(im1[:,cc],cmap='twilight')
    plt.axis('off')
    plt.title('xz',fontsize=30)
    plt.colorbar()
    ax = fig.add_subplot(2,N,2*N)
    plt.imshow(im2[:,cc],cmap='twilight')
    plt.axis('off')
    plt.colorbar()
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
        
        psfcompare(rois_avg,I_model,p.pixel_size.z)

    else:
        Nchannel = f.rois.psf_data.shape[0]    
        zoffset = f.res.channel0.zoffset
        for ch in range(0,Nchannel):
            rois = f.rois.psf_data[ch]
            I_model = f.res['channel'+str(ch)].I_model
            if p.channeltype == '4pi':
                I_model = f.res['channel'+str(ch)].psf_model
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
            psfcompare(rois_avg,I_model,p.pixel_size.z)

    return

def showlocalization(f,p):
    loc = f.locres.loc
    plotlocbias(loc,p)
    if hasattr(f.locres,'loc_FD'):
        loc = f.locres.loc_FD
        plotlocbias(loc,p)
    return

def plotlocbias(loc,p):
    Nz = loc.z.shape[1]
    fig = plt.figure(figsize=[16,4])
    spec = gridspec.GridSpec(ncols=3, nrows=1,
                         width_ratios=[3, 3,3], wspace=0.3,
                         hspace=0.3, height_ratios=[1])
    ax = fig.add_subplot(spec[0])
    plt.plot(loc.x.transpose()*p.pixel_size.x*1e3,'k',alpha=0.1)
    plt.plot(loc.x[0]*0.0,'r')
    ax.set_xlabel('z slice')
    ax.set_ylabel('x bias (nm)')
    ax = fig.add_subplot(spec[1])
    plt.plot(loc.y.transpose()*p.pixel_size.y*1e3,'k',alpha=0.1)
    plt.plot(loc.y[0]*0.0,'r')
    ax.set_xlabel('z slice')
    ax.set_ylabel('y bias (nm)')
    ax = fig.add_subplot(spec[2])
    bias_z = (loc.z-np.linspace(0,Nz-1,Nz))*p.pixel_size.z*1e3
    plt.plot(bias_z.transpose(),'k',alpha=0.1)
    plt.plot(loc.z[0]*0.0,'r')
    ax.set_xlabel('z slice')
    ax.set_ylabel('z bias (nm)')
    ax.set_ylim([np.maximum(np.quantile(bias_z[:,2:-2],0.001),-300),np.minimum(np.quantile(bias_z[:,2:-2],0.999),300)])
    plt.show()
    return

def showtransform(f):
    Nchannel = f.rois.psf_data.shape[0]
    ref_pos = f.res.channel0.pos
    dxy = f.res.xyshift 
    fig = plt.figure(figsize=[5*Nchannel,10])
    spec = gridspec.GridSpec(ncols=Nchannel, nrows=2,
                        width_ratios=list(np.ones(Nchannel)), wspace=0.3,
                        hspace=0.2, height_ratios=[1,1])

    cor_ref = np.concatenate((ref_pos[:,1:], np.ones((ref_pos.shape[0], 1))), axis=1)

    for i in range(1,Nchannel):
        pos = f.res['channel'+str(i)].pos
        if Nchannel<3:
            cor_target = np.matmul(cor_ref-f.res.imgcenter, f.res.T)[..., :-1]+f.res.imgcenter[:-1]
        else:
            cor_target = np.matmul(cor_ref-f.res.imgcenter, f.res.T[i-1])[..., :-1]+f.res.imgcenter[:-1]

        ax = fig.add_subplot(spec[i])
        plt.plot(ref_pos[:,-1],ref_pos[:,-2],'.')
        plt.plot(pos[:,-1]-dxy[i][-1],pos[:,-2]-dxy[i][-2],'o',markersize = 8,mfc='none')
        plt.plot(f.res.imgcenter[1],f.res.imgcenter[0],'*')
        ax.set_xlabel('x (pixel)')
        ax.set_ylabel('y (pixel)')
        plt.title('channel'+str(i))
        ax1 = fig.add_subplot(spec[Nchannel+i])
        plt.plot(cor_target[:,-1],cor_target[:,-2],'.')
        plt.plot(pos[:,-1],pos[:,-2],'o',markersize = 8,mfc='none')
        plt.plot(f.res.imgcenter[1],f.res.imgcenter[0],'*')
        ax1.set_xlabel('x (pixel)')
        ax1.set_ylabel('y (pixel)')

    
    ax.legend(['ref','target','center'])
    ax1.legend(['ref_trans','target','center'])


def showpsf(f,p):
    if p.channeltype == 'single':
        im1 = f.res.I_model
        psfdisp(im1,p.pixel_size.z)
    else:
        Nchannel = f.rois.psf_data.shape[0]
        for ch in range(0,Nchannel):
            if p.channeltype == '4pi':
                im1 = f.res['channel'+str(ch)].psf_model
            else:
                im1 = f.res['channel'+str(ch)].I_model
            print('channel '+str(ch))
            psfdisp(im1,p.pixel_size.z)
    return


def psfdisp(im1,pz):
    Nz = im1.shape[0]
    zrange = np.linspace(-Nz/2+0.5,Nz/2-0.5,Nz)*pz

    zind = range(0,Nz,4)
    cc = im1.shape[-1]//2
    N = len(zind)+1
    fig = plt.figure(figsize=[3*N,3])
    for i,id in enumerate(zind):
        ax = fig.add_subplot(1,N,i+1)
        plt.imshow(im1[id],cmap='twilight')
        plt.title(str(np.round(zrange[id],2))+r'$\ \mu$m',fontsize=30)
        plt.axis('off')
    ax = fig.add_subplot(1,N,N)
    plt.imshow(im1[:,cc],cmap='twilight')
    plt.axis('off')
    plt.title('xz',fontsize=30)
    plt.colorbar()
    plt.show()
    return


def showcoord(f,p):
    if p.channeltype == 'single':
        fig = plt.figure(figsize=[5,5])
        cor = f.res.cor
        cor_all = f.res.cor_all

        plt.plot(cor_all[:,-1],cor_all[:,-2],'.')
        plt.plot(cor[:,-1],cor[:,-2],'o',markersize = 8,mfc='none')
        plt.xlabel('x (pixel)')
        plt.ylabel('y (pixel)')

        plt.legend(['all','selected'])
    else:
        Nchannel = f.rois.psf_data.shape[0]
        fig = plt.figure(figsize=[5*Nchannel,5])
        spec = gridspec.GridSpec(ncols=Nchannel, nrows=1,
                            width_ratios=list(np.ones(Nchannel)), wspace=0.3,
                            hspace=0.2, height_ratios=[1])

        for i in range(0,Nchannel):
            cor = f.res['channel'+str(i)].cor
            cor_all = f.res['channel'+str(i)].cor_all

            ax = fig.add_subplot(spec[i])
            plt.plot(cor_all[:,-1],cor_all[:,-2],'.')
            plt.plot(cor[:,-1],cor[:,-2],'o',markersize = 8,mfc='none')
            ax.set_xlabel('x (pixel)')
            ax.set_ylabel('y (pixel)')
            plt.title('channel'+str(i))

        ax.legend(['all','selected'])