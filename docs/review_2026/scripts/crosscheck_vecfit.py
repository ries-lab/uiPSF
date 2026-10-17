# Cross-check: uiPSF vector PSF (PSFZernikeBased.genpsfmodel) vs a numpy port of the TU Delft vecfit forward model
import os
# (get_pupil_matrix.m + get_field_matrix_derivatives.m + get_psfs_derivatives.m, 'free' dipole, ztype 'stage', zvals=0)
import sys, numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'bead_fit_scaling.py')).read().split("def run(")[0].replace("N, Nz, L = int(sys.argv[1]) if len(sys.argv)>1 else 8, 41, 21","N, Nz, L = 1, 41, 21"))
import tensorflow as tf
o = opts(); o.model.bin = 1
NA, lam, nimm, nmed, ncov = 1.43, 0.68, 1.516, 1.335, 1.516
px_x, px_y, pz, Np = 0.127, 0.116, 0.05, 64
p = PSFZernikeBased(o); p.psftype='vector'; d = Data(np.ones((1,Nz,L,L))); p.data = d
p.calc_initials(d)
Zk = p.Zk.shape[0]
c = np.zeros((2, Zk, 1, 1), np.float32); c[0,0] = 1
# Noll index (0-based) -> phase in rad
ab = {4:0.6, 5:-0.3, 6:0.25, 7:-0.2, 10:0.4}
for j,v in ab.items(): c[1,j] = v
I_ui, _ = p.genpsfmodel(tf.constant([0.,0.]), Zcoeff=tf.constant(c))
I_ui = np.real(np.array(I_ui)); s0=(I_ui.shape[0]-Nz)//2; I_ui=I_ui[s0:s0+Nz]

# ---- vecfit port ----
D = 2/Np; xy = np.arange(-1+D/2, 1, D)
YP, XP = np.meshgrid(xy, xy)          # MATLAB [YPupil,XPupil] = meshgrid(XYPupil,XYPupil)
r2 = XP**2+YP**2
argMed = 1-r2*NA**2/nmed**2
cmed = np.sqrt(np.abs(argMed))*np.where(argMed<0, -1j, 1)   # vecfit SAF branch
ccov = np.sqrt(1-r2*NA**2/ncov**2+0j); cimm = np.sqrt(1-r2*NA**2/nimm**2+0j)
FPmc = 2*nmed*cmed/(nmed*ccov+ncov*cmed); FSmc = 2*nmed*cmed/(nmed*cmed+ncov*ccov)
FPci = 2*ncov*ccov/(ncov*cimm+nimm*ccov); FSci = 2*ncov*ccov/(ncov*ccov+nimm*cimm)
FP, FS = FPmc*FPci, FSmc*FSci
phi = np.arctan2(YP, XP); cp, sp = np.cos(phi), np.sin(phi); sth = np.sqrt(1-cmed**2)
pvec = np.stack([FP*cmed*cp, FP*cmed*sp, -FP*sth]); svec = np.stack([-FS*sp, FS*cp, 0*FS])
Pol = np.stack([cp*pvec-sp*svec, sp*pvec+cp*svec])      # (2,3,Np,Np)
ap = (r2<1).astype(float)
Amp = ap*np.sqrt(cimm)/(nmed*cmed)
# Zernike in Noll ordering with vecfit normalisation, phase in rad (uiPSF convention), via uiPSF's own polynomials on this grid
from psflearning.learning.utilities import genZern1
Zv = genZern1(8, Np)                                       # uiPSF grid is meshgrid(krange,krange): x along columns
W = sum(v*Zv[j] for j,v in ab.items())
def psf_vecfit(Wrad, sign):
    Pm = Amp*np.exp(sign*1j*Wrad)*Pol
    kx = 2*np.pi*NA/lam*XP; ky = 2*np.pi*NA/lam*YP; kzi = 2*np.pi*nimm/lam*cimm
    xs = (np.arange(L)-(L-1)/2)*px_x; ys = (np.arange(L)-(L-1)/2)*px_y
    out = np.zeros((Nz,L,L))
    zr = (np.arange(Nz)-(Nz-1)/2)*pz
    Ex = np.exp(1j*np.einsum('x,ij->xij', xs, kx)); Ey = np.exp(1j*np.einsum('y,ij->yij', ys, ky))
    for iz,z in enumerate(zr):
        F = np.exp(-1j*(0 - z)*kzi)*Pm                      # stage defocus, emitter at 0
        E = np.einsum('abij,xij,yij->abxy', F, Ex, Ey)      # field at (x along dim0, y along dim1)
        out[iz] = (np.abs(E)**2).sum((0,1))/3
    return out
best = None
for sign in (+1,-1):
    for transW in (False, True):
        Iv = psf_vecfit(W.T if transW else W, sign)
        for tr in (False, True):
            for fz in (False, True):
                for fx in (False, True):
                    for fy in (False, True):
                        A = Iv.transpose(0,2,1) if tr else Iv
                        if fz: A = A[::-1]
                        if fx: A = A[:,:,::-1]
                        if fy: A = A[:,::-1,:]
                        a = A/A.sum(); b = I_ui/I_ui.sum()
                        err = np.abs(a-b).max()/b.max()
                        if best is None or err < best[0]: best = (err, sign, transW, tr, fz, fx, fy, np.sqrt(((a-b)**2).mean())/b.max())
print("max |diff| / peak = %.2e   rms/peak = %.2e   (phase sign %+d, W transposed %s, img transposed %s, z flipped %s, x flipped %s, y flipped %s)" % (best[0], best[7], best[1], best[2], best[3], best[4], best[5], best[6]))
# also with no aberration
W0 = 0*W; Iv0 = psf_vecfit(W0, 1)
p0 = c.copy(); p0[1]=0
I0,_ = p.genpsfmodel(tf.constant([0.,0.]), Zcoeff=tf.constant(p0)); I0=np.real(np.array(I0))[s0:s0+Nz]
errs=[]
for tr in (False,True):
    for fz in (False,True):
        A = Iv0.transpose(0,2,1) if tr else Iv0
        if fz: A=A[::-1]
        errs.append(np.abs(A/A.sum()-I0/I0.sum()).max()/(I0/I0.sum()).max())
print("unaberrated: max |diff| / peak = %.2e" % min(errs))
