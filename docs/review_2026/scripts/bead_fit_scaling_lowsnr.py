# Synthetic bead test of uiPSF zernike_vector learning (test_roi branch code, unmodified, small numpy-2 shims)
import os, sys, time, json, numpy as np
np.product = np.prod
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
import types; _p=types.ModuleType('psflearning'); _p.__path__=[os.path.join(REPO,'psflearning')]; sys.modules['psflearning']=_p
from unittest import mock
for _m in ['tkinter','tkinter.messagebox','omegaconf','psflearning.io','psflearning.io.param','psflearning.io.h5']: sys.modules[_m]=mock.MagicMock()
_l=types.ModuleType('psflearning.learning'); _l.__path__=[os.path.join(REPO,'psflearning','learning')]; sys.modules['psflearning.learning']=_l
import tensorflow as tf
from types import SimpleNamespace as NS
from psflearning.learning.psfs.PSFZernikeBased_file import PSFZernikeBased
from psflearning.learning.fitters.Fitter_file import Fitter
from psflearning.learning.optimizers import L_BFGS_B
from psflearning.learning.loss_functions import mse_real_zernike

def opts():
    return NS(model=NS(with_IMM=False, const_pupilmag=False, symmetric_mag=False, blur_sigma=0.5, var_blur=True,
                       var_photon=True, estimate_drift=False, bin=2, pupilsize=64, n_max=6, with_apoid=True),
              imaging=NS(NA=1.43, emission_wavelength=0.68, RI=NS(imm=1.516, med=1.335, cov=1.516)))

class Data:
    def __init__(s, rois):
        s.rois = rois.astype(np.float32); s.pixelsize_x=0.127; s.pixelsize_y=0.116; s.pixelsize_z=0.05
        s.bead_radius=0.0; s.skew_const=None
        s.centers = np.tile(np.array([[20,100,100]]),(rois.shape[0],1)); s.centers_all=s.centers
    def get_image_data(s): return s.rois, s.rois, s.centers, np.zeros(s.rois.shape[0])

N, Nz, L = int(sys.argv[1]) if len(sys.argv)>1 else 8, 41, 21
rng = np.random.default_rng(0)
# ground truth
gt = PSFZernikeBased(opts()); gt.psftype='vector'
gt.data = Data(np.ones((N,Nz,L,L)))
init, _ = gt.calc_initials(gt.data); gt.weight = np.ones(5, np.float32)
Zk = gt.Zk.shape[0]
zphase = np.zeros(Zk); zphase[[4,5,6,7,10,11]] = [0.9,-0.3,0.25,-0.2,0.4,0.15]   # rad: astig, coma, spherical...
zmag = np.zeros(Zk); zmag[0]=1; zmag[3]=-0.15
Ztrue = np.stack([zmag,zphase])[:,:,None,None].astype(np.float32)
pos = np.c_[rng.uniform(-3,3,N), rng.uniform(-1.5,1.5,(N,2))].astype(np.float32)
I = rng.uniform(1.5e3,3e3,(N,1,1,1)).astype(np.float32); bg = rng.uniform(80,150,(N,1,1,1)).astype(np.float32)
v = [pos, bg/gt.weight[1], np.repeat(I,Nz,1)/gt.weight[0], np.stack([Ztrue[0]/gt.weight[4], Ztrue[1]/gt.weight[3]]),
     np.ones(2,np.float32)*0.6*np.pi, np.zeros((N,2),np.float32)]
gt.ind=[0,N]
clean = gt.calc_forward_images([tf.constant(x) for x in v]).numpy()
data = rng.poisson(np.maximum(clean,0)).astype(np.float32)

def run(wscale=(1,1,1,1,1), rate=1.1, maxiter=200, tag=''):
    class P(PSFZernikeBased):
        def calc_initials(self, d, start_time=None):
            out, st = super().calc_initials(d, start_time)
            ws = np.array(wscale, np.float32)
            self.weight = self.weight*ws
            out[1] = out[1]/ws[1]; out[2]=out[2]/ws[0]; out[3]=out[3]/ws[4]
            return out, st
    p = P(opts()); p.psftype='vector'
    d = Data(data)
    opt = L_BFGS_B(maxiter=maxiter); opt.batch_size=1600; opt.rate=rate
    f = Fitter(d, p, opt, mse_real_zernike, loss_weight=[1,1,0,0.01,1,1,1,0,0.1])
    t=time.time(); res,_ = f.learn_psf(start_time=0); dt=time.time()-t
    Z = res[6]
    st = opt.status
    err_phase = np.sqrt(np.mean((Z[1,4:,0,0]-Ztrue[1,4:,0,0])**2))
    out = dict(tag=tag, wscale=list(wscale), rate=rate, msg=str(st.message), nit=int(st.nit), nfev=int(st.nfev),
               loss=float(st.fun), rms_phase_err_rad=float(err_phase),
               z_err_px=float(np.std((res[0][:,0]-(Nz-1)//2) - pos[:,0])), time_s=dt, s_per_eval=dt/st.nfev)
    print(json.dumps(out), flush=True)
    return out

if __name__=='__main__':
    run(tag='low default')
    for ws in [(1,10,1,1,1),(1,0.1,1,1,1),(1,1,1,0.2,0.2),(1,1,1,5,5),(0.1,1,1,1,1)]:
        run(wscale=ws, tag='low scaled')
