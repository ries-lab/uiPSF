# Same harness, but the simulated data contain one effect the uiPSF model does not include.
import os
import sys
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'bead_fit_scaling.py')).read().split("def run(")[0])   # defines opts, Data, N, Nz, L, Ztrue, pos, I, bg, v, gt, clean, data, rng
src_run = "def run(" + open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'bead_fit_scaling.py')).read().split("def run(")[1].split("if __name__")[0]

def make_clean(lam=0.68, pz=0.05):
    o = opts(); o.imaging.emission_wavelength = lam
    g = PSFZernikeBased(o); g.psftype='vector'
    dd = Data(np.ones((N,Nz,L,L))); dd.pixelsize_z = pz; g.data = dd
    g.calc_initials(dd); g.weight = np.ones(5, np.float32); g.ind=[0,N]
    vv = [pos, bg, np.repeat(I,Nz,1), Ztrue*1.0, np.ones(2,np.float32)*0.6*np.pi, np.zeros((N,2),np.float32)]
    return g.calc_forward_images([tf.constant(x) for x in vv]).numpy()

case = sys.argv[2]
if case == 'poly':        # emission spectrum: 3 lines across a ~60 nm band instead of one wavelength
    clean = (make_clean(0.65) + make_clean(0.68) + make_clean(0.71)) / 3
elif case == 'zstep':     # piezo moves 5% more than nominal
    clean = make_clean(pz=0.0525)
elif case == 'bggrad':    # background ramp of +-30% across the ROI
    ramp = 1 + 0.3*np.linspace(-1,1,L)[None,None,None,:]
    clean = make_clean() + bg*(ramp-1)
data = rng.poisson(np.maximum(clean,0)).astype(np.float32)
exec(src_run)
for ws in [(1,1,1,1,1),(1,0.1,1,1,1),(1,1,1,5,5)]:
    run(wscale=ws, tag=case)
