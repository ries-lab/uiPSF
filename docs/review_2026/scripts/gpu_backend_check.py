"""
Check whether a PyTorch GPU backend (Apple MPS, NVIDIA CUDA) can run the operations a
PyTorch uiPSF would need, and how fast it is compared with the CPU.

What it tests, on the GPU and against a float64 CPU reference:
  1. complex64 arithmetic and exp(i*phase)
  2. complex FFT (fft2 / ifft2) forward
  3. gradients through phase -> complex pupil -> FFT -> |E|^2 -> loss (autograd with complex intermediates)
  4. the same via a matrix Fourier transform (complex matmul), the alternative to padded FFT/CZT
  5. torch.compile of that function on the device
and then times a uiPSF-sized vectorial workload (beads x z-slices x 6 dipole fields) on GPU and CPU.

Usage:  python gpu_backend_check.py            (picks cuda, then mps, then cpu)
        python gpu_backend_check.py --device mps
Prints a summary and writes gpu_backend_check_result.json next to where it is run.
"""
import argparse, json, platform, sys, time, traceback
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('--device', default=None)
p.add_argument('--beads', type=int, default=20)
p.add_argument('--nz', type=int, default=41)
args = p.parse_args()

if args.device:
    dev = torch.device(args.device)
elif torch.cuda.is_available():
    dev = torch.device('cuda')
elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
    dev = torch.device('mps')
else:
    dev = torch.device('cpu')

def sync():
    if dev.type == 'cuda': torch.cuda.synchronize()
    elif dev.type == 'mps': torch.mps.synchronize()

report = dict(
    python=sys.version.split()[0], torch=torch.__version__, platform=platform.platform(),
    machine=platform.machine(), device=str(dev),
    mps_built=bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_built()),
    mps_available=bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()),
    cuda_available=torch.cuda.is_available(), tests={}, timing={})
print(json.dumps({k: v for k, v in report.items() if k not in ('tests', 'timing')}, indent=1))

def run_test(name, fn):
    try:
        ok, detail = fn()
        report['tests'][name] = dict(ok=bool(ok), detail=detail)
    except Exception as e:
        report['tests'][name] = dict(ok=False, detail=f"{type(e).__name__}: {str(e)[:300]}")
        traceback.print_exc(limit=1)
    t = report['tests'][name]
    print(f"[{'PASS' if t['ok'] else 'FAIL'}] {name}: {t['detail']}")

rng = np.random.default_rng(0)
Np, M = 64, 42
yy, xx = np.meshgrid(np.linspace(-1, 1, Np), np.linspace(-1, 1, Np), indexing='ij')
aperture = ((xx**2 + yy**2) < 1).astype(np.float32)
phase0 = (0.8*(xx**2 - yy**2) + 0.3*(3*(xx**2 + yy**2) - 2)*xx).astype(np.float32)

def rel(a, b):
    a = a.detach().cpu().to(torch.complex128 if a.is_complex() else torch.float64)
    b = b.detach().cpu().to(a.dtype)
    return float((a - b).abs().max() / b.abs().max())

TOL = 1e-4   # float32 on device vs float64 reference

def t_complex():
    ph = torch.tensor(phase0)
    ref = torch.tensor(aperture, dtype=torch.float64) * torch.exp(1j*ph.double())
    out = torch.tensor(aperture, device=dev) * torch.exp(1j*ph.to(dev))
    e = rel(out, ref); return e < TOL, f"max rel err {e:.1e}"

def t_fft():
    a = torch.tensor(rng.standard_normal((8, Np, Np)) + 1j*rng.standard_normal((8, Np, Np)))
    ref = torch.fft.ifft2(torch.fft.fft2(a) * 2.0)
    out = torch.fft.ifft2(torch.fft.fft2(a.to(torch.complex64).to(dev)) * 2.0)
    e = rel(out, ref); return e < TOL, f"max rel err {e:.1e}"

def psf_loss_fft(ph, dtype, device):
    pupil = torch.tensor(aperture, device=device, dtype=dtype) * torch.exp(1j*ph)
    pad = torch.zeros((2*Np, 2*Np), dtype=pupil.dtype, device=device)
    pad[:Np, :Np] = pupil
    E = torch.fft.fftshift(torch.fft.fft2(pad))
    I = (E.real**2 + E.imag**2)
    return (I * torch.linspace(0, 1, I.numel(), device=device, dtype=dtype).reshape(I.shape)).sum() / I.sum()

def t_grad_fft():
    ph_ref = torch.tensor(phase0, dtype=torch.float64, requires_grad=True)
    psf_loss_fft(ph_ref, torch.float64, 'cpu').backward()
    ph = torch.tensor(phase0, device=dev, requires_grad=True)
    psf_loss_fft(ph, torch.float32, dev).backward()
    e = rel(ph.grad, ph_ref.grad); return e < 1e-3, f"max rel grad err {e:.1e}"

xs = np.linspace(-M/2 + 0.5, M/2 - 0.5, M) * 0.06            # image grid (um), 2x upsampled 120 nm pixels
ks = np.linspace(-1 + 1/Np, 1 - 1/Np, Np) * 1.45 / 0.68       # pupil grid (1/um)
F = np.exp(2j*np.pi*np.outer(xs, ks))                        # (M, Np) matrix DFT

def psf_loss_mft(ph, dtype, device):
    cdt = torch.complex128 if dtype == torch.float64 else torch.complex64
    Fm = torch.tensor(F, dtype=cdt, device=device)
    pupil = torch.tensor(aperture, device=device, dtype=dtype) * torch.exp(1j*ph)
    E = Fm @ pupil @ Fm.T
    I = (E.real**2 + E.imag**2)
    return (I * torch.linspace(0, 1, I.numel(), device=device, dtype=dtype).reshape(I.shape)).sum() / I.sum()

def t_grad_mft():
    ph_ref = torch.tensor(phase0, dtype=torch.float64, requires_grad=True)
    psf_loss_mft(ph_ref, torch.float64, 'cpu').backward()
    ph = torch.tensor(phase0, device=dev, requires_grad=True)
    psf_loss_mft(ph, torch.float32, dev).backward()
    e = rel(ph.grad, ph_ref.grad); return e < 1e-3, f"max rel grad err {e:.1e}"

def t_compile():
    # constants are built outside the compiled function: under tracing, torch.tensor(numpy complex128)
    # becomes a complex128 tensor on the device first, which MPS does not support
    Fm = torch.tensor(F, dtype=torch.complex64).to(dev)
    ap = torch.tensor(aperture).to(dev)
    wts = torch.linspace(0, 1, M*M, device=dev).reshape(M, M)
    def loss(ph):
        E = Fm @ (ap * torch.exp(1j*ph)) @ Fm.T
        I = E.real**2 + E.imag**2
        return (I * wts).sum() / I.sum()
    f = torch.compile(loss)
    ph = torch.tensor(phase0, device=dev, requires_grad=True)
    l = f(ph); l.backward()
    ph2 = torch.tensor(phase0, device=dev, requires_grad=True)
    psf_loss_mft(ph2, torch.float32, dev).backward()
    e = rel(ph.grad, ph2.grad); return e < 1e-4, f"compiled vs eager grad diff {e:.1e}"

for name, fn in [('complex_arith', t_complex), ('complex_fft', t_fft), ('grad_through_fft', t_grad_fft),
                 ('grad_through_matrix_dft', t_grad_mft), ('torch_compile', t_compile)]:
    run_test(name, fn)

# ---- timing: uiPSF-sized vectorial workload, forward + backward ----
def workload(device, N, Nz, use_fft):
    dt = torch.float32
    g = torch.Generator().manual_seed(1)
    ph = (torch.tensor(phase0) + 0.1*torch.randn(Np, Np, generator=g)).to(device).requires_grad_(True)
    z = torch.linspace(-1, 1, Nz, device=device).reshape(1, Nz, 1, 1, 1)
    zb = (torch.randn(N, generator=g)*0.3).to(device).reshape(N, 1, 1, 1, 1)
    kz = torch.tensor(np.sqrt(np.maximum((1.516/0.68)**2 - (xx**2 + yy**2)*(1.45/0.68)**2, 0)), dtype=dt, device=device)
    h = torch.randn(6, Np, Np, generator=g).to(device).to(torch.complex64)   # stand-in for the 6 dipole field components
    Fm = torch.tensor(F, dtype=torch.complex64, device=device)
    ap = torch.tensor(aperture, device=device)
    def run():
        pupil = ap * torch.exp(1j*ph)
        defocus = torch.exp(-2j*np.pi*kz*(z + zb))                          # (N, Nz, 1, Np, Np)
        P = pupil * defocus * h                                              # (N, Nz, 6, Np, Np)
        if use_fft:
            pad = torch.zeros(P.shape[:-2] + (2*Np, 2*Np), dtype=P.dtype, device=device)
            pad[..., :Np, :Np] = P
            E = torch.fft.fft2(pad)[..., :M, :M]
        else:
            E = Fm @ P @ Fm.T
        I = (E.real**2 + E.imag**2).sum(2)
        loss = I.mean(); loss.backward()
        return loss
    run(); sync()
    t = time.time(); n = 5
    for _ in range(n): run()
    sync()
    return (time.time() - t) / n

for use_fft in (False, True):
    key = 'fft' if use_fft else 'matrix_dft'
    try:
        tc = workload(torch.device('cpu'), args.beads, args.nz, use_fft)
        td = workload(dev, args.beads, args.nz, use_fft) if dev.type != 'cpu' else tc
        report['timing'][key] = dict(cpu_s=round(tc, 4), device_s=round(td, 4), speedup=round(tc/td, 2))
        print(f"[time] {key}: {args.beads} beads x {args.nz} z x 6 fields, fwd+bwd: cpu {tc:.3f} s, {dev.type} {td:.3f} s, speedup {tc/td:.1f}x")
    except Exception as e:
        report['timing'][key] = dict(error=f"{type(e).__name__}: {str(e)[:300]}")
        print(f"[time] {key}: FAILED {type(e).__name__}: {str(e)[:200]}")

with open('gpu_backend_check_result.json', 'w') as f:
    json.dump(report, f, indent=1)
print("\nwrote gpu_backend_check_result.json")
