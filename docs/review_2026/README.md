# uiPSF review (October 2026)

A review of uiPSF written to start a discussion. It covers:

- the theory, and the physics the model leaves out
- what an excitation-PSF extension would need (confocal, ISM, MINFLUX)
- a cross-check against the TU Delft vectorial PSF (vecfit)
- field dependence, compared with FD-DeepLoc
- bugs on `test_roi`, hyperparameters, performance and the choice of framework

Nothing in `psflearning/` was changed.

- `uiPSF_review.html`: the full review; open it in a browser.
- `scripts/`: the synthetic tests behind the numbers in the review.
- `results/`: their raw output, one JSON line per fit.

## Scripts

Run all of them from inside `scripts/`.

| Script | What it does |
|---|---|
| `bead_fit_scaling.py` | Fits simulated beads with the unmodified `PSFZernikeBased` + `L_BFGS_B` + `Fitter`, rescaling entries of `self.weight`. Bright beads. |
| `bead_fit_scaling_lowsnr.py` | The same with dim beads (1.5–3·10³ photons/slice, background 80–150). |
| `bead_fit_mismatch.py <N> poly\|zstep\|bggrad` | The data contain one effect the model lacks: polychromatic emission, a 5 % piezo scale error, or a ±30 % background ramp. |
| `eval_timing_tf.py <N>` | Time for one loss + gradient evaluation: eager vs `tf.function` vs XLA. |
| `crosscheck_vecfit.py` | Compares the uiPSF vector PSF with a NumPy port of the TU Delft vecfit forward model. |
| `gpu_backend_check.py` | Checks that PyTorch on Apple MPS, CUDA or CPU can do complex FFT and matrix-DFT with gradients, and times a uiPSF-sized workload on the GPU vs the CPU. |

The uiPSF scripts were run against the `test_roi` branch (commit `b367f00`). Check out `psflearning/` from that commit before running them. They need TensorFlow 2.20 (CPU is enough), NumPy, SciPy and tqdm. They stub out tkinter, omegaconf and the io package, and add a NumPy 2 shim for `np.product`.

`gpu_backend_check.py` needs only `torch` and `numpy`.
