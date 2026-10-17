# Brief: re-implementing uiPSF (v2)

This is the starting brief for a planning session. It condenses the decisions and evidence in `uiPSF_review.html` (same folder). Read the review for details and file:line references. Nothing here is settled code; it is the agreed direction and the open questions.

## Why a new core

- **The forward model is correct.** It agrees with an independent port of the TU Delft vectorial PSF (vecfit) to float32 precision, including supercritical-angle emission (`scripts/crosscheck_vecfit.py`).
- **The robustness problems are in the fitting.**
  - The hand-set variable scaling (`self.weight`) is only needed because L-BFGS-B does not converge well. On dim synthetic beads, rescaling one variable changes the z error of the learned model from 61 nm to 5 nm, with a perfectly matched model (`scripts/bead_fit_scaling_lowsnr.py`).
  - The Poisson loss masks non-finite pixels and is averaged instead of summed.
  - Positivity penalties use a ramped `mu` that changes the objective during line searches.
  - There are exact or near degeneracies: pupil magnitude scale vs photons, and Z4 defocus vs bead z.
  - With more beads than `batch_size`, the loss and the per-bead gradients are inconsistent.
- **Model mismatch in real data is a second, separate problem.** It gives the same biased answer for every scaling and leaves an excess in the loss. A background gradient and a piezo z-scale error were the most damaging effects simulated (`scripts/bead_fit_mismatch.py`).
- **Speed.** The current code runs eager with a NumPy round trip on every evaluation, and recomputes the full vectorial PSF per bead and per slice.

## Decisions

1. **Framework: PyTorch**, running eager by default.
   - Devices: CUDA, then Apple MPS, then CPU, chosen automatically.
   - Apple M1 Pro is verified: complex FFT and matrix DFT with gradients run natively on MPS (error < 1e-6), about 2.5× faster than its CPU (`scripts/gpu_backend_check.py`).
   - Inductor does not generate code for complex ops, so compiling helps little unless the maths is split into real and imaginary parts.
   - MPS has no float64. Run the small linear-algebra steps on the CPU in float64.
2. **Forward model: matrix DFT** (two complex matmuls) instead of a padded FFT or CZT. It is faster on both CPU and MPS. Keep the uiPSF physics: vectorial pupil, Fresnel coefficients, aplanatic apodisation √cos θ_imm / cos θ_med, freely rotating dipoles, analytic bead kernel and pixel integration.
3. **Likelihood:**
   - Exact Poisson, or shifted Poisson for sCMOS (`d + σ²/g²` vs `μ + σ²/g²`), computed with `xlogy` and no masking.
   - **Summed**, not averaged.
   - Optional per-pixel offset, gain and variance maps.
   - EMCCD handled through an excess-noise factor.
4. **Solver: Levenberg–Marquardt / Gauss–Newton** on the likelihood.
   - Jacobian-vector products via `torch.func` (jvp/vjp).
   - Block structure: per-bead parameters plus shared pupil and field parameters, solved with a Schur complement.
   - Marquardt diagonal scaling makes it independent of parameter units, so there is **no `self.weight`**.
   - Bounds handled by reparametrisation (softplus/exp) or projected steps. No penalty ramps.
   - Fallback and comparison: L-BFGS with Fisher-diagonal preconditioning.
   - Parameter uncertainties (CRLB) come from the same Fisher matrix.
5. **Parametrisation without degeneracies:**
   - No piston, tilt or defocus in the phase.
   - Pupil normalised to unit energy, so photons are physical totals.
   - Zernike set chosen by n_max with the option of high-NA modes orthogonalised against k_z.
   - Smooth bleaching model instead of free photons per slice.
6. **Field dependence (default when beads span the FOV):**
   - Zernike coefficients as low-order polynomials in field position, structured by nodal aberration theory (about 50 parameters; compare TU Delft `funnat`, Droste et al. *Optica* 2025).
   - A fitted optical-axis centre.
   - An optional coarse B-spline residual with a second-difference penalty.
   - A shared focal surface z_f(x, y) covering stage tilt, camera tilt and field curvature.
   - Fitted jointly with the pupil, initialised from a quick per-bead or per-cell plus NAT fit.
   - Export the analytic model and a coverage/uncertainty map.
7. **Cheap physics terms to add** (the "stage tilt" family). Add each only where the residual diagnostics show it is needed:
   - Bead stacks: per-slice shared z/xy offsets and photon factor, and a global z scale.
   - Planar background per ROI.
   - Saturation mask.
   - In-situ: fitted n_med, full Gibson–Lanni including nominal immersion RI and working distance, and coverslip-thickness offset.
8. **Residual atlas diagnostics** after every fit. Deviance per pixel (≈1 when the model is adequate) grouped:
   - by slice, field position, radius, z and sensor position;
   - plus held-out-bead deviance and z-bias.

   These are the quality metric, and the input for any automatic tuning or agent triage.
9. **Packaging:**
   - Pure Python, `pip install` (plus conda); an installer script handles the Windows CUDA wheel.
   - Configs stay as YAML (omegaconf).
   - Outputs stay compatible with current uiPSF .h5 / spline exports, so SMAP and FD-DeepLoc keep working. Document the coordinate and sign conventions.

## Scope

**v1:**
- Bead stacks, single channel: Zernike-vector and pixel pupils, voxel kept as a baseline.
- Multi-channel (biplane, ratiometric, polarisation channels).
- Field-dependent bead model.
- In-situ single channel.

**Later:**
- 4Pi.
- In-situ multi-channel and field-dependent.
- Polychromatic emission.
- Depth-integrated bead model.
- Jones pupil.
- Excitation PSFs (confocal, ISM, MINFLUX bead scans, MINFLUX in-situ). See review section C for the forward-model differences.

## Validation plan

1. **Oracles:**
   - The current TF uiPSF forward model (`PSFZernikeBased.genpsfmodel`).
   - The vecfit NumPy port in `scripts/crosscheck_vecfit.py`.

   Match both to float32 precision before anything else.
2. **Gradient checks** against finite differences in float64 on CPU.
3. **Synthetic benchmark** with known ground truth, a superset of the review scripts:
   - bright, dim and very dim beads;
   - each mismatch effect;
   - field-dependent aberrations;
   - in-situ simulations.

   Metrics: Zernike RMS error, z-bias, held-out deviance, runtime and convergence rate.
4. **Real benchmark:** 5–10 bead datasets that currently need tuning (to be collected), plus a few well-behaved ones. Success means v2 needs no per-dataset weights and matches or beats tuned v1.
5. **CI:** CPU unit tests for the forward model, gradients and solver. GPU tests run manually on CUDA and MPS (`scripts/gpu_backend_check.py`).

## Milestones (suggested)

1. Forward model module, device handling, oracle tests.
2. Likelihood and LM/Gauss–Newton solver with Schur complement; synthetic single-channel bead benchmark.
3. Data pipeline: ROI finding (regression-tested against the `main` and `test_roi` finders), readers, camera models; residual atlas.
4. Field-dependent model (NAT); multi-channel.
5. In-situ.
6. Exports and compatibility; documentation; real-data benchmark; release.

## Open questions for the planning session

- New repository or a `v2/` package inside ries-lab/uiPSF? Name?
- How strict must .h5 output compatibility be, and which downstream consumers matter (SMAP, FD-DeepLoc, others)?
- GUI: none in v1 (notebooks plus CLI), or plan for one from the start? Collaborating groups have desktop GUIs that could sit on top of a shared core.
- Coordination with collaborators working on PyTorch and Julia implementations: one shared core plus test data?
- Which real datasets form the benchmark, and who provides them?
- Is stage tilt a measured input or a fitted parameter?

## Files

- `uiPSF_review.html`: the full review (theory, missing physics, excitation PSFs, vecfit comparison, field dependence, bugs, hyperparameters, performance, framework).
- `scripts/`: synthetic tests and checks; `results/`: their raw outputs.
