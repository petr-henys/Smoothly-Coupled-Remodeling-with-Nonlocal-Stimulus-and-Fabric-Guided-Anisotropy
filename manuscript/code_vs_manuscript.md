# Manuscript ↔ implementation (subsolvers + parameters)

This note summarizes the *current* implementation in `simulation/` and serves as a checklist for verifying that it matches the model and notation described in `manuscript/manuscript.tex`.

## Scope (implementation files)

- Implemented PDE blocks: `simulation/solvers/`
  - `MechanicsSolver` (elasticity)
  - `StimulusSolver` (stimulus field `S`)
  - `DensitySolver` (density field `rho`)
  - `FabricSolver` (log-fabric field `L`)
- Implemented mechanics drivers: `simulation/drivers.py` (`GaitDriver`, `StimulusCalculator`)
- Coupling / fixed-point / Anderson: `simulation/model.py`, `simulation/fixedsolver.py`, `simulation/anderson.py`
- Parameter defaults: `simulation/params.py` (via `simulation/config.py`)

## High-level mapping (implementation)

1. **Coupling order**
   - mechanics-driver → fabric → stimulus → density
   - Anderson stacks state fields in registration order: `L`, then `S`, then `rho`.
   - Block order is defined in `simulation/model.py` (`_build_solvers()`).

2. **Mechanical “daily” driver `psi_day`**
   - cycle-weighted **power mean** of *strain energy density* (SED) across discrete loading cases:
     - per case: `psi_i = max(0, 0.5 * sigma(u):eps(u))`
     - aggregate: `psi = (sum_i c_i * psi_i^p / sum_i c_i)^(1/p)`, where `c_i = day_cycles` and `p = stimulus_power_p`.
     - implemented in `simulation/drivers.py` (`GaitDriver._update_snapshots()`), with `p = cfg.stimulus.stimulus_power_p`.

3. **Fabric driver**
   - target built from cycle-weighted average of deviatoric-stress outer-products:
     - per case: `sigma_dev = sigma - (tr(sigma)/3)*I`, `Q_i = sym(sigma_dev*sigma_dev^T)` (DG0 tensor)
     - average: `Qbar = sum_i c_i Q_i / sum_i c_i`
     - target `L_target(Qbar)` uses eigenvalue normalization by geometric mean, exponent `gammaF`, clamp to `[m_min, m_max]`, and final log-space mean removal (`tr(L_target)=0`).
     - implemented in `simulation/drivers.py` (`StimulusCalculator.compute_Q`) and `simulation/solvers/fabric.py` (`FabricSolver._L_target_from_Qbar()`).

4. **Stimulus PDE + source term**
   - linear diffusion–reaction in `S`, but with a **saturating mechanostat** drive (lazy-zone + `tanh` saturation):
     - computes specific energy `m = psi / rho_safe`
     - normalized deviation `delta = (m - m_ref)/m_ref` with `m_ref = psi_ref / rho_ref`
     - optional lazy-zone gate with `stimulus_delta0`
     - final drive `drive = S_max * tanh(delta_eff / stimulus_kappa)`
     - time constant and diffusion appear as `tau` and `tau*D` (see `StimulusSolver._compile_forms()` in `simulation/solvers/stimulus.py`).

5. **Density PDE**
   - **isotropic** diffusion (`D_rho` scalar) and bounded formation/resorption kinetics with separate gains:
     - `S_pos = smooth_max(S, 0)`, `S_neg = smooth_max(-S, 0)` (no explicit `Δ_S` threshold in density)
     - formation: `k_form * S_pos * (1 - rho/rho_max)`
     - resorption: `k_res * S_neg * (1 - rho/rho_min)` (negative for `rho>rho_min`)
     - optional *surface availability factor* `A_surf(rho_old)` (porosity→specific surface proxy), enabled by `density.surface_use`.
     - implemented in `DensitySolver._compile_forms()` in `simulation/solvers/density.py`.

6. **Smooth “max” clamp**
   - code uses `smooth_abs(x) = sqrt(x^2+eps^2) - eps` so that `smooth_max(x,x)=x` exactly (`simulation/utils.py`).

## Current parameter names + defaults (from `simulation/params.py`)

### Material (`cfg.material`)

- `E0 = 7500.0` MPa
- `nu0 = 0.3`
- `n_trab = 2.0`, `n_cort = 1.3`
- `rho_trab_max = 1.0`, `rho_cort_min = 1.25` g/cm³
- `stiff_pE = 1.0`, `stiff_pG = 1.0`

### Stimulus (`cfg.stimulus`)

- `stimulus_power_p = 4.0`
- `psi_ref = 0.01` MPa
- `stimulus_tau = 25.0` day
- `stimulus_D = 1.0` mm²/day
- `stimulus_S_max = 1.0`
- `stimulus_kappa = 0.5`
- `stimulus_delta0 = 0.10`

### Density (`cfg.density`)

- `rho_min = 0.1`, `rho_max = 2.0`, `rho0 = 1.0`, `rho_ref = 1.0` g/cm³
- `k_rho_form = 2e-2`, `k_rho_resorb = 2e-2` g/cm³/day
- `D_rho = 2e-2` mm²/day
- Surface availability (optional):
  - `surface_use = True`
  - `rho_tissue = 2.0` g/cm³
  - `surface_A_min = 0.02`
  - `surface_S0 = 1.0` 1/mm

### Fabric (`cfg.fabric`)

- `fabric_tau = 50.0` day
- `fabric_D = 1.0` mm²/day
- `fabric_cA = 1.0`
- `fabric_gammaF = 1.0`
- `fabric_epsQ = 1e-12` MPa²
- `fabric_aniso_eps = 1e-4` (dimensionless activity-gate width)
- `fabric_m_min = 0.2`, `fabric_m_max = 5.0`

### Solver (`cfg.solver`)

- Linear: `ksp_type = "minres"`, `pc_type = "gamg"`, `ksp_rtol = 1e-6`, `ksp_atol = 1e-7`, `ksp_max_it = 100`, `ksp_reuse_pc = False`
- Fixed-point/AA: `accel_type="anderson"`, `m=5`, `beta=1.0`, `lam=1e-9`, `gamma=0.05`, `safeguard=True`, `backtrack_max=5`, `step_limit_factor=2.0`, `restart_on_reject_k=2`, `restart_on_stall=1.10`, `restart_on_cond=1e12`, `coupling_tol=1e-4`, `max_subiters=25`, `min_subiters=2`

### Time (`cfg.time`)

- `total_time = 500.0` day
- `dt_initial = 25.0` day
- `adaptive_dt = False` (if enabled: `adaptive_rtol = 1e-2`, `adaptive_atol = 1e-3`, `dt_min = 1e-4` day, `dt_max = 100.0` day)

### Numerics (`cfg.numerics`)

- `quadrature_degree = 4`
- `smooth_eps = 1e-6` (small absolute smoothing width; interpreted in the units of the smoothed quantity)

### Output (`cfg.output`)

- `results_dir = ".results"`
- `saving_interval = 1`
- `log_file = "simulation.log"`

### Geometry (`cfg.geometry`)

- `fix_tag = 1`
- `load_tag = 2`

## Manuscript locations to keep in sync

Not exhaustive, but key ones to revisit when the implementation changes:

- Abstract and keywords: coupling order + “anisotropic transport” wording.
- `Stimulus submodel` (`eq:stim-strong-nd`, `eq:stim-psi-nd`) vs `StimulusSolver` + `GaitDriver`.
- `Density submodel` (`eq:Drho-nd`, `eq:smooth-splits-nd`, `eq:density-strong-nd`) vs `DensitySolver`.
- `Direction (fabric) submodel` (`eq:L-strong-nd`, `eq:M-def-nd`, `eq:M-hat-nd`) vs `Qbar`-based target and activity factor.
- “Gait cycle integration” section: `psi_day` power mean and loading-cycle weights.
- Backward–Euler and unified weak forms: stimulus/density/fabric coefficients and sources.
- Fixed-point solver section: state ordering, block update order, driver description, and Anderson safeguards.
- “Physical Parameter Analysis…” section: parameter values/names (material, stimulus, density, fabric, solver) vs `simulation/params.py`.
