# Full-waveform inversion sample

The sample runs the author's multiscale FWI implementation with either the convergent Born series (CBS) solver or S2NO. It includes complex source estimation, the reciprocal-field adjoint gradient, Gaussian smoothing, bound-constrained nonlinear conjugate gradients (NCG), and the full More–Thuente line search. The two solvers share the inversion protocol.

The numerical core comes from `USCT_S2NO_FWI_Phantom_main/inversion/arm_small`. The supplied configurations retain the original arm sample 51 and limb sample 12 settings. This repository contains source and text configuration only. Observations, pretrained weights, homogeneous wavefields, and reconstruction outputs are external assets.

## Run

Install the repository requirements and a CUDA-compatible PyTorch build. The original solver requires an NVIDIA CUDA GPU; the full 256-source CBS calculation uses substantial GPU memory. Run from the repository root:

```bash
# Check the external observation arrays without using a GPU.
python -m fwi.run --config examples/fwi/limb.yaml --solver cbs \
  --data-root /path/to/fwi_data --validate-only

# Full CBS inversion; no learned checkpoint or homogeneous-field input is needed.
python -m fwi.run --config examples/fwi/limb.yaml --solver cbs \
  --data-root /path/to/fwi_data --output outputs/limb12_cbs

# Full S2NO inversion using one checkpoint and one homogeneous bank per frequency.
python -m fwi.run --config examples/fwi/limb.yaml --solver s2no \
  --data-root /path/to/fwi_data --checkpoint-root /path/to/checkpoints \
  --output outputs/limb12_s2no
```

Use `examples/fwi/arm.yaml` for the original arm sample 51 setup. `--sample` changes only the input filename index; it does not tune the initialization or ROI for another sample. `--device 0` selects the visible CUDA device. Existing output directories are refused, including incomplete runs.

For an execution check, append `--max-stages 1 --max-iterations 1` and use a fresh output directory. This is a truncated run, not the full reconstruction protocol. Overrides are recorded in `protocol.json`.

## External data layout

The default YAML paths expect:

```text
fwi_data/
├── observations/
│   ├── 250k/test_12.npy
│   ├── 300k/test_12.npy
│   └── ...                     # through 600k, every 50 kHz
└── homogeneous/                # S2NO only
    ├── homo250.mat
    ├── homo300.mat
    └── ...
checkpoints/
├── 250k/<one-model>.ckpt
├── 300k/<one-model>.ckpt
└── ...
```

Paths are templates in the YAML and can be adapted to an existing dataset without copying files. `{frequency}` is the integer frequency in kHz; `{sample}` is the sample index. For example, an original archive may use `observed_data/limb/{frequency}k/test_{sample}.npy`. A checkpoint glob must match exactly one file per frequency. Conventional MAT observations are also accepted by selecting a `.mat` path and its `observation_key`.

| Input | Required shape and convention |
|---|---|
| Observations | Complex `(256 receivers, 256 sources)`, `.npy` or MAT variable `data`; the original FWI field normalization, source order, and phase convention must match. Near-source receiver pairs are muted internally. |
| Homogeneous fields | Complex `(256, 480, 480)` source-first bank; MAT variable `u_all_single`, or a `.npy` file. These are the original neural-operator reference fields before the input normalization. A 64-source bank is insufficient for this FWI example. |
| Checkpoints | One S2NO state dictionary or Lightning checkpoint per frequency. Width, Fourier modes, and depth are inferred by the shared loader and weights are loaded strictly. Padding and `add_term` are explicit in the YAML; original small-model values are 6 and `false`. |
| Ground truth | Not required. It is never used by this runner for initialization, optimization, or selection of the final iterate. |

Loaded complex arrays are converted to `complex64`, matching the original float32 model and wavefield buffers.

Do not substitute arbitrary physical-pressure measurements or reorient complex arrays by guessing: normalization, axes, phase, and the transducer order are part of the input contract. This code release does not claim to supply external measurements, reference fields, or pretrained weights.

## Grid and protocol

`fwi/geometry.py` contains the exact integer transducer coordinates exported from the original `param_arm.mat`, including its SHA-256. Both source and receiver arrays were checked element by element against the MAT file, as were the MATLAB linear indices. The coordinates are stored as text; a newly generated ideal circle is not substituted.

- Grid: 480 × 480, 0.5 mm spacing, 256 co-located transmitters and receivers. Discard Tx/Rx pairs separated by less than 50 mm. MATLAB coordinates are converted from 1-based to 0-based indexing when sampling.
- Frequencies: two sweeps over 250, 300, 350, 400, 450, 500, 550, and 600 kHz.
- Model variable: slowness `1 / speed`, flattened and restored in Fortran order. Speed bounds are 1400–2000 m/s; these bounds cannot recover faster bone speeds.
- Initial model: circular inclusion in 1500 m/s water, followed by the original sigma-10 Gaussian filter with replicated boundaries. Sample-specific radius and inclusion speed are in the YAML.
- At each objective evaluation, fit one complex amplitude per source by least squares. The adjoint uses reciprocity and the same full source wavefield bank. S2NO therefore supplies a physical approximate adjoint; this is not automatic differentiation through the neural network.
- The gradient retains the original `2 * omega**2 / speed` factor, final transpose, scaling, smoothing, and ROI mask.
- Original PR+ NCG, bound clipping, restart settings, stopping tolerances, and More–Thuente search are retained. Objective and gradient are independently normalized at each frequency; the initial normalization probe retains the original default gradient sigma of 2.
- Original CBS stopping: the minimum source update energy is normalized by its initial value, with threshold `1e-5` and `Bornmax=1000`. It is not a worst-source PDE residual test.

The limb YAML retains the original sample 12 settings: initial radius 140 and speed 1640 m/s; ROI radii 128 then 180; seven NCG iterations and eight line-search evaluations per stage; first-sweep model smoothing `[2,2,2,2,2,2,2,8]`; gradient smoothing `[5,5,5,5,3,3,3,3]` followed by all 2. The arm YAML retains the original sample 51 settings.

## Outputs

`initial_speed.npy` stores the initial model. `stages/r1_250.npy`, etc., store each stage's NCG output before subsequent smoothing; the associated JSON stores iteration counts, exit reason, objective trace, scaling, and elapsed time. `vdata.npy` stacks all completed stage outputs. `final_speed.npy` is the last completed stage output in m/s, without clipping or ground-truth-based replacement. `protocol.json`, `summary.json`, and `run.log` record the run.

The NCG exit description is reported as produced by the original optimizer. Independent objective/gradient scaling can make the line search terminate with a tiny step; an exit flag alone is not evidence of a successful reconstruction. No projected/backtracking optimizer from subsequent reproduction experiments is included.

## Source mapping and release adaptations

| Release file | Original file |
|---|---|
| `fwi/cbs.py` | `utils/CBS_utils.py` |
| `fwi/ncg.py` | `Optimizer_ncg_utils.py` |
| `fwi/objective_s2no.py`, `fwi/objective_cbs.py` | `cal_fcost_grad.py`, `cal_fcost_grad_cbs.py` |
| `fwi/source_s2no.py`, `fwi/source_cbs.py` | `source_estimation.py`, `source_estimation_cbs.py` |
| `fwi/smoothing.py` | 2D functions in `gauss.py` |
| `fwi/run.py` | Multiscale loops and stage setup from `main_cal_all_compile_S2NO.py` / `main_cal_all_CBS.py` |

The release replaces machine-specific paths with CLI/YAML inputs, removes plotting and training-validation side effects, shares the canonical training model, validates external array shapes and normalizes complex input dtype to the solver's `complex64`, loads one model at a time, and refuses destructive output replacement. A small attribute-access dictionary replaces the original `configmypy.Bunch` dependency in the numerical core. Compilation and preload warmups are omitted; this runner does not reproduce the paper's compilation/timing benchmark.

One explicit entry-point bug is fixed: `param['freq']` is now updated to `frequency_khz / 100` for S2NO at every stage, matching the original CBS entry point. In the archived S2NO entry point this value stayed at 5.

The original S2NO phase order is deliberately retained: divide the predicted complex field by `(1.9514e9 - 2.7546e8j)`, multiply by `1e10`, then transpose and conjugate. This differs from later locally calibrated reproduction runners. The source-estimation, gradient, CBS stopping rule, and NCG algorithm have not been replaced by those later variants. Numerical equivalence to the mathematical core does not imply identical outputs to a repaired reproduction pipeline or bitwise identity across CUDA versions.
