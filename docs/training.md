# Forward-operator training

`training/` is the cleaned pretraining implementation supplied in `S2NO_main_clean.tar.gz`. It contains S2NO and the BFNO, FNO, and UNet baselines. The model equations, tensor parameter names, losses, data normalization, and optimizer schedule are retained from that source. Training data and checkpoints are external inputs; this repository does not include them.

## Environment and launch

Install the repository requirements in an environment with a PyTorch build appropriate for your GPU. The source training environment used PyTorch **2.6.0+cu118**; the public code requires PyTorch, PyTorch Lightning, NumPy, SciPy, PyYAML, pandas, and matplotlib. No Weights & Biases account or dependency is required.

From the repository root:

```bash
export S2NO_DATA_ROOT=/path/to/datasets

# Check the full expected file inventory before allocating GPUs.
python -m training.main_pretrain --check-data

# Original four-GPU training configuration.
bash training/scripts/train_s2no_500k.sh

# A single-GPU run, with a separate output location.
python -m training.main_pretrain \
  --config-file training/configs/s2no_500k_20_7_128.yaml \
  --data-root /path/to/datasets \
  --devices 1 \
  --output-dir outputs/training/s2no-example
```

The equivalent baseline launchers are `train_bfno_500k.sh`, `train_fno_500k.sh`, and `train_unet_500k.sh`. Each launcher uses the active Python environment, works from any current directory, and forwards additional CLI arguments. Set `PYTHON=/path/to/python` when a specific interpreter is needed. `python training/main_pretrain.py --help` also works without the scientific Python dependencies or dataset installed.

`--data-root` overrides `S2NO_DATA_ROOT`, which overrides `data.root` in YAML. Relative data roots and output locations resolve from the repository root. Every relative dataset path in YAML resolves beneath the selected data root; absolute paths are supported unchanged. Use `--print-config` to inspect the resulting paths. `--accelerator cpu --devices 1` is supported for small development runs; the released 480 × 480 configurations are intended for GPUs.

## Dataset layout and tensor convention

For each of `breast`, `arm`, and `limb`, prepare the following layout:

```text
datasets/
├── breast_data/
│   ├── speed/
│   │   ├── train/train_1.npy ... train_7210.npy
│   │   └── test/test_1.npy ... test_320.npy
│   └── 500k/
│       ├── train/train_1.npy ... train_7210.npy
│       └── test/test_1.npy ... test_320.npy
├── arm_data/             # same layout, counts below
└── limb_data/            # same layout, counts below
```

File indices are **one-based**. A speed file stores a real `(480, 480)` sound-speed image in m/s. Its matching field file stores a complex `(64, 480, 480)` wavefield array, with the first dimension indexing 64 source excitations. The loader uses NumPy `.npy` files, not the MATLAB files accepted by the FWI example. Training and validation concatenate the organs in the order **breast → arm → limb**.

| Organ | Training file pairs | Validation file pairs | Homogeneous reference field |
| --- | ---: | ---: | --- |
| Breast | 7,210 | 320 | `breast_data/500k/train/train_7201.npy` |
| Arm | 7,163 | 373 | `arm_data/500k/train/train_7154.npy` |
| Limb | 6,903 | 108 | `limb_data/500k/train/train_6894.npy` |

These are the original counts, including extra homogeneous/source reference files. Do not subtract the reference files from the counts unless you rebuild and reindex the dataset. Each field file yields 64 training items: **1,361,664** training items and **51,264** validation items per full pass. For a zero-based global item index `i`, the loader uses speed file `i // 64` in the concatenated list and source `i % 64` from that organ's homogeneous reference. `data.exp: multi` is the supported release mode; the incomplete legacy single-source branch is rejected explicitly.

Each item is `(speed, source, target, index)`:

| Value | Tensor shape | Transformation |
| --- | --- | --- |
| `speed` | `(H, W, 1)`, float32 | `(1500 / c - 1) * 30` |
| `source` | `(H, W, 2)`, float32 | homogeneous complex field × `2e-3`, split into `[real, imaginary]` |
| `target` | `(H, W, 2)`, float32 | target complex field × `2e-3`, split into `[real, imaginary]` |
| `index` | scalar integer | zero-based index over all file/source pairs |

The models receive batches shaped `(B, H, W, 1)` and `(B, H, W, 2)` and return `(B, H, W, 2)`. Convert predictions back to complex physical field units with `(prediction[..., 0] + 1j * prediction[..., 1]) / 2e-3`. The legacy configuration name `source_type: theta` still feeds the two homogeneous-field channels; it does not create an angle channel in this multi-source loader.

The source training sampler shuffles individual file/source items; validation preserves the concatenated order. With distributed training, Lightning supplies its distributed sampler. `data.batch` is the DataLoader batch size; the retained `train.batchsize` field is not used by the loader.

## Configurations and optimization

| Model | YAML in `training/configs/` | Original settings |
| --- | --- | --- |
| S2NO | `s2no_500k_20_7_128.yaml` | width 20, 7 layers, 128 Fourier modes, padding 6 |
| BFNO | `bfno_500k_20_7_128.yaml` | width 20, 7 layers, 128 Fourier modes, padding 6 |
| FNO | `fno_500k_20_7_128.yaml` | width 20, 7 Fourier layers, 128 modes each, padding 6 |
| UNet | `unet_500k.yaml` | 16 scales, skip width 4, channels listed in YAML |

Keep a positive padding value for BFNO and FNO: their original forward methods crop using `:-padding`. All supplied configurations use 500 kHz data, 480 × 480 grids, batch size 10, 50 epochs, learning rate `0.005`, AdamW weight decay `1e-5`, and gradient clipping `0.8`. `add_term: false` is retained. The original implementation uses **StepLR(step_size=6, gamma=0.1)**; its stored YAML `train.step_size`, `train.gamma`, and `train.eta_min` do not override that hardcoded scheduler. They are retained for source/checkpoint compatibility. To reproduce the supplied behavior, use the scheduler implemented in each model's `configure_optimizers`.

The configured `rel_l2` training loss is the batch mean of

```text
||prediction - target||₂ / ||target||₂
```

with all spatial locations and both real/imaginary channels flattened per sample. Validation `RRMSE` computes the same relative norm through squared norms and a square root. The source implementation has no denominator epsilon; target fields must have nonzero norm. Organ-level validation logging retains the original midpoint-of-batch assignment, now using the configured organ counts. A batch crossing an organ boundary is assigned to the organ containing its middle index.

## Optional experimental diagnostics

Experimental diagnostics are disabled by default in the public YAML files (`model.val_exp: false`), so training does not require the experimental MATLAB resources. Enable them with `--experimental-validation` when these files are available beneath the data root:

```text
resources/
├── exp_speed/
│   ├── breast0718.mat       # key: image, shape (480, 480)
│   ├── armcbs1022.mat       # key: speed, shape (480, 480)
│   └── limb0718.mat         # key: image, shape (480, 480)
└── exp_field/
    ├── breast/experimental500.mat
    ├── arm/experimental500.mat
    └── limb/experimental500.mat
```

Each field MAT file must contain complex `u_all_single` with shape `(256, 480, 480)`. The preserved diagnostic selects every fourth source (`[::4]`) to compare with 64 predicted fields. The three `speed_path_*`, `gt_path_*`, and `src_path_*` YAML settings are configurable. These diagnostics run every 1,280 validation batches, including the initial batch, and write comparison figures/curves. They can use substantial memory because each organ is evaluated with all 64 sources together. The diagnostic relative-error call retains the source argument order (`RRMSE(reference, prediction)`), so its denominator is the prediction norm; the ordinary validation denominator is the target norm.

## Outputs and resume

By default, results are written under `outputs/training/<ckpt.save_dir>/`: Lightning checkpoints, CSV logs, a resolved `config.yaml`, and optional diagnostic images. A nonempty output directory is protected: the launcher stops instead of erasing or silently starting over. Choose a new `--output-dir`, or explicitly resume:

```bash
python -m training.main_pretrain \
  --data-root /path/to/datasets \
  --output-dir outputs/training/s2no-example \
  --resume outputs/training/s2no-example/last.ckpt
```

Resume restores Lightning model, optimizer, and scheduler state. Use the same architecture/configuration. If `Project.save` is enabled, the legacy full-model export is written as `model.pt` inside the protected run directory. `Project.erase: true` is rejected; this release never automatically deletes an old training run. Generated data, logs, and checkpoints are excluded from version control.

## Release changes and checks

The release adds package imports, a portable CLI, centralized path resolution, optional experimental diagnostics, configured validation split boundaries, and output protection. It removes cluster-specific environment activation, the unconditional NCCL override, and disabled WandB logging. Experimental validation now obeys `val_exp=False`, including UNet. Model forward computations and checkpoint tensor keys are unchanged.

Small CPU checks use synthetic arrays and reduced model sizes; they verify source indexing across organ boundaries, normalization, four baseline forward/backward passes, the relative losses, optional validation, scheduler settings, and output protection:

```bash
python -m pytest training/tests -q
```

These smoke checks do not establish training convergence or reproduce the paper's quantitative results; that requires the original data and training configuration.
