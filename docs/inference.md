# Forward inference

The inference entry point reuses `training.models.s2no_pretrain.S2NO_pretrain`.
It accepts a Lightning checkpoint (`state_dict`) or a plain PyTorch state dict.
Parameter names must match exactly; width, layer count, and Fourier modes are
read from the tensors. No weights or validation datasets are loaded implicitly.

```bash
python -m inference.predict \
  --checkpoint /path/to/model.ckpt \
  --speed /path/to/speed.npy \
  --homogeneous /path/to/homogeneous.npy \
  --output outputs/wavefields.npy \
  --batch-size 8
```

| Input / output | Shape | Meaning |
| --- | --- | --- |
| `--speed` | `(480, 480)` real | Positive sound speed in m/s |
| `--homogeneous` | `(sources, 480, 480)` complex | Homogeneous reference fields in the training data's orientation and units |
| `--output` | `(sources, 480, 480)` complex64 | Predicted fields in that same orientation and units |

All arrays are NumPy `.npy` files. The source reference must have the same
frequency, source locations, source amplitudes and sampling grid as the training
data for the checkpoint. This command does not synthesize reference fields.
The training data uses 64 sampled sources; a checkpoint can also be evaluated
with the corresponding 256-source reference for FWI, in manageable batches.

The original normalization is preserved:

```python
speed_input = (1500 / speed - 1) * 30
source_input = stack([homogeneous.real, homogeneous.imag], axis=-1) * 2e-3
wavefield = (prediction[..., 0] + 1j * prediction[..., 1]) * 500
```

No conjugation, transpose, receiver sampling or FWI source-amplitude convention
is applied here. Those operations belong to the existing FWI implementation;
see [FWI](fwi.md).

`--padding 6` and the default `add_term=False` match the released 500 kHz training
configuration. These two settings have no tensors in the checkpoint, so they
cannot be inferred. Set `--padding` and `--add-term` to match a different training
configuration. The release assumes the original `F_feature=False` setting.

Use `--device cpu` for small integration checks; production-sized checkpoints
are intended for a CUDA GPU. Reduce `--batch-size` when memory is limited.
Existing output files are never overwritten. A failed run does not leave a
completed output file.
