# Source release notes

This release reorganizes the supplied research implementation. Its training
source is `S2NO_main_clean.tar.gz`; its FWI numerical core comes from
`USCT_S2NO_FWI_Phantom_main/inversion/arm_small`. Original source fingerprints
are listed in [source-provenance.json](source-provenance.json). The original
archives, trained weights, datasets, measurements and run outputs are not in Git.

## What changed

- Training, forward inference and FWI have separate module entry points, with
  shared S2NO model definitions and portable paths.
- Cluster launch scripts were replaced with ordinary Python/shell commands.
  Existing nonempty outputs are protected. Training can resume explicitly.
- Experimental training diagnostics are opt-in and require their own assets.
  The network equations, training losses, sampler and optimizer schedule remain
  those of the supplied training package.
- FWI retains the original CBS propagation, complex source fit, reciprocal
  adjoint gradient, Gaussian filters, PR+ NCG and More–Thuente search. Geometry
  is exported exactly from the original MAT file into text constants.
- Model loading infers tensor-defined architecture settings and loads strictly.
  Padding and the optional multiplicative output term remain explicit settings.
- The S2NO FWI entry point now updates its frequency parameter at each stage,
  matching the CBS entry point. External complex arrays are converted to the
  original single-precision solver format. See [FWI](fwi.md) for the retained
  phase convention and the complete list of adaptations.

## Verification

Run the lightweight checks with:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The CPU suite covers training data normalization, small-model forward/backward
passes, optimizer setup, safe output handling, checkpoint round trips, forward
inference including the final partial batch, exact geometry conventions, the
original NCG on a bounded quadratic, S2NO phase ordering, and array precision.
These checks do not require datasets or pretrained weights.

During release preparation, the original and reorganized implementations were
also compared directly: all four networks with identical weights produced
bitwise-equal outputs on reduced fixtures; the CBS propagator, NCG trajectory,
and both FWI objective/gradient routines matched their originals on numerical
fixtures. Original geometry and masks were compared element by element. These
are implementation checks, not a repetition of the paper's full training or
quantitative reconstruction experiments.

The [training](training.md), [inference](inference.md), and [FWI](fwi.md) guides
describe the external inputs needed for larger experiments. MIT applies to the
code in this repository; consult the respective sources for dataset terms.
