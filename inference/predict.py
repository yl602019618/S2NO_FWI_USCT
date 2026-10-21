"""Predict wavefields for one speed map and a batch of source reference fields."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--speed", type=Path, required=True, help="Real [480,480] NPY, m/s")
    parser.add_argument("--homogeneous", type=Path, required=True,
                        help="Complex [sources,480,480] NPY in training field units")
    parser.add_argument("--output", type=Path, required=True, help="New complex NPY file")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0 or cpu")
    parser.add_argument("--padding", type=int, default=6)
    parser.add_argument("--add-term", action="store_true", help="Match train.add_term: true")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.output.suffix != ".npy":
        parser.error("--output must end in .npy")
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")

    import json
    import numpy as np
    import torch
    from .model import load_s2no, predict_fields

    speed = np.load(args.speed, allow_pickle=False)
    homogeneous = np.load(args.homogeneous, mmap_mode="r", allow_pickle=False)
    if (homogeneous.ndim != 3 or homogeneous.shape[1:] != (480, 480)
            or len(homogeneous) == 0 or not np.iscomplexobj(homogeneous)):
        raise ValueError("--homogeneous must contain complex [sources,480,480] fields")
    model = load_s2no(args.checkpoint, args.device, padding=args.padding, add_term=args.add_term)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # A temporary sibling keeps a failed run from looking like a complete output.
    import tempfile
    with tempfile.TemporaryDirectory(prefix=".predict-", dir=args.output.parent) as temporary:
        partial = Path(temporary) / args.output.name
        result = np.lib.format.open_memmap(partial, mode="w+", dtype=np.complex64,
                                          shape=homogeneous.shape)
        with torch.inference_mode():
            for start in range(0, len(homogeneous), args.batch_size):
                stop = min(start + args.batch_size, len(homogeneous))
                fields = predict_fields(model, speed, np.array(homogeneous[start:stop]))
                if not torch.isfinite(fields).all():
                    raise FloatingPointError("The model produced non-finite wavefields")
                result[start:stop] = fields.cpu().numpy()
        result.flush()
        del result
        # Hard-link creation is atomic and refuses to replace an existing file.
        args.output.hardlink_to(partial)
    print(json.dumps({"output": str(args.output.resolve()), "shape": list(homogeneous.shape),
                      "dtype": "complex64", "architecture": model.architecture}, indent=2))


if __name__ == "__main__":
    main()
