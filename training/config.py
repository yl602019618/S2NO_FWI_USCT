"""Path resolution and preflight checks shared by training tools."""

import copy
import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs/s2no_500k_20_7_128.yaml"
ORGANS = ("breast", "arm", "limb")


def resolve_config(config, data_root=None):
    """Resolve all data paths once, before creating either models or loaders.

    Relative data roots and output paths are relative to the repository root.
    Absolute paths remain absolute. CLI data_root overrides S2NO_DATA_ROOT,
    which overrides data.root in YAML.
    """
    config = copy.deepcopy(config)
    data = config["data"]
    root = Path(data_root or os.environ.get("S2NO_DATA_ROOT") or data.get("root", "datasets")).expanduser()
    if not root.is_absolute():
        root = REPOSITORY_ROOT / root
    root = root.resolve()
    data["root"] = str(root)
    for key, value in data.items():
        if key.startswith("PATH_") or "_path_" in key:
            path = Path(value).expanduser()
            data[key] = str((path if path.is_absolute() else root / path).resolve())
    return config


def load_config(config_file=DEFAULT_CONFIG, data_root=None):
    import yaml

    with Path(config_file).expanduser().open() as stream:
        config = yaml.safe_load(stream)
    return resolve_config(config, data_root=data_root)


def output_directory(config, override=None):
    path = Path(override).expanduser() if override else Path(config["ckpt"]["PATH"]).expanduser() / config["ckpt"]["save_dir"]
    return (path if path.is_absolute() else REPOSITORY_ROOT / path).resolve()


def check_data(config, all_samples=False):
    """Check the configured inputs, without loading full wavefield arrays."""
    data = config["data"]
    if data["exp"] != "multi":
        raise ValueError("The released pretraining configurations support data.exp: multi (64 sources).")
    missing = []
    checked = 0
    for organ in ORGANS:
        for split, prefix, count_key in (("train", "train", "train_num"), ("test", "test", "valid_num")):
            for kind in ("data", "target"):
                directory = Path(data[f"PATH_{kind}_{split}_{organ}"])
                checked += 1
                if not directory.is_dir():
                    missing.append(str(directory))
                elif all_samples:
                    for index in range(1, data[f"{count_key}_{organ}"] + 1):
                        sample = directory / f"{prefix}_{index}.npy"
                        checked += 1
                        if not sample.is_file():
                            missing.append(str(sample))
        keys = [f"u_homo_path_{organ}"]
        if config["model"].get("val_exp", False):
            keys += [f"src_path_{organ}", f"gt_path_{organ}", f"speed_path_{organ}"]
        for key in keys:
            checked += 1
            if not Path(data[key]).is_file():
                missing.append(data[key])
    if missing:
        details = "\n".join(f"  {path}" for path in missing[:20])
        suffix = f"\n  ... and {len(missing)-20} more" if len(missing) > 20 else ""
        raise FileNotFoundError(f"Missing {len(missing)} dataset inputs under {data['root']}:\n{details}{suffix}\nSet --data-root or S2NO_DATA_ROOT; see docs/training.md.")
    return checked
