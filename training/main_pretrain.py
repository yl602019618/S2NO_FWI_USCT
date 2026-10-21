"""Train the original S2NO, BFNO, FNO, and UNet forward operators."""

import argparse
import os
import sys
from pathlib import Path

# Support both `python training/main_pretrain.py` and `python -m training.main_pretrain`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "training"

from .config import DEFAULT_CONFIG, check_data, load_config, output_directory


def prepare_output(path, resume=None):
    """Never delete previous results; a nonempty run requires explicit resume."""
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"Output path is not a directory: {path}")
    # Lightning's DDP subprocesses share the run created by rank zero.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0 and path.exists() and any(path.iterdir()) and not resume:
        raise FileExistsError(f"Output directory is not empty: {path}. Choose --output-dir for a new run, or --resume CHECKPOINT to continue.")
    path.mkdir(parents=True, exist_ok=True)


def main(args):
    config = load_config(args.config_file, data_root=args.data_root)
    if args.no_experimental_validation:
        config["model"]["val_exp"] = False
    if args.experimental_validation:
        config["model"]["val_exp"] = True
    for key in ("accelerator", "devices"):
        if getattr(args, key) is not None:
            config["Project"][key] = getattr(args, key)
    if args.print_config:
        import yaml
        print(yaml.safe_dump(config, sort_keys=False))
        return
    checked = check_data(config, all_samples=args.check_data)
    if args.check_data:
        print(f"Dataset preflight passed ({checked} paths checked).")
        return

    # Keep `--help` usable without PyTorch or any dataset installed.
    import pytorch_lightning as pl
    import torch
    import yaml
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from .data.iter_pretrain import datasetFactory
    from .utils.model_select import model_select

    torch.set_float32_matmul_precision("medium")
    c_proj = config["Project"]
    if c_proj.get("erase", False):
        raise ValueError("Automatic output deletion is disabled. Set Project.erase: false and choose a new --output-dir.")
    if c_proj["do"] != "train":
        raise ValueError("main_pretrain.py runs training; Project.do must be train.")
    resume = args.resume or c_proj.get("ckpt_path") or None
    if resume:
        resume = str(Path(resume).expanduser().resolve())
        if not Path(resume).is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume}")
    save_file = output_directory(config, override=args.output_dir)
    prepare_output(save_file, resume=resume)
    print("Model config:", config["model"])
    print("Output directory:", save_file)
    model = model_select(config["model"], config["train"], config["data"])
    model.save_path = str(save_file)
    model.exp_name = c_proj["experiment"]

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(save_file), every_n_epochs=1, save_last=True,
        monitor="val_loss", mode="min", save_top_k=c_proj["save_top_k"],
        filename="model-{epoch:03d}-{val_loss:.4f}-{val_loss_breast:.4f}-{val_loss_arm:.4f}-{val_loss_limb:.4f}",
    )
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        # Version the effective configuration when resuming to keep the prior record.
        config_path = save_file / "config.yaml"
        index = 1
        while config_path.exists():
            config_path = save_file / f"config-resume-{index}.yaml"
            index += 1
        with config_path.open("x") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)

    train_loader, val_loader = datasetFactory(config=config, do="train")
    logger = CSVLogger(str(save_file), name=c_proj["experiment"])
    trainer_kwargs = dict(
        max_epochs=config["train"]["epochs"], accelerator=c_proj["accelerator"],
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="step")],
        gradient_clip_val=0.8, logger=logger,
    )
    if c_proj["devices"] == 1:
        trainer = pl.Trainer(devices=1, **trainer_kwargs)
    else:
        devices = list(range(c_proj["devices"])) if c_proj["accelerator"] == "gpu" else c_proj["devices"]
        trainer = pl.Trainer(devices=devices, strategy="ddp_find_unused_parameters_true", **trainer_kwargs)
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume)
    if c_proj.get("save", False) and trainer.is_global_zero:
        # Legacy optional full-model export is kept inside this protected run.
        torch.save(model, save_file / "model.pt")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config-file", "--config_file", dest="config_file", default=str(DEFAULT_CONFIG), help="YAML configuration")
    parser.add_argument("--data-root", help="Dataset root (overrides S2NO_DATA_ROOT and data.root)")
    parser.add_argument("--output-dir", help="New output directory (relative to the repository root)")
    parser.add_argument("--resume", metavar="CHECKPOINT", help="Resume model, optimizer, and scheduler state from a trusted Lightning checkpoint")
    parser.add_argument("--accelerator", choices=("cpu", "gpu", "auto", "mps"), help="Override Project.accelerator")
    parser.add_argument("--devices", type=int, help="Override Project.devices")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--experimental-validation", action="store_true", help="Enable original experimental MAT diagnostics")
    group.add_argument("--no-experimental-validation", action="store_true", help="Skip optional experimental MAT diagnostics")
    parser.add_argument("--check-data", action="store_true", help="Verify all expected input files and exit")
    parser.add_argument("--print-config", action="store_true", help="Print resolved configuration and exit")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
