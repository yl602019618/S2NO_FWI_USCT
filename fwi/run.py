"""Portable orchestration of the original multiscale CBS/S2NO FWI sample."""

import argparse
import json
import logging
from pathlib import Path
import random
import time

import numpy as np
import yaml

from .geometry import circular_roi, make_parameters


def load_array(path, key=None):
    """Load NPY or a conventional (non-v7.3) MAT array without axis guessing."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix == ".mat":
        from scipy.io import loadmat
        arrays = loadmat(path)
        if key not in arrays:
            raise ValueError(f"{path} does not contain MAT variable {key!r}")
        return arrays[key]
    raise ValueError(f"Expected a .npy or .mat array: {path}")


def load_complex_array(path, key=None):
    """Match external MAT/NPY precision to the original complex64 field buffers."""
    return np.asarray(load_array(path, key), dtype=np.complex64)


def require_array(array, shape, path, complex_required=False):
    if array.shape != shape:
        raise ValueError(f"{path}: expected shape {shape}, received {array.shape}")
    if complex_required and not np.iscomplexobj(array):
        raise ValueError(f"{path}: expected a complex-valued array")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: contains non-finite values")
    return array


def input_path(root, pattern, sample, frequency):
    return root / pattern.format(sample=sample, frequency=frequency)


def checkpoint_path(root, pattern, frequency):
    matches = sorted(root.glob(pattern.format(frequency=frequency)))
    if len(matches) != 1 or not matches[0].is_file():
        raise ValueError(
            f"Expected exactly one checkpoint for {frequency} kHz under {root} "
            f"matching {pattern!r}; found {len(matches)}. Set paths.checkpoint in the YAML."
        )
    return matches[0]


def selected_stages(config, limit=None):
    stages = []
    for sweep, frequencies in enumerate(config["frequencies_khz"]):
        if len(config["model_smoothing"][sweep]) != len(frequencies):
            raise ValueError("model_smoothing must match each frequency sweep")
        if len(config["gradient_smoothing"][sweep]) != len(frequencies):
            raise ValueError("gradient_smoothing must match each frequency sweep")
        for index, frequency in enumerate(frequencies):
            stages.append((sweep, index, int(frequency)))
    if not stages:
        raise ValueError("At least one frequency stage is required")
    return stages if limit is None else stages[:limit]


def validate_inputs(config, args, stages):
    paths = config["paths"]
    frequencies = sorted({stage[2] for stage in stages})
    for frequency in frequencies:
        observed = input_path(args.data_root, paths["observations"], args.sample, frequency)
        require_array(load_array(observed, paths.get("observation_key", "data")),
                      (256, 256), observed, complex_required=True)
        if args.solver == "s2no":
            homogeneous = input_path(args.data_root, paths["homogeneous"], args.sample, frequency)
            require_array(load_array(homogeneous, paths.get("homogeneous_key", "u_all_single")),
                          (256, 480, 480), homogeneous, complex_required=True)
            checkpoint_path(args.checkpoint_root, paths["checkpoint"], frequency)


def optimize_stage(v0, objective, arguments, config, sigma_grad):
    """Original stage normalization, slowness bounds and NCG call."""
    import torch
    from .ncg import ncg

    slowness = torch.tensor(1 / v0.detach().cpu().numpy().reshape(-1, order="F"))
    # The source code uses the objective's default sigma_grad=2 in this scaling probe.
    f0, g0 = objective(x_sp=slowness, fscale=1, gscale=1, **arguments)
    fscale = f0 / 1e3
    gscale = torch.norm(g0.reshape(-1)) / torch.norm(1 / v0.reshape(-1)) * 1e3
    if not bool(torch.isfinite(fscale)) or not bool(torch.isfinite(gscale)):
        raise FloatingPointError("Non-finite stage normalization; inspect observations and fields")
    if float(fscale) == 0 or float(gscale) == 0:
        raise FloatingPointError("Zero stage normalization; the original NCG scaling is undefined")

    def fun(x):
        return objective(x_sp=x, fscale=fscale, gscale=gscale,
                         sigma_grad=sigma_grad, **arguments)

    options = {
        "MaxIters": config["max_iterations"],
        "StopTol": 1e-13,
        "RelFuncTol": 1e-10,
        "LineSearch_ibound": True,
        "LineSearch_ub": torch.ones_like(slowness).cuda() / 1400,
        "LineSearch_lb": torch.ones_like(slowness).cuda() / 2000,
        "LineSearch_maxfev": config["max_line_search_evaluations"],
        "TraceX": True,
        "TraceFunc": True,
    }
    result = ncg(options, fun, slowness.cuda())
    speed = 1 / result.X.detach().cpu().numpy().reshape(480, 480, order="F")
    return torch.tensor(speed), {
        "iterations": result.Iters,
        "function_evaluations": result.FuncEvals,
        "exit_flag": result.ExitFlag,
        "exit_description": result.ExitDescription,
        "objective": float(result.F),
        "objective_trace": [float(value) for value in result.TraceFunc],
        "fscale": float(fscale), "gscale": float(gscale),
    }


def execute(config, args, stages):
    import torch
    from ._config import Bunch
    from .smoothing import blur

    if not torch.cuda.is_available():
        raise RuntimeError("This original FWI implementation requires an NVIDIA CUDA GPU")
    torch.cuda.set_device(args.device)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Refuse replacement of any existing output, including a partially completed run.
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "stages").mkdir()
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output / "run.log")])
    metadata = {
        "solver": args.solver, "sample": args.sample, "config": config,
        "data_root": str(args.data_root.resolve()),
        "checkpoint_root": str(args.checkpoint_root.resolve()) if args.checkpoint_root else None,
        "device": args.device, "stage_limit": args.max_stages,
        "optimizer": "original PR+ NCG / More-Thuente",
        "phase_convention": "original source_estimation.py",
        "frequency_fix": "param.freq = frequency_khz / 100 at each stage",
    }
    (args.output / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")
    param = make_parameters()
    initial = config["initial_model"]
    v0 = initial["speed_m_s"] * torch.ones(480, 480) * circular_roi(initial["center"], initial["radius"])
    v0[v0 == 0] = 1500
    v0 = blur(v0, sigma=10)
    np.save(args.output / "initial_speed.npy", v0.numpy())
    paths = config["paths"]
    reconstructions, reports = [], []
    start = time.perf_counter()
    for sweep, index, frequency in stages:
        stage_start = time.perf_counter()
        stage_name = f"r{sweep + 1}_{frequency}"
        logging.info("Stage %s (%s)", stage_name, args.solver)
        param["roi"] = circular_roi(config["roi"]["center"], config["roi"]["radii"][sweep])
        # Unlike the legacy S2NO entry point, update the frequency at every stage.
        param["freq"] = frequency / 100
        observation_path = input_path(args.data_root, paths["observations"], args.sample, frequency)
        data = torch.from_numpy(load_complex_array(observation_path, paths.get("observation_key", "data")))
        arguments = dict(dobs=data, param=param, name=stage_name)
        if args.solver == "cbs":
            from .objective_cbs import cal_fcost_grad
            # CBS mutates its configuration; use a fresh dictionary for each stage.
            arguments["config"] = Bunch(CBSConfig=Bunch(config["cbs"]))
        else:
            from inference.model import load_s2no
            from .objective_s2no import cal_fcost_grad
            model_config = config["s2no"]
            model_path = checkpoint_path(args.checkpoint_root, paths["checkpoint"], frequency)
            model = load_s2no(model_path, device="cuda", padding=model_config["padding"],
                              add_term=model_config["add_term"])
            homogeneous_path = input_path(args.data_root, paths["homogeneous"], args.sample, frequency)
            homogeneous = torch.from_numpy(load_complex_array(homogeneous_path, paths.get("homogeneous_key", "u_all_single")))
            arguments.update(model=model, u_allsource=homogeneous, batchsize=model_config["batch_size"])
        v_out, report = optimize_stage(v0, cal_fcost_grad, arguments, config,
                                       config["gradient_smoothing"][sweep][index])
        reconstruction = v_out.numpy()
        if not np.isfinite(reconstruction).all():
            raise FloatingPointError(f"Non-finite reconstruction at {stage_name}")
        np.save(args.output / "stages" / f"{stage_name}.npy", reconstruction)
        reconstructions.append(reconstruction)
        report.update(stage=stage_name, frequency_khz=frequency,
                      elapsed_seconds=time.perf_counter() - stage_start)
        reports.append(report)
        (args.output / "stages" / f"{stage_name}.json").write_text(json.dumps(report, indent=2) + "\n")
        sigma = config["model_smoothing"][sweep][index]
        v0 = blur(v_out, sigma=sigma) if sigma != 0 else v_out
        # Preserve the original end-of-first-sweep / every-second-sweep ROI reset.
        if index == 7 or sweep > 0:
            v0 = v0 * param["roi"]
            v0[v0 == 0] = 1500
        if args.solver == "s2no":
            del arguments, model, homogeneous
            torch.cuda.empty_cache()
    np.save(args.output / "vdata.npy", np.stack(reconstructions))
    np.save(args.output / "final_speed.npy", reconstructions[-1])
    (args.output / "summary.json").write_text(json.dumps({
        "solver": args.solver, "sample": args.sample, "stages_completed": len(reports),
        "elapsed_seconds": time.perf_counter() - start, "stages": reports,
    }, indent=2) + "\n")
    logging.info("Saved %s", args.output / "final_speed.npy")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--solver", choices=("cbs", "s2no"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample", type=int, help="Override the sample number in the YAML")
    parser.add_argument("--device", type=int, default=0, help="Visible CUDA device index")
    parser.add_argument("--validate-only", action="store_true", help="Check input arrays and paths without using a GPU")
    parser.add_argument("--max-stages", type=int, help="Limit stages for a smoke run, not a full reconstruction")
    parser.add_argument("--max-iterations", type=int, help="Override the per-stage NCG budget; recorded in protocol.json")
    args = parser.parse_args(argv)
    if args.solver == "s2no" and args.checkpoint_root is None:
        parser.error("--checkpoint-root is required for S2NO")
    if not args.validate_only and args.output is None:
        parser.error("--output is required unless --validate-only is set")
    if args.max_stages is not None and args.max_stages < 1:
        parser.error("--max-stages must be positive")
    if args.max_iterations is not None and args.max_iterations < 1:
        parser.error("--max-iterations must be positive")
    config = yaml.safe_load(args.config.read_text())
    if args.sample is None:
        args.sample = config["sample"]
    if args.max_iterations is not None:
        config["max_iterations"] = args.max_iterations
    stages = selected_stages(config, args.max_stages)
    validate_inputs(config, args, stages)
    if args.validate_only:
        print(f"Validated {args.solver} inputs for sample {args.sample}, {len(stages)} stages")
        return
    execute(config, args, stages)


if __name__ == "__main__":
    main()
