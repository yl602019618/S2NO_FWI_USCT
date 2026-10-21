"""Strict checkpoint loading and the original forward-model normalization."""

from pathlib import Path
import re

import torch


def load_s2no(checkpoint, device="cuda", *, padding=6, add_term=False):
    """Load a Lightning checkpoint or a plain state_dict into the training model.

    Width, depth and Fourier modes are inferred from parameter shapes. Padding
    and add_term are not represented by tensors: supply the training settings.
    The defaults match training/configs/s2no_500k_20_7_128.yaml.
    """
    from training.models.s2no_pretrain import S2NO_pretrain

    checkpoint = Path(checkpoint).expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Expected a Lightning checkpoint or a state_dict mapping.")
    state = payload.get("state_dict", payload)
    if "fc_0.weight" not in state or "net.0.conv0.weights1" not in state:
        raise ValueError("Checkpoint does not contain the released S2NO parameter names.")
    width, channels = state["fc_0.weight"].shape
    if channels != 5:
        raise ValueError("S2NO expects speed, two coordinates and two source channels.")
    layers = sorted({int(match.group(1)) for key in state
                     if (match := re.fullmatch(r"net\.(\d+)\.conv0\.weights1", key))})
    if layers != list(range(len(layers))):
        raise ValueError(f"Non-contiguous S2NO layer indices: {layers}")
    shape = tuple(state["net.0.conv0.weights1"].shape)
    if len(shape) != 5 or shape[:2] != (width, width) or shape[-1] != 2:
        raise ValueError(f"Unexpected Fourier weight shape: {shape}")
    if padding < 0:
        raise ValueError("padding must be nonnegative")
    architecture = dict(width=int(width), modes1=shape[2], modes2=shape[3],
                        layer_num=len(layers), padding=int(padding),
                        add_term=bool(add_term), F_feature=False,
                        dim_input=1, source_type="theta")
    model = S2NO_pretrain(**architecture, val_exp=False, loss="l1")
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    model.architecture = architecture
    return model


def predict_fields(model, speed, homogeneous):
    """Predict complex wavefields in the training array orientation and units.

    speed: real [480,480] in m/s. homogeneous: complex [B,480,480],
    the homogeneous reference fields used by the training dataset. There is no
    transpose, conjugation, source-amplitude conversion or receiver extraction.
    """
    device = next(model.parameters()).device
    speed = torch.as_tensor(speed, device=device)
    source = torch.as_tensor(homogeneous, device=device)
    if torch.is_complex(speed) or tuple(speed.shape) != (480, 480):
        raise ValueError("speed must be a real [480,480] array in m/s")
    if not torch.isfinite(speed).all() or not (speed > 0).all():
        raise ValueError("speed must contain finite, positive values")
    if (not torch.is_complex(source) or source.ndim != 3
            or tuple(source.shape[1:]) != (480, 480) or source.shape[0] == 0):
        raise ValueError("homogeneous must be complex [B,480,480], B > 0")
    if not torch.isfinite(source).all():
        raise ValueError("homogeneous fields must be finite")
    speed = speed.to(torch.float32)
    source = source.to(torch.complex64)
    normalized_speed = ((1500.0 / speed - 1.0) * 30.0)[None, ..., None]
    normalized_speed = normalized_speed.expand(source.shape[0], -1, -1, -1)
    normalized_source = torch.view_as_real(source.resolve_conj()) * 2e-3
    output = model(normalized_speed, normalized_source)
    return torch.view_as_complex(output.contiguous()) * 500.0
