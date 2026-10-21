"""Check strict loading and normalization against a direct model call."""

import numpy as np
import pytest
import torch

from inference.model import load_s2no, predict_fields
from inference.predict import main
from training.models.s2no_pretrain import S2NO_pretrain


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(7)
    torch.set_num_threads(2)
    model = S2NO_pretrain(width=2, modes1=3, modes2=3, layer_num=1,
                         padding=6, add_term=False, val_exp=False, loss="l1").eval()
    path = tmp_path / "tiny.ckpt"
    torch.save({"state_dict": model.state_dict()}, path)
    return model, path


def test_loading_and_physical_normalization(checkpoint):
    original, path = checkpoint
    loaded = load_s2no(path, "cpu")
    speed = torch.full((480, 480), 1550.0)
    field = torch.randn(2, 480, 480, dtype=torch.complex64)
    with torch.inference_mode():
        direct = original(((1500 / speed - 1) * 30)[None, ..., None].expand(2, -1, -1, -1),
                          torch.view_as_real(field) * 2e-3)
        expected = torch.view_as_complex(direct.contiguous()) * 500
        actual = predict_fields(loaded, speed, field)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not loaded.training
    assert not any(p.requires_grad for p in loaded.parameters())


def test_cli_preserves_last_partial_batch(checkpoint, tmp_path):
    _, path = checkpoint
    speed = np.full((480, 480), 1500, np.float32)
    fields = np.full((3, 480, 480), 1 + 2j, np.complex64)
    np.save(tmp_path / "speed.npy", speed)
    np.save(tmp_path / "homo.npy", fields)
    output = tmp_path / "prediction.npy"
    argv = ["--checkpoint", str(path), "--speed", str(tmp_path / "speed.npy"),
            "--homogeneous", str(tmp_path / "homo.npy"), "--output", str(output),
            "--device", "cpu", "--batch-size", "2"]
    main(argv)
    result = np.load(output)
    assert result.shape == fields.shape
    assert result.dtype == np.complex64
    assert np.isfinite(result).all()
    with pytest.raises(FileExistsError):
        main(argv)


def test_invalid_inputs_and_mismatched_checkpoint(checkpoint, tmp_path):
    _, path = checkpoint
    model = load_s2no(path, "cpu")
    with pytest.raises(ValueError, match="positive"):
        predict_fields(model, torch.zeros(480, 480), torch.ones(1, 480, 480, dtype=torch.complex64))
    state = torch.load(path, weights_only=True)["state_dict"]
    del state["fc2.weight"]
    bad = tmp_path / "incomplete.ckpt"
    torch.save({"state_dict": state}, bad)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_s2no(bad, "cpu")
