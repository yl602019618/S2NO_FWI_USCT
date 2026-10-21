"""Small CPU checks for the release interfaces; no training data or weights needed."""

from pathlib import Path

import numpy as np
import pytest
import torch

from training.config import DEFAULT_CONFIG, load_config, resolve_config
from training.data.iter_pretrain import File_Loader
from training.main_pretrain import prepare_output
from training.utils.loss import LpLoss, RRMSE
from training.utils.model_select import model_select


@pytest.mark.parametrize("name", ["s2no", "bfno", "fno", "unet"])
def test_four_baselines_forward_backward_and_optional_validation(name):
    config_name = f"{name}_500k" + ("_20_7_128" if name != "unet" else "") + ".yaml"
    config = load_config(DEFAULT_CONFIG.parent / config_name)
    config["model"].update(val_exp=False, features=4, layer_num=2, padding=2)
    config["model"]["wavenumber"] = [2, 2] if name == "fno" else 2
    if name == "unet":
        config["model"].update(scales=2, channels=[8, 8])
    model = model_select(config["model"], config["train"], config["data"])
    sos = torch.randn(2, 16, 16, 1)
    src = torch.randn(2, 16, 16, 2)
    target = torch.randn(2, 16, 16, 2)
    prediction = model(sos, src)
    assert prediction.shape == target.shape
    assert torch.isfinite(prediction).all()
    loss = model.criterion(prediction, target)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
    # The original val_exp=False option must work at val_iter=0, with no MAT files.
    model.log = lambda *args, **kwargs: None
    assert torch.isfinite(model.validation_step((sos, src, target, torch.tensor([0, 1])), 0))
    optimizers = model.configure_optimizers()
    assert isinstance(optimizers["optimizer"], torch.optim.AdamW)
    scheduler = optimizers["lr_scheduler"]["scheduler"]
    assert scheduler.step_size == 6
    assert scheduler.gamma == 0.1


def test_loader_source_index_organ_boundaries_and_normalization(tmp_path):
    size = 4
    data_paths, target_paths, source_paths = [], [], []
    for organ in range(3):
        data_path = tmp_path / f"speed-{organ}.npy"
        target_path = tmp_path / f"target-{organ}.npy"
        source_path = tmp_path / f"source-{organ}.npy"
        np.save(data_path, np.full((size, size), 1600 + organ, dtype=np.float32))
        target = np.full((64, size, size), 500 + 750j, dtype=np.complex64)
        source = np.broadcast_to((100 * organ + np.arange(64) + 1j)[:, None, None], (64, size, size)).astype(np.complex64)
        np.save(target_path, target)
        np.save(source_path, source)
        data_paths.append(str(data_path))
        target_paths.append(str(target_path))
        source_paths.append(str(source_path))
    loader = File_Loader(data_paths, target_paths, *source_paths, size=size, exp="multi", split_num=[1, 1])
    assert len(loader) == 3 * 64
    for index in [0, 63, 64, 127, 128, 191]:
        organ, source_index = divmod(index, 64)
        sos, source, target, returned_index = loader[index]
        assert sos.shape == (size, size, 1)
        assert source.shape == target.shape == (size, size, 2)
        torch.testing.assert_close(sos, torch.full_like(sos, (1500 / (1600 + organ) - 1) * 30))
        torch.testing.assert_close(source[..., 0], torch.full((size, size), (100 * organ + source_index) * 2e-3))
        torch.testing.assert_close(source[..., 1], torch.full((size, size), 2e-3))
        torch.testing.assert_close(target[..., 0], torch.ones(size, size))
        torch.testing.assert_close(target[..., 1], torch.full((size, size), 1.5))
        assert returned_index.item() == index


def test_relative_loss_definition():
    target = torch.ones(2, 4, 4, 2)
    prediction = 1.25 * target
    for loss in (LpLoss(), RRMSE()):
        torch.testing.assert_close(loss(prediction, target), torch.tensor(0.25))


def test_data_root_precedence_and_output_protection(tmp_path, monkeypatch):
    monkeypatch.setenv("S2NO_DATA_ROOT", str(tmp_path / "environment"))
    config = load_config()
    assert config["data"]["PATH_data_train_breast"] == str(tmp_path / "environment/breast_data/speed/train")
    explicit = load_config(data_root=tmp_path / "explicit")
    assert explicit["data"]["u_homo_path_arm"] == str(tmp_path / "explicit/arm_data/500k/train/train_7154.npy")
    absolute = resolve_config({"data": {"PATH_sample": "/absolute/sample.npy"}}, data_root=tmp_path)
    assert absolute["data"]["PATH_sample"] == "/absolute/sample.npy"
    output = tmp_path / "run"
    prepare_output(output)
    sentinel = output / "previous-result"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        prepare_output(output)
    assert sentinel.read_text() == "keep"
