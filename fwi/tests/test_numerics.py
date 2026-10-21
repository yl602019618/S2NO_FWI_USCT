"""Small CPU checks for conventions retained by the CUDA FWI sample."""

from unittest.mock import patch

import numpy as np
import torch

from fwi.geometry import make_parameters, circular_roi
from fwi.ncg import ncg
from fwi import source_s2no


def test_geometry_fortran_index_and_reciprocal_mask():
    p = make_parameters()
    assert len(p["x_src"]) == len(p["y_src"]) == 256
    np.testing.assert_array_equal(p["i_rec"], p["x_rec"] + (p["y_rec"] - 1) * 480)
    np.testing.assert_array_equal(p["x_src"], p["x_rec"])
    np.testing.assert_array_equal(p["y_src"], p["y_rec"])
    np.testing.assert_array_equal(p["select_mat"], p["select_mat"].T)
    assert not np.diag(p["select_mat"]).any()
    assert p["select_mat"][0, 128] == 1
    roi = circular_roi([240, 240], 140)
    assert roi[240, 240] == 1
    assert roi[380, 240] == 0  # strict interior, as in the original sample


def test_original_ncg_solves_bounded_quadratic():
    x0 = torch.tensor([1., -.8, .8, -.2], dtype=torch.float64)
    target = torch.tensor([.1, .2, -.4, .7], dtype=x0.dtype)
    weights = torch.tensor([1., 2., 3., 4.], dtype=x0.dtype)

    def objective(x):
        residual = x - target
        return .5 * (weights * residual.square()).sum(), weights * residual

    out = ncg({
        "MaxIters": 20, "StopTol": 1e-10, "RelFuncTol": 1e-12,
        "LineSearch_ibound": True,
        "LineSearch_lb": torch.full_like(x0, -2),
        "LineSearch_ub": torch.full_like(x0, 2),
    }, objective, x0)
    torch.testing.assert_close(out.X, target, atol=1e-8, rtol=1e-8)
    assert out.F < 1e-15


def test_original_s2no_phase_order_is_preserved():
    # Conjugating before division would rotate the reciprocal adjoint's phase.
    raw = torch.tensor([[[1+2j, 3-4j], [-2+.5j, .3-.7j]]], dtype=torch.complex64)
    bank = torch.zeros_like(raw)
    with patch.object(source_s2no, "matpy_prediction", return_value=raw):
        actual = source_s2no.batch_cal_wvfields(None, [0], bank, None)
    amp = 1.9514e9 - 2.7546e8j
    expected = (raw / amp * 1e10).transpose(2, 1).conj()
    changed_phase = raw.transpose(2, 1).conj() / amp * 1e10
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, changed_phase)


def test_external_complex128_is_normalized_to_solver_precision(tmp_path):
    from scipy.io import savemat
    from fwi.run import load_complex_array

    field = np.array([[1 + 2j, -.5 + .125j]], dtype=np.complex128)
    np.save(tmp_path / "field.npy", field)
    savemat(tmp_path / "field.mat", {"u_all_single": field})
    for name in ["field.npy", "field.mat"]:
        loaded = load_complex_array(tmp_path / name, "u_all_single")
        assert loaded.dtype == np.complex64
        np.testing.assert_array_equal(loaded, field)
