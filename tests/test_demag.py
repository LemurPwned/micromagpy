import numpy as np


def test_tensor_disk_cache(tmp_path, monkeypatch):
    from micromagpy.demag import compute_demag_tensor_real

    monkeypatch.setenv("MICROMAGPY_CACHE", str(tmp_path))
    a = compute_demag_tensor_real((256, 256, 1), (4e-9, 4e-9, 3e-9))
    assert len(list(tmp_path.glob("newell_*.npy"))) == 1
    np.testing.assert_array_equal(compute_demag_tensor_real((256, 256, 1), (4e-9, 4e-9, 3e-9)), a)
    monkeypatch.setenv("MICROMAGPY_CACHE", "")
    np.testing.assert_array_equal(compute_demag_tensor_real((256, 256, 1), (4e-9, 4e-9, 3e-9)), a)
