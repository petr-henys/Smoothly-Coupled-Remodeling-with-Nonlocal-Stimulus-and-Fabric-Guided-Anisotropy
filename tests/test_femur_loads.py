import numpy as np
import pytest

from femur import HIPJointLoad


class _DummyMesh:
    def __init__(self, hits):
        self._hits = hits

    def ray_trace(self, start, end, first_point=True):
        return None, list(self._hits)


def _make_minimal_hip_load(*, hits):
    # Avoid heavy pyvista construction: create a bare instance and patch attributes.
    hip = HIPJointLoad.__new__(HIPJointLoad)

    hip.head_center_world = np.zeros(3)
    hip.head_radius = 2.0
    hip.centers_css = np.array([[1.0, 0.0, 0.0]])
    hip.mesh_world = _DummyMesh(hits)

    # Minimal logger stub
    hip.logger = type("_L", (), {"warning": lambda *_, **__: None})()

    # Patch force resolution: treat CSS as world for the unit test.
    hip._resolve_force_vector = lambda f: (np.asarray(f, dtype=float), float(np.linalg.norm(f)))

    return hip


def test_get_contact_point_css_raises_on_ray_miss_by_default():
    hip = _make_minimal_hip_load(hits=[])
    with pytest.raises(RuntimeError, match="Ray cast missed"):
        hip.get_contact_point_css(np.array([0.0, 0.0, 10.0]))


def test_get_contact_point_css_allows_explicit_fallback_on_ray_miss():
    hip = _make_minimal_hip_load(hits=[])
    p = hip.get_contact_point_css(
        np.array([0.0, 0.0, 10.0]),
        allow_ray_miss_fallback=True,
    )
    assert np.allclose(p, np.array([0.0, 0.0, hip.head_radius]))


def test_get_contact_point_css_returns_hit_center_when_hit_exists():
    hip = _make_minimal_hip_load(hits=[0])
    p = hip.get_contact_point_css(np.array([0.0, 0.0, 10.0]))
    assert np.allclose(p, hip.centers_css[0])
