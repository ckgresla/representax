import numpy as np
from structural import visible_nodes
from tax import TAX_FACES
from wider_a import (
    adaptive_nodes,
    edge_fractions,
    visible_segments,
    wider_faces,
    wider_regions,
)


def test_width_changes_preserve_t_and_rigid_x():
    original = {name: np.array(vertices) for name, _, _, vertices in TAX_FACES}
    for extra in (0, 14, 28):
        current = {name: vertices for name, _, _, vertices in wider_faces(extra)}
        for name, vertices in original.items():
            if name in ("T stem", "T stem side", "T foot", "T bar") or name.startswith(
                "A left"
            ):
                np.testing.assert_array_equal(current[name], vertices)
            elif name.startswith("X "):
                np.testing.assert_array_equal(current[name], vertices + (extra, 0))
        bridge, side = current["A bridge"], current["A bridge side"]
        np.testing.assert_array_equal(side[:2], bridge[[0, 3]])
        np.testing.assert_array_equal(side[[3, 2]], side[:2] + (14, 24))
        if extra:
            for name in ("T bar front", "T bar side"):
                assert (current[name] @ (1, 7 / 12) <= 168 + 1e-10).all()


def test_zero_expansion_is_original_geometry():
    for original, candidate in zip(TAX_FACES, wider_faces(0), strict=True):
        assert original[:3] == candidate[:3]
        np.testing.assert_array_equal(original[3], candidate[3])


def test_long_edges_have_sparse_centers_and_dense_ends():
    gaps = np.diff(edge_fractions(1.0))
    np.testing.assert_allclose(gaps[[0, -1]], 0.045)
    assert gaps[1:-1].min() > 0.1
    assert np.diff(edge_fractions(0.2)).max() * 0.2 <= 0.045


def test_original_corners_retained_and_samples_on_visible_segments():
    for extra in (0, 14, 28):
        regions = wider_regions(extra)
        corners, _ = visible_nodes(regions)
        points, _ = adaptive_nodes(regions)
        assert (
            np.linalg.norm(corners[:, None] - points[None, :], axis=-1).min(axis=1)
            < 1e-7
        ).all()
        segments = np.array(list(visible_segments(regions)))
        start, end = segments[:, 0], segments[:, 1]
        direction = end - start
        fractions = ((points[:, None] - start) * direction).sum(axis=-1) / (
            direction**2
        ).sum(axis=-1)
        nearest = start + np.clip(fractions, 0, 1)[..., None] * direction
        assert (
            np.linalg.norm(points[:, None] - nearest, axis=-1).min(axis=1) < 1e-7
        ).all()


def test_projection_scale_does_not_change_density():
    regions = wider_regions(14)
    a, _ = adaptive_nodes(regions)
    scaled = [([ring * 2 for ring in rings], group) for rings, group in regions]
    b, _ = adaptive_nodes(scaled, scale=0.5)
    assert len(a) == len(b)
    np.testing.assert_allclose(a, b / 2, atol=1e-7)
