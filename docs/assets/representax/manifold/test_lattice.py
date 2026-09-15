from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from lattice import LatticeConfig, canvas, face_labels, lattice
from matplotlib.colors import rgb_to_hsv, to_rgb


def test_all_vertices_are_distinct_lattice_intersections():
    cfg = LatticeConfig()
    xy, triangles, faces, labels = lattice(cfg)
    drawing = xy * (100, -100) + (126, 72)
    j = drawing[:, 1] / (24 / cfg.divisions)
    i = drawing[:, 0] / (28 / cfg.divisions) - j / 2
    np.testing.assert_allclose(i, np.round(i), atol=1e-12)
    np.testing.assert_allclose(j, np.round(j), atol=1e-12)
    assert len(np.unique(xy, axis=0)) == len(xy)
    assert len(np.unique(triangles)) == len(xy)
    assert np.all(labels >= 0)
    np.testing.assert_array_equal(face_labels(xy[triangles].mean(axis=1)), faces)
    np.testing.assert_array_equal(xy, lattice(cfg)[0])
    # This point is inside the central opening, not a face to be interpolated.
    aperture = (np.array([[142, 60]]) - (126, 72)) * (0.01, -0.01)
    assert face_labels(aperture)[0] == -1


def test_wider_spacing_is_a_nested_lattice_not_another_random_cloud():
    config = LatticeConfig()
    dense = {tuple(p) for p in np.round(lattice(config)[0], 10)}
    coarse = {tuple(p) for p in np.round(lattice(replace(config, divisions=4))[0], 10)}
    assert coarse < dense
    with pytest.raises(ValueError):
        lattice(replace(config, divisions=3))


def test_face_tint_is_precomposed_without_lines_or_alpha_seams():
    config = LatticeConfig(divisions=2, size_px=400)
    geometry = lattice(config)
    fig, xyz, screen = canvas(config, *geometry)
    try:
        assert np.all(np.isfinite(xyz))
        assert np.all(np.isfinite(screen))
        for mesh in fig.axes[0].collections[:-1]:
            rgba = mesh.get_facecolors()
            np.testing.assert_array_equal(rgba[:, 3], 1)
            assert np.all(rgba[:, :3] >= 1 - config.fill_opacity)
            assert mesh.get_edgecolors().size == 0
        assert len(fig.axes[0].collections[-1].get_offsets()) == len(geometry[0])
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        assert np.mean(np.all(rgb == 255, axis=-1)) > 0.7
        assert np.any(rgb < 230)
    finally:
        plt.close(fig)


def test_saturated_mark_is_level_with_exact_ten_percent_face_opacity():
    from generate import embed, project
    from saturated import CONFIG

    assert CONFIG.deformation == 0
    hsv = rgb_to_hsv(np.array([to_rgb(color) for color in CONFIG.palette]))
    np.testing.assert_array_equal(hsv[:, 1], 1)
    xy = lattice(CONFIG)[0]
    xyz, _ = embed(xy, CONFIG.deformation)
    np.testing.assert_array_equal(xyz[:, :2], xy)
    np.testing.assert_array_equal(xyz[:, 2], 0)
    assert CONFIG.yaw_degrees == CONFIG.pitch_degrees == 0
    assert CONFIG.fill_opacity == 0.1 and CONFIG.gradient_floor == 1
    screen = project(xyz, CONFIG)
    np.testing.assert_array_equal(screen[:, :2], xy)
    baseline = np.isclose(xy[:, 1], -0.72)
    assert baseline.sum() > 10
    assert np.ptp(screen[baseline, 1]) == 0


def test_tax_replaces_the_hook_with_a_top_bar_without_changing_style():
    from saturated import CONFIG
    from tax import TAX_FACES

    assert not any(face[0].startswith("J ") for face in TAX_FACES)
    points = (np.array([[100, 12], [14, 96]]) - (126, 72)) * (0.01, -0.01)
    labels = face_labels(points, TAX_FACES)
    assert TAX_FACES[labels[0]][0] == "T bar"
    assert labels[1] == -1
    xy, triangles, faces, _ = lattice(CONFIG, TAX_FACES)
    np.testing.assert_array_equal(
        face_labels(xy[triangles].mean(axis=1), TAX_FACES), faces
    )
