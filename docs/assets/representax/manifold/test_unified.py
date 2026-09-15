import numpy as np
from unified import (
    CONFIG,
    GLYPHS,
    area,
    closed_path,
    front_points,
    layout,
    project,
    rings_for,
)


def test_whole_name_shares_one_height_baseline_and_projection():
    letters = layout()
    assert "".join(letter for letter, _, _ in letters) == "REPRESENTAX"
    assert CONFIG.text[: CONFIG.black_prefix_length] == "REPRESEN"
    for _, offset, rings in letters:
        projected = project(rings[0], offset)
        np.testing.assert_allclose(
            [projected[:, 1].min(), projected[:, 1].max()], [0, 6]
        )
        np.testing.assert_allclose(
            projected[:, 0] - rings[0][:, 0], CONFIG.shear * rings[0][:, 1] + offset
        )


def test_deterministic_lattice_preserves_glyph_counters():
    for letter in GLYPHS:
        rings = rings_for(letter)
        assert area(rings[0]) > 0
        points = front_points(rings)
        assert len(points) > 0
        np.testing.assert_array_equal(points, front_points(rings))
        np.testing.assert_allclose(
            points / CONFIG.grid_step, np.round(points / CONFIG.grid_step), atol=1e-12
        )
        for hole in rings[1:]:
            assert area(hole) < 0
            assert (
                not closed_path(hole[::-1]).contains_points(points, radius=-1e-9).any()
            )


def test_upright_prefix_matches_original_tax_height_without_shearing_it():
    from lattice import lattice
    from saturated import CONFIG as TAX_CONFIG
    from tax import TAX_FACES
    from upright import PREFIX, VARIANTS, placement

    assert PREFIX.shear == 0 and PREFIX.text == "REPRESEN"
    geometry = lattice(TAX_CONFIG, TAX_FACES)
    mapping, _ = placement(geometry)
    prefix_bounds = mapping.transform([[0, PREFIX.depth_y], [0, PREFIX.cap_height]])
    np.testing.assert_allclose(
        prefix_bounds[:, 1], [geometry[0][:, 1].min(), geometry[0][:, 1].max()]
    )
    assert mapping.get_matrix()[0, 1] == 0
    assert VARIANTS[0][2:] == (0.1, 0.1)
    assert VARIANTS[1][3] > VARIANTS[0][3]


def test_uniform_surfaces_have_no_shaded_edges_or_crowded_points():
    from dataclasses import replace

    import matplotlib.pyplot as plt
    from midtones import DEEP_PREFIX
    from scipy.spatial import cKDTree
    from unified import draw_letters, even_surface_points

    config = replace(DEEP_PREFIX, text="R")
    points = even_surface_points(rings_for("R"), config)
    distances, _ = cKDTree(points).query(points, k=2)
    assert distances[:, 1].min() >= 0.8 * config.grid_step - 1e-9
    assert np.any(points[:, 1] < 0)  # Dots also cover the extruded bottom face.
    fig, ax = plt.subplots()
    try:
        draw_letters(ax, 0.25, config)
        for patch in ax.patches:
            np.testing.assert_allclose(patch.get_facecolor(), [0.75, 0.75, 0.75, 1])
            assert patch.get_edgecolor()[3] == 0
        assert len(ax.collections) == 1  # Shared edges cannot stack dot layers.
    finally:
        plt.close(fig)


def test_italic_prefix_keeps_clearance_and_uniform_colors():
    from lattice import lattice
    from saturated import CONFIG as TAX_CONFIG
    from tax import TAX_FACES
    from tilted import VARIANTS
    from upright import placement

    geometry = lattice(TAX_CONFIG, TAX_FACES)
    for _, _, opacity, config in VARIANTS:
        assert config.shear == CONFIG.shear
        assert config.uniform_surface and opacity in (0.25, 0.40)
        mapping, _ = placement(geometry, config)
        front = np.concatenate(
            [project(rings[0], offset, config) for _, offset, rings in layout(config)]
        )
        sides = front + (config.depth_x, config.depth_y)
        bounds = mapping.transform(np.concatenate((front, sides)))
        np.testing.assert_allclose(geometry[0][:, 0].min() - bounds[:, 0].max(), 0.09)
        np.testing.assert_allclose(
            [bounds[:, 1].min(), bounds[:, 1].max()],
            [geometry[0][:, 1].min(), geometry[0][:, 1].max()],
        )


def test_optical_join_tightens_slanted_outlines_without_changing_gap_sign():
    from extended import fitted_join

    for width in (1.15, 1.30):
        tight = fitted_join(width, 0.1)
        loose = fitted_join(width, 0.2)
        assert np.isfinite(tight) and tight < 0.09
        np.testing.assert_allclose(loose - tight, 0.1)


def test_higher_resolution_preserves_relative_dot_size():
    from extended import HIRES_PREFIX
    from tilted import SHALLOW

    assert HIRES_PREFIX.width_px == 11200
    assert HIRES_PREFIX.width_px / SHALLOW.width_px == 2
    assert HIRES_PREFIX.diameter_px / SHALLOW.diameter_px == 2
    assert HIRES_PREFIX.grid_step == SHALLOW.grid_step
