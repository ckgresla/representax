import matplotlib.pyplot as plt
import numpy as np
from structural import fill_regions, orient, visible_nodes


def test_nodes_are_visible_corners_and_actual_occlusion_junctions():
    back = orient(((0, 0), (3, 0), (3, 2), (0, 2)))
    front = orient(((1, -1), (4, -1), (4, 3), (1, 3)))
    points, groups = visible_nodes([([back], 0), ([front], 1)])
    actual = {tuple(point): group for point, group in zip(points, groups, strict=True)}
    assert (3.0, 0.0) not in actual and (3.0, 2.0) not in actual
    assert actual[(1.0, 0.0)] == actual[(1.0, 2.0)] == 1
    assert set(actual) == {
        (0.0, 0.0),
        (0.0, 2.0),
        (1.0, 0.0),
        (1.0, 2.0),
        (1.0, -1.0),
        (4.0, -1.0),
        (4.0, 3.0),
        (1.0, 3.0),
    }


def test_one_constant_fill_per_color_without_internal_outlines():
    regions = [
        ([orient(((0, 0), (2, 0), (2, 2), (0, 2)))], 0),
        ([orient(((1, 1), (3, 1), (3, 3), (1, 3)))], 0),
    ]
    fig, ax = plt.subplots()
    try:
        fill_regions(ax, regions, np.zeros((1, 3)))
        assert len(ax.patches) == 1
        assert not ax.images and not ax.collections
        np.testing.assert_allclose(ax.patches[0].get_facecolor(), [0.6, 0.6, 0.6, 1])
        assert ax.patches[0].get_edgecolor()[3] == 0
    finally:
        plt.close(fig)


def test_added_dots_stay_on_visible_edges_and_retain_original_junctions():
    back = orient(((0, 0), (3, 0), (3, 2), (0, 2)))
    front = orient(((1, -1), (4, -1), (4, 3), (1, 3)))
    regions = [([back], 0), ([front], 1)]
    corners, _ = visible_nodes(regions)
    points, _ = visible_nodes(regions, edge_spacing=0.25)
    assert set(map(tuple, corners)) <= set(map(tuple, points))
    assert len(points) > len(corners)
    for x, y in points:
        on_back = (x == 0 and 0 <= y <= 2) or (y in (0, 2) and 0 <= x <= 1)
        on_front = (x in (1, 4) and -1 <= y <= 3) or (y in (-1, 3) and 1 <= x <= 4)
        assert on_back or on_front
    top = np.sort(points[np.isclose(points[:, 1], 3), 0])
    assert np.diff(top).max() <= 0.25 + 1e-10


def test_fill_and_point_opacity_are_independent():
    from structural import scatter

    fig, ax = plt.subplots()
    try:
        fill_regions(
            ax,
            [([orient(((0, 0), (1, 0), (1, 1), (0, 1)))], 0)],
            np.zeros((1, 3)),
            opacity=0.2,
        )
        scatter(ax, np.array([[0, 0], [1, 1]]), np.zeros((2, 3)), opacity=0.8)
        np.testing.assert_allclose(ax.patches[0].get_facecolor(), [0.8, 0.8, 0.8, 1])
        assert ax.collections[0].get_alpha() == 0.8
    finally:
        plt.close(fig)
