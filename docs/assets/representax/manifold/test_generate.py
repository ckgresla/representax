import numpy as np
from generate import Config, cloud, domain_labels, embed, sample_domain, save_mark
from PIL import Image


def test_sampling_is_repeatable_and_lives_in_domain():
    config = Config(points=160)
    xy, labels = sample_domain(config)
    again, again_labels = sample_domain(config)
    np.testing.assert_array_equal(xy, again)
    np.testing.assert_array_equal(labels, again_labels)
    np.testing.assert_array_equal(domain_labels(xy), labels)
    assert len(np.unique(xy, axis=0)) == config.points
    assert np.all(labels >= 0)
    # Preserve the central aperture instead of filling the word silhouette.
    assert domain_labels(np.array([[(142 - 126) / 100, (72 - 60) / 100]]))[0] == -1


def test_shears_have_an_exact_inverse_and_normals_are_correct():
    xy = np.random.default_rng(4).uniform(-0.6, 0.6, size=(20, 2))
    amount = 1.3
    xyz, normal = embed(xy, amount)
    u, v, _ = xyz.T
    recovered_y = v - 0.16 * amount * np.sin(2 * u)
    recovered_x = u - 0.10 * amount * np.sin(2 * recovered_y)
    np.testing.assert_allclose(
        np.column_stack((recovered_x, recovered_y)), xy, atol=1e-14
    )
    np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1, atol=1e-14)
    for i in range(2):
        offset = np.eye(2)[i] * 1e-6
        tangent = (embed(xy + offset, amount)[0] - embed(xy - offset, amount)[0]) / 2e-6
        np.testing.assert_allclose(np.sum(tangent * normal, axis=1), 0, atol=1e-9)
    flat, _ = embed(xy, 0)
    np.testing.assert_array_equal(flat[:, :2], xy)
    np.testing.assert_array_equal(flat[:, 2], 0)


def test_render_and_points_are_repeatable_and_sparse(tmp_path):
    config = Config(points=160, size_px=600, diameter_px=1.8)
    xy, labels = sample_domain(config)
    xyz, colors = cloud(config, xy, labels)
    np.testing.assert_array_equal(xyz, cloud(config, xy, labels)[0])
    assert np.all(np.isfinite(xyz))
    assert np.all((colors >= 0) & (colors <= 1))
    save_mark(tmp_path, "first", config, xy, labels)
    save_mark(tmp_path, "second", config, xy, labels)
    assert (tmp_path / "first.png").read_bytes() == (
        tmp_path / "second.png"
    ).read_bytes()
    for suffix in ("pdf", "svg", "png", "-points.csv"):
        path = tmp_path / ("first" + ("" if suffix.startswith("-") else ".") + suffix)
        assert path.stat().st_size > 0
    image = np.array(Image.open(tmp_path / "first.png").convert("RGB"))
    assert image.shape == (600, 600, 3)
    assert np.mean(np.all(image == 255, axis=-1)) > 0.99
    assert np.mean(np.any(image < 240, axis=-1)) > 0.0001
