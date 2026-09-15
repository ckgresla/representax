import numpy as np
from a_arm import arm_faces
from tax import TAX_FACES


def test_only_original_left_arm_and_foot_change():
    original = {name: np.array(vertices) for name, _, _, vertices in TAX_FACES}
    for extra in (0, 7, 14):
        current = {name: vertices for name, _, _, vertices in arm_faces(extra)}
        assert current.keys() == original.keys()
        for name, vertices in original.items():
            if name not in ("A left", "A left cap"):
                np.testing.assert_array_equal(current[name], vertices)
        np.testing.assert_array_equal(current["A left"][:, 1], original["A left"][:, 1])
        np.testing.assert_array_equal(current["A left"][2:], original["A left"][2:])
        np.testing.assert_array_equal(
            current["A left"][:2], original["A left"][:2] - (extra, 0)
        )
        np.testing.assert_array_equal(
            current["A left cap"][:2], current["A left"][[0, 3]]
        )
        np.testing.assert_array_equal(
            current["A left cap"][[3, 2]], current["A left cap"][:2] + (14, 24)
        )


def test_arm_is_parallel_and_uniform_width():
    for extra in (0, 7, 14):
        arm = next(
            vertices for name, _, _, vertices in arm_faces(extra) if name == "A left"
        )
        np.testing.assert_array_equal(arm[1] - arm[0], (56, -96))
        np.testing.assert_array_equal(arm[2] - arm[3], (56, -96))
        np.testing.assert_array_equal(arm[3] - arm[0], (14 + extra, 0))
        np.testing.assert_array_equal(arm[2] - arm[1], (14 + extra, 0))
