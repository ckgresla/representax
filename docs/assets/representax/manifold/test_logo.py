import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from condensed import CELL_GAP, CELL_WIDTH, composition, mono_layout
from logo import ARM_EXTRA, FILL_OPACITY, POINT_OPACITY, PREFIX_WIDTH


def test_selected_assets_and_provenance():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "manifest.json").read_text())
    assert ARM_EXTRA == manifest["arm_extra_units"] == 7
    assert FILL_OPACITY == manifest["fill_opacity"] == 0.3
    assert POINT_OPACITY == manifest["point_opacity"] == 0.9
    assert PREFIX_WIDTH == manifest["glyph_width_scale"] == 0.8
    assert manifest["prefix_points"] == 1129
    assert manifest["tax_points"] == 279
    for name, expected in manifest["sources"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    for name, artifact in manifest["artifacts"].items():
        assert (
            hashlib.sha256((root / name).read_bytes()).hexdigest() == artifact["sha256"]
        )
        if name.endswith(".svg"):
            assert ElementTree.parse(root / name).getroot().tag.endswith("svg")


def test_condensed_prefix_uses_equal_advances_without_changing_tax():
    letters = mono_layout(PREFIX_WIDTH)
    np.testing.assert_allclose(
        np.diff([offset for _, offset, _ in letters]),
        CELL_WIDTH * PREFIX_WIDTH + CELL_GAP,
    )
    original = composition(None, ARM_EXTRA)
    condensed = composition(PREFIX_WIDTH, ARM_EXTRA)
    np.testing.assert_array_equal(
        original[2].get_matrix()[:2, :2], condensed[2].get_matrix()[:2, :2]
    )
    for (before, a), (after, b) in zip(original[1], condensed[1], strict=True):
        assert a == b
        for x, y in zip(before, after, strict=True):
            np.testing.assert_array_equal(x, y)
    assert condensed[3][1] - condensed[3][0] < original[3][1] - original[3][0]
