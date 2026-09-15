import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree

from logo import ARM_EXTRA, FILL_OPACITY, POINT_OPACITY


def test_selected_assets_and_provenance():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "manifest.json").read_text())
    assert ARM_EXTRA == manifest["arm_extra_units"] == 7
    assert FILL_OPACITY == manifest["fill_opacity"] == 0.3
    assert POINT_OPACITY == manifest["point_opacity"] == 0.9
    assert manifest["prefix_points"] == 1167
    assert manifest["tax_points"] == 279
    for name, expected in manifest["sources"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    for name, artifact in manifest["artifacts"].items():
        assert (
            hashlib.sha256((root / name).read_bytes()).hexdigest() == artifact["sha256"]
        )
        if name.endswith(".svg"):
            assert ElementTree.parse(root / name).getroot().tag.endswith("svg")
