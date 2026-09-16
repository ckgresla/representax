"""Execute the manuscript's public-API example on CPU, without model downloads."""

import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest


HERE = Path(__file__).resolve().parent


class ExampleTests(unittest.TestCase):
    @unittest.skipUnless(
        all(importlib.util.find_spec(name) for name in ("jax", "equinox", "representax")),
        "API example requires the Representax environment",
    )
    def test_task_composition(self):
        manuscript = (HERE / "paper.org").read_text()
        block = re.search(
            r"^#\+NAME: example:task-composition\n.*?"
            r"^#\+begin_src python\n(.*?)^#\+end_src$",
            manuscript, re.M | re.S,
        )
        self.assertIsNotNone(block, "missing canonical API example")
        checks = """
assert jax.default_backend() == 'cpu'
assert loss.shape == () and bool(jnp.isfinite(loss))
leaves = jax.tree.leaves(gradients)
assert leaves and all(bool(jnp.all(jnp.isfinite(x))) for x in leaves)
assert any(bool(jnp.any(x != 0)) for x in leaves)
assert task.loss(model, batch).loss.shape == ()
"""
        result = subprocess.run(
            [sys.executable, "-"], input=block[1] + checks,
            cwd=HERE.parent, text=True, capture_output=True, timeout=120,
            env={**os.environ, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
