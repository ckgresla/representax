"""Check the canonical Org manuscript without starting training or loading data."""

import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


HERE = Path(__file__).resolve().parent


@unittest.skipUnless(shutil.which("emacs"), "Org export requires Emacs")
class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name) / "paper"
        shutil.copytree(
            HERE,
            cls.root,
            ignore=shutil.ignore_patterns("build", "__pycache__", "evidence.json", "assets"),
        )
        for mode in ("preprint", "review"):
            cls.export(mode)

    @classmethod
    def export(cls, mode):
        subprocess.run(
            ["emacs", "--batch", "-Q", "--script", "export.el", mode],
            cwd=cls.root, check=True, capture_output=True, text=True,
        )

    def test_approved_abstract_is_unchanged(self):
        abstract = (self.root / "build/preprint/abstract.txt").read_bytes()
        self.assertEqual(
            hashlib.sha256(abstract).hexdigest(),
            "ccd7cc71580354859b92c96928161ce85317fcbb5c56dea39c9531ccae74d27a",
        )
        tex = (self.root / "build/preprint/paper.tex").read_text()
        block = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)[1]
        self.assertEqual(" ".join(block.split()), abstract.decode().strip())

    def test_review_has_no_author_identity_and_notes_are_excluded(self):
        tex = (self.root / "build/review/paper.tex").read_text()
        self.assertIn(r"\author{Anonymous authors}", tex)
        self.assertIn(r"\newcommand{\representaxreview}{}", tex)
        self.assertNotIn("Chris Kerwell Gresla", tex)
        for mode in ("preprint", "review"):
            text = (self.root / f"build/{mode}/paper.tex").read_text()
            self.assertNotIn("Author Notes", text)
            self.assertNotIn("/raid/", text)
            self.assertNotIn(str(self.root), text)
            self.assertNotIn(r"\begin{figure}tbp", text)

    def test_citations_and_figures_resolve_locally(self):
        output = self.root / "build/preprint"
        tex = (output / "paper.tex").read_text()
        figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex)
        self.assertEqual(len(figures), 5)
        for figure in figures:
            self.assertTrue((output / figure).is_file(), figure)
        bib = (output / "references.bib").read_text()
        for group in re.findall(r"\\citep\{([^}]+)\}", tex):
            for citation in group.split(","):
                self.assertIn("{" + citation.strip() + ",", bib)
        self.assertIn(r"\bibliography{references}", tex)

    def test_export_is_stable_and_removes_stale_figures(self):
        output = self.root / "build/preprint"
        before = (output / "paper.tex").read_bytes()
        stale = output / "figures/not-in-manuscript.pdf"
        stale.touch()
        self.export("preprint")
        self.assertEqual(
            hashlib.sha256(before).hexdigest(),
            hashlib.sha256((output / "paper.tex").read_bytes()).hexdigest(),
        )
        self.assertFalse(stale.exists())

    def test_main_comparison_is_a_table_and_seed_plot_is_in_appendix(self):
        tex = (self.root / "build/preprint/paper.tex").read_text()
        table = tex.index(r"\label{tab:framework-throughput}")
        appendix = tex.index(r"\appendix")
        figure = tex.index(r"\includegraphics[width=\linewidth]{figures/framework-throughput.pdf}")
        compiled = tex.index(r"\label{sec:compiled-reference}")
        self.assertLess(table, appendix)
        self.assertGreater(figure, appendix)
        self.assertGreater(compiled, appendix)


if __name__ == "__main__":
    unittest.main()
