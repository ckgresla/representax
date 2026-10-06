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
            ignore=shutil.ignore_patterns("build", "__pycache__", "evidence.json"),
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

    def test_approved_introduction_and_conclusion_are_unchanged(self):
        text = (self.root / "paper.org").read_text()
        # Introduction: checkpoint 187488f; conclusion: approved 2026-09-24 addition,
        # with the autotuning sentence moved before the results summary on 2026-09-25.
        expected = {
            "Introduction": "cb805d51be9e69f1ac499e281180606351eb24541a09b6332eb10cfc07c6607e",
            "Conclusion": "6daf21115cf6e2a0fbf2c6b15ede455fc66224682483a9741bc40f494763f12e",
        }
        for name, digest in expected.items():
            section = re.search(r"(?m)^\* " + name + r"\n.*?(?=^\* )", text, re.S)[0]
            self.assertEqual(hashlib.sha256(section.encode()).hexdigest(), digest, name)

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

    def test_approved_related_work_and_framework_prose_are_unchanged(self):
        text = (self.root / "paper.org").read_text()
        # Approved checkpoint 647aae5; figure placement is not part of the prose.
        # Section 3.2 was condensed to one paragraph in the 2026-09-25 length pass.
        related = re.search(r"(?m)^\* Related Work\n.*?(?=^\* )", text, re.S)[0]
        self.assertEqual(hashlib.sha256(related.encode()).hexdigest(),
                         "10422d897fb2e92774f9f8e824974263789ef4a5ac162a2091a35ad693010617")
        framework = re.search(r"(?m)^\* Representax\n.*?(?=^\* )", text, re.S)[0]
        framework = re.sub(
            r"(?m)^#\+(?:LATEX|NAME|CAPTION|ATTR_LATEX):.*\n|^\[\[file:figures/.*\n",
            "", framework,
        )
        normalized = re.sub(r"\s+", " ", framework)
        self.assertEqual(hashlib.sha256(normalized.encode()).hexdigest(),
                         "543efdc76b818af8bc5610f843432c780733039b7aede17d5385798a67fb3566")

    def test_citations_and_figures_resolve_locally(self):
        output = self.root / "build/preprint"
        tex = (output / "paper.tex").read_text()
        figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex)
        self.assertEqual(len(figures), 7)
        self.assertEqual(len(set(figures)), len(figures))
        for figure in figures:
            self.assertTrue((output / figure).is_file(), figure)
        bib = (output / "references.bib").read_text()
        for group in re.findall(r"\\cite[pt]\{([^}]+)\}", tex):
            for citation in group.split(","):
                self.assertIn("{" + citation.strip() + ",", bib)
        self.assertIn(r"\bibliography{references}", tex)

    def test_selected_logo_is_packaged_only_in_preprint(self):
        expected = (self.root / "assets/representax-wordmark.pdf").read_bytes()
        output = self.root / "build/preprint"
        self.assertEqual((output / "figures/representax-wordmark.pdf").read_bytes(), expected)
        self.assertIn("figures/representax-wordmark.pdf", (output / "preamble.tex").read_text())
        self.assertFalse((self.root / "build/review/figures/representax-wordmark.pdf").exists())
        self.assertFalse((output / "figures/representax-mark.pdf").exists())

    def test_review_export_removes_stale_logo(self):
        logo = self.root / "build/review/figures/representax-wordmark.pdf"
        shutil.copyfile(self.root / "assets/representax-wordmark.pdf", logo)
        self.export("review")
        self.assertFalse(logo.exists())

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

    def test_main_comparison_is_a_table_and_controls_are_in_appendix(self):
        tex = (self.root / "build/preprint/paper.tex").read_text()
        table = tex.index(r"\label{tab:framework-throughput}")
        appendix = tex.index(r"\appendix")
        compiled = tex.index(r"\label{sec:compiled-reference}")
        self.assertLess(table, appendix)
        self.assertGreater(compiled, appendix)
        self.assertGreater(tex.index(r"\label{sec:design}"), appendix)
        self.assertGreater(tex.index(r"\label{tab:design-diagnostics}"), appendix)
        for name in ("loss-gpu", "loss-tpu"):
            self.assertGreater(tex.index(r"\label{fig:" + name + "}"), appendix)

    def test_comparisons_use_workload_bullets_and_five_seeds(self):
        text = (self.root / "paper.org").read_text()
        section = re.search(r"(?m)^\* Framework and Accelerator Comparisons\n.*?(?=^\* )",
                            text, re.S)[0]
        self.assertEqual(len(re.findall(r"(?m)^- \*", section)), 11)
        self.assertIn("the same workload across five seeds for each configuration", section)
        self.assertIn("The workloads cover the following training settings.", section)
        self.assertIn("written specifically for this recipe", section)

    def test_main_tables_follow_their_complete_study_descriptions(self):
        tex = (self.root / "build/preprint/paper.tex").read_text()
        for name, preceding in (("framework-throughput", "Predictive video learning"),
                                ("learning", "Image--Text Alignment")):
            label = tex.index(r"\label{tab:" + name + "}")
            start = tex.rfind(r"\begin{table}", 0, label)
            self.assertTrue(tex[start:].startswith(r"\begin{table}[H]"))
            self.assertLess(tex.index(preceding), start)
        comparison = tex[tex.index(r"\label{sec:comparisons}"):
                         tex.index(r"\label{tab:framework-throughput}")]
        self.assertIn(r"\end{itemize}", comparison)

    def test_learning_has_distinct_studies_and_appendix_negative_result(self):
        text = (self.root / "paper.org").read_text()
        learning = re.search(r"(?m)^\* End-to-End Representation Learning\n.*?(?=^\* )",
                             text, re.S)[0]
        intro = learning.split("\n** ", 1)[0]
        self.assertIn("fifteen runs across three workload families", intro)
        self.assertIn("retrieval regression", intro)
        self.assertIn("[[#sec:late-followup]]", intro)
        self.assertNotIn("nDCG@10", intro)
        self.assertNotIn("50 queries", intro)
        self.assertEqual(re.findall(r"(?m)^\*\* (.+)$", learning), [
            "Dense Retrieval and Transfer", "Image--Text Alignment",
            "Multimodal Adaptation and Retention",
        ])
        self.assertNotIn("0.7104", learning)
        appendix = text[text.index("#+LATEX: \\appendix"):]
        self.assertIn(":CUSTOM_ID: sec:late-followup", appendix)
        self.assertIn("0.7104 to 0.6925", appendix)
        self.assertIn("Those evaluations have not been performed", appendix)

    def test_limits_stay_with_experiments_and_design_is_in_appendix(self):
        tex = (self.root / "build/preprint/paper.tex").read_text()
        main = tex[:tex.index(r"\appendix")]
        comparisons = main[main.index(r"\label{sec:comparisons}"):main.index(r"\label{sec:learning}")]
        learning = main[main.index(r"\label{sec:learning}"):main.index(r"\label{sec:scaling}")]
        self.assertNotIn(r"\section{Design Analysis}", main)
        self.assertNotIn(r"\section{Capabilities, Limitations, and Reproducibility}", main)
        self.assertNotIn("long-running", comparisons)
        self.assertIn("Startup and checkpoint-related", comparisons)
        self.assertIn(r"\textbf{Limitations.}", comparisons)
        self.assertIn("evaluate a shared recipe", comparisons)
        self.assertIn("50 queries and 5,043 passages", learning)
        self.assertIn("upstream pretraining contamination", learning)
        self.assertIn("not state-of-the-art quality", learning)
        self.assertIn("requires the recorded accelerator configurations, environments, and upstream", main)

    def test_excluded_statements_follow_flushed_main_text(self):
        tex = (self.root / "build/review/paper.tex").read_text()
        boundary = tex.index(r"\section*{Reproducibility Statement}")
        self.assertTrue(tex[:boundary].rstrip().endswith(r"\clearpage"))
        self.assertIn(r"\label{sec:statements-start}", tex[boundary:])
        self.assertLess(tex.index(r"\section*{AI Use Statement}"), tex.index(r"\bibliography"))


if __name__ == "__main__":
    unittest.main()
