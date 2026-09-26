"""Focused regression checks for the manuscript submission guard."""

import unittest

from check_submission import FIGURES, check_figure_pages, check_identity, main_pages


class SubmissionTests(unittest.TestCase):
    def test_figures_share_their_discussion_pages(self):
        aux = "\n".join(
            rf"\newlabel{{{kind}:{name}}}{{{{1}}{{{i}}}{{Caption}}{{}}{{}}}}"
            for i, name in enumerate(FIGURES, start=3)
            for kind in ("fig", "discussion")
        )
        check_figure_pages(aux)
        with self.assertRaisesRegex(ValueError, "discussion is on page 4"):
            check_figure_pages(aux.replace(
                r"\newlabel{discussion:architecture}{{1}{3}",
                r"\newlabel{discussion:architecture}{{1}{4}",
            ))

    def test_missing_figure_discussion_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Missing figure/discussion"):
            check_figure_pages("")

    def test_limit_excludes_statements_references_and_appendix(self):
        aux = (
            r"\newlabel{sec:statements-start}{{7}{10}{Reproducibility Statement}{section*.1}{}}"
            "\n" + r"\newlabel{appendix}{{A}{22}{Appendix}{appendix.A}{}}"
        )
        self.assertEqual(main_pages(aux), 9)

    def test_ten_main_pages_fail(self):
        with self.assertRaisesRegex(ValueError, "occupies 10 pages"):
            main_pages(r"\newlabel{sec:statements-start}{{7}{11}{Statement}{}{}}")

    def test_missing_boundary_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Missing compiled"):
            main_pages("")

    def test_known_identity_in_text_metadata_or_link_fails(self):
        for value in (
            "Chris\nKerwell Gresla", "Author: Chris Kerwell Gresla",
            "https://github.com/ckgresla/representax", "/raid/representax-paper",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                check_identity(value)

    def test_anonymous_author_and_external_citations_are_allowed(self):
        check_identity("Author: Anonymous authors https://github.com/jax-ml/jax")


if __name__ == "__main__":
    unittest.main()
