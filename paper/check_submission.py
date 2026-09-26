"""Mechanical ICLR checks on a freshly compiled review PDF, not author approval."""

import hashlib
from pathlib import Path
import re
import subprocess


HERE = Path(__file__).resolve().parent
STYLE_HASHES = {
    "iclr2027_conference.sty": "797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6",
    "iclr2027_conference.bst": "2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5",
}
FIGURES = (
    "architecture", "scaling", "learning", "design",
    "loss-gpu", "loss-tpu", "omni-trajectories",
)


def check_figure_pages(aux):
    pages = dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{[^}]*\}\{(\d+)\}", aux))
    for name in FIGURES:
        figure, discussion = f"fig:{name}", f"discussion:{name}"
        if figure not in pages or discussion not in pages:
            raise ValueError(f"Missing figure/discussion page labels: {name}")
        if pages[figure] != pages[discussion]:
            raise ValueError(
                f"Figure {name} is on page {pages[figure]}, "
                f"but its discussion is on page {pages[discussion]}"
            )


def main_pages(aux):
    match = re.search(
        r"\\newlabel\{sec:statements-start\}\{\{[^}]*\}\{(\d+)\}", aux
    )
    if match is None:
        raise ValueError("Missing compiled statements boundary; rebuild the review PDF")
    # The manuscript flushes all main-text floats before this new-page section.
    pages = int(match[1]) - 1
    if not 1 <= pages <= 9:
        raise ValueError(f"Main text occupies {pages} pages; initial ICLR limit is 9")
    return pages


def check_identity(text):
    normalized = " ".join(text.casefold().split())
    for identity in ("chris kerwell gresla", "ckgresla", "/home/ckg", "/raid/"):
        if identity in normalized:
            raise ValueError(f"Known identifying or private string in review PDF: {identity}")


def check(root):
    for name, expected in STYLE_HASHES.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Official style changed: {name}")
    tex = (root / "paper.tex").read_text()
    if r"\author{Anonymous authors}" not in tex or r"\representaxreview" not in tex:
        raise ValueError("Not an anonymous review export")
    aux = (root / "paper.aux").read_text()
    pages = main_pages(aux)
    check_figure_pages(aux)
    pdf = root / "paper.pdf"
    if pdf.stat().st_size > 50_000_000:
        raise ValueError("Review PDF exceeds the submission form's 50 MB limit")
    commands = (
        ["pdftotext", "-layout", str(pdf), "-"],
        ["pdfinfo", str(pdf)],
        ["pdfinfo", "-meta", str(pdf)],
        ["pdfinfo", "-url", str(pdf)],
    )
    outputs = [
        subprocess.run(command, check=True, capture_output=True, text=True).stdout
        for command in commands
    ]
    for output in outputs:
        check_identity(output)
    # Poppler separates the initial capitals from the rest of small-cap words.
    rendered = "".join(outputs[0].casefold().split())
    for heading in ("reproducibility statement", "ethics statement", "ai use statement"):
        if heading.replace(" ", "") not in rendered:
            raise ValueError(f"Missing rendered section: {heading}")
    if not re.search(r"^Author:\s+Anonymous authors\s*$", outputs[1], re.M):
        raise ValueError("PDF author metadata is not anonymous")
    print(f"Review checks passed: {pages}/9 main-text pages, figure adjacency, official style, known-identity scan.")
    print("Human review of claims, layout, disclosure, and anonymous supplements remains required.")


if __name__ == "__main__":
    check(HERE / "build/review")
