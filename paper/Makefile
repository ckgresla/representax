.DEFAULT_GOAL := pdf
EMACS ?= emacs
LATEXMK ?= latexmk

.PHONY: pdf review tex arxiv check

tex:
	$(EMACS) --batch -Q --script export.el preprint

pdf: tex
	cd build/preprint && $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -file-line-error paper.tex

review:
	$(EMACS) --batch -Q --script export.el review
	cd build/review && $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -file-line-error paper.tex

# Explicit source-only allowlist: no evidence JSON, Org, logs, or author notes.
arxiv: pdf
	tar -czf build/representax-arxiv.tar.gz -C build/preprint paper.tex paper.bbl references.bib preamble.tex iclr2027_conference.sty iclr2027_conference.bst figures

check:
	python3 -m unittest discover -s . -p 'test_*.py'
