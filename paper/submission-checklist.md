# Submission and Release Checklist

Requirements checked against official sources on 2026-09-17. This is an author
checklist, not evidence that either submission has been made.

## ICLR 2027

- Abstract: September 18, 2026, 23:59 AoE (September 19, 04:59 PDT).
- Full paper: September 25, 2026, 23:59 AoE (September 26, 04:59 PDT).
  Verify the OpenReview receipt before each deadline.
  Submitted 2026-09-25 17:58 PDT: submission 26733, forum
  `https://openreview.net/forum?id=BMX2iJpaaP`; OpenReview confirmed the
  revision by email. Source tag `iclr2027-submission`.
  [Official dates](https://iclr.cc/Conferences/2027/Dates).
- Initial main text: at most **9 pages**, using the official style. References,
  post-reference appendices, and reproducibility/ethics/AI statements are excluded.
  Ethics: at most one page. Review PDF and supplements must be anonymous.
  Author membership freezes at the abstract deadline. Confirm the reciprocal
  reviewer or applicable exemption in OpenReview.
  [Author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).
- Disclose AI assistance in both manuscript and submission form. The existing
  disclosure covers implementation, debugging, experiment design, analysis,
  literature discovery and drafting; the author must review its completeness.
  [AI policy](https://iclr.cc/Conferences/2027/AIPolicyForAuthors).

### Our Working Convention

`make -C paper review` builds the anonymous PDF and checks the nine-page main-text
boundary, official style hashes, required statement headings, 50 MB PDF limit,
anonymous author metadata, and known identity/private-path strings in rendered
text, PDF metadata and link targets. Install `poppler-utils` for these checks.
The statements start on a new page after all main-text floats have been flushed;
do not move substantive results into the excluded statements to fit the limit.
Keep the boundary immediately after the conclusion.

As of the evening of 2026-09-25 the main text is nine pages and the guard
passes. Design Analysis is in the appendix, the standalone
capabilities/limitations section is removed, and the conclusion is Section 7.
Prefer shortening or moving supporting
detail to the appendix over shrinking fonts, margins or line spacing. The guard
does not certify all formatting or anonymity: visually review each final PDF,
inspect supplementary files, and check identifying acknowledgements/self-citations.
It does not validate the scientific claims or submission-form metadata.

Before upload, confirm title, approved abstract, author profile, keywords, primary
area, reviewer exemption if applicable, AI-use categories, license, ethics and
visibility acknowledgements. The author makes those attestations. The PDF and
supplementary limits shown in the supplied form are 50 MB and 100 MB respectively.
Decision (2026-09-25): no code supplement accompanies the ICLR submission; the
named arXiv version links the repository.

### Final Layout Review

- [x] Author requirement (2026-09-25): place each figure and its caption on the
  same page as its first substantive discussion, not merely somewhere nearby.
  In particular, Figure 1 must accompany Section 3.1, Core Abstractions, rather
  than floating onto the preceding page. Later cross-references need not repeat
  the figure. Verify both named and anonymous PDFs after the content settles.
- [x] Restore the nine-page budget (2026-09-25): condensed Section 3.2, the
  Section 4 workload bullets, the Section 5 intro and scope caveats, the
  Section 5.3 recipes and limitations, Section 4.1's compiled-control
  paragraph, the Section 6 setup, and the Figure 1, Figure 2, Table 1 and
  Table 3 captions; Figure 2 at half linewidth. No font, margin or spacing
  changes. Pages 1, 7 and 9 were visually inspected after the final build.
- [x] Author-approved first-page logo in the named preprint only: a compact selected
  vector mark beside the author block, with one explanatory footnote, not a
  numbered research figure or a repeated footer. Canonical asset:
  `../docs/assets/representax/representax-mark.pdf`; the paper-local copy is
  `assets/representax-mark.pdf`. The named export and self-contained arXiv source
  archive include it. The anonymous submission omits the logo asset and its
  explanatory footnote, superseding the earlier approval for both variants.
  Known-identity scans still run on the anonymous variant;
  they do not replace human review of all identifying content.

## arXiv and Sharing

Use the named-author preprint, not the anonymous conference PDF. Upload the
`make -C paper arxiv` source bundle: LaTeX, bibliography, required style files
and referenced figures, without build debris. Inspect arXiv's compiled preview
before final submission. Confirm author metadata, category, license and any
account/endorsement requirements in the submission flow.
[arXiv submission overview](https://info.arxiv.org/help/submit/index.html),
[TeX requirements](https://info.arxiv.org/help/submit_tex.html).

ICLR permits arXiv posting during review; the conference artifacts must still be
anonymous. I found no publicity embargo in the checked author guidelines.
[ICLR policy](https://iclr.cc/Conferences/2027/AuthorGuidelines).

Recommendation, not a venue rule: circulate a reviewed draft to a few colleagues,
then share the arXiv link and repository publicly as a preprint and invite feedback.
Discuss the work rather than its ICLR submission, do not target likely reviewers,
and never imply acceptance. Recheck policy immediately before submission/sharing.

## Public Software Release

PyPI currently serves `0.0.1` (2026-08-15); its description still calls several
implemented tasks planned. Local package metadata is also `0.0.1`.
[Published package](https://pypi.org/project/representax/0.0.1/).
Select a new version only after release review; `0.1.0` is a candidate, not an
approved version bump. Follow [the release procedure](../docs/releasing.md).

- Review the release tree, public API/dependencies, README, compatibility claims,
  licenses, reproduction commands, changelog and compact evidence inventory.
- Inspect both wheel and sdist; keep caches, credentials, private paths,
  checkpoints and discarded design studies out. Publish paper evidence as a
  separate research artifact when it is not needed to install the library.
- Build, check metadata, and smoke-test the exact wheel in clean CPU/GPU
  environments. Inspect the sdist and rebuild its wheel too. Check CI and the
  protected publishing configuration before approving a release tag.
- Keep experiment-linked commits intact. Cleanup commits should improve the
  current tree, not rewrite provenance or silently discard intermediate results.
  An annotated/signed release tag and curated release notes provide a clean entry
  point without erasing history.
- Prepare the anonymous review snapshot separately, without `.git`, identities,
  private paths or identifying links. Preserve required third-party attribution;
  do not alter original public evidence just to anonymize a review artifact.
- Publish only with explicit approval, then verify the PyPI version, GitHub tag,
  distribution hashes and installed behavior. A normal branch push is not a
  package release; a pushed `v*` tag starts the publishing workflow.

No history rewrite, repository deletion, package upload or submission is authorized
or performed by these build/check commands.
