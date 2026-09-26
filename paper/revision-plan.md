# Interior Revision, 2026-09-25

## Section 4 Approval

- [x] Land the approved Section 4 with workload bullets and the same workload across five seeds.
- [x] Preserve the original PDF table and improve its spacing without changing values.
- [x] Add paired GPU/TPU loss curves to the appendix and the compiler comparison citation.
- [x] Move Design Analysis to Appendix F and remove the standalone limitations section.
- [x] Place experimental limitations with their results and retain capability/reproduction details.
- [x] Apply the PDF comments: table after the full workload list, shorter timing scope, and labeled limitations.
- [x] Expand Section 5 around distinct scientific studies and move the late-interaction details to Appendix B.2.
- [x] Rebuild both PDFs and the source archive; all 59 paper tests pass.
- [ ] Restore the nine-page main-text budget after content review; the author explicitly deferred cuts.

The main text now has seven sections, ending with the unchanged conclusion.
The latest PDFs have eleven main-text pages and 34 total pages; the prior
nine-page validation below is historical. No shortening pass is requested yet.

## Earlier Interior Pass

Checkpoint before the remaining-sections pass:
`647aae52bd22938db9161ca7c0fc30b2714e1c5c`.
Sections 2 and 3 are now approved alongside the abstract, introduction, and
conclusion. Preserve their prose and all frozen evidence. Figure placement
may change to satisfy the author's same-page requirement.

- [x] Checkpoint the current paper and record Sections 2 and 3 as approved.
- [x] Revise Sections 4--8 around questions, explicit choices, and evidence.
- [x] Integrate the selected logo in the named preprint only and fix figure adjacency.
- [x] Verify report reconstruction, manuscript tests, and the source archive.
- [x] Inspect both PDFs and satisfy the unmodified nine-page submission guard.
- [x] Hand off Sections 4--8 for final author review; do not submit or upload.

Outcome: 51 tests pass in the CPU experiment environment; both PDFs have nine
main-text pages and 28 total pages. The review guard checks all six figure-page
pairs as well as the original page/style/identity requirements. The isolated
renderer and independently extracted source archive rebuild. Frozen evidence
and numeric tables are unchanged. This completes drafting and verification,
not final author approval or submission.

## Previous Interior Pass, 2026-09-23

Author-requested checkpoint: `187488f836ada1720de4d9af82091c5c01c3aa32`.
Preserve the approved abstract, introduction, and conclusion. Keep all frozen
measurements, unfavorable results, and comparison qualifications intact.

- [x] Commit the approved draft and recover the nine-section outline.
- [x] Recover the author's palette and inspect the website's visual conventions.
- [x] Revise Related Work and the Representax walkthrough against source interfaces.
- [x] Revise comparison, adaptation, scaling, design analysis, and limitations.
- [x] Redesign scientific figures from frozen evidence and inspect rendered pages.
- [x] Validate citations, numerical tables, executable example, and PDF exports.
- [x] Produce an anonymous review draft within nine pages and a named author-review PDF.

Outcome: 47 paper tests pass. Both PDFs have eight main-text pages and 26 total
pages. The review guard passes, and the named source bundle builds independently
after extraction. Approved prose and the original frozen evidence are unchanged.
This completes the requested drafting pass, not author approval or submission.

## Editorial Contract

The narrative proceeds from flexible research interfaces to fair execution
comparisons, useful learning, scaling, and the implications for further research.
Experiments answer different questions; throughput is not evidence of learning
quality, and implementation support is not evidence of execution at every scale.
Design diagnostics remain separate from the replicated benchmark panel.
No new training, public release, push, or conference submission is included.

## Figure Contract

Use white backgrounds, Inter, restrained rules, aligned labels, direct
annotations, and vector exports. Use the author's light palette: Sky `#3f78b5`,
Rose `#c85f65`, Mint `#4f8a67`, Periwinkle `#8278b8`, Apricot `#cc7a3a`,
Cream `#b78a28`, and Slate `#68717c`. Encode comparisons with position and
marker shape as well as color. Preserve individual seeds and distinguish
standard deviations from confidence intervals. No invented trajectories,
selective axis clipping, or treating unavailable measurements as zero.

The scientific comparison remains a main-text table, as previously approved;
the per-seed throughput ratio chart remains in the appendix. ICLR's official
style, font sizes, margins, and initial nine-page limit remain unchanged.
