# Author Review Handoff

The complete manuscript is `paper.org`. The approved abstract is unchanged.
Author: Chris Kerwell Gresla, Independent Researcher. The main structure is
toolkit design, reference throughput, useful native adaptation, strong scaling,
and limitations, with detailed recipes and reproduction records in appendices.
The initial writing pass launched no training. The subsequent padding correction
reran five GPU TRL references; no core library/training code was changed.

## Evidence Qualifications

The writing audit reads historical artifacts, not today's launcher defaults:

1. **Process reward (2026-09-16 correction):** the shared preparation selected
   1,920 examples containing 102--256 real tokens. All five native GPU runs
   actually used 256-token buckets; the original TRL runs padded to 2,048.
   Five corrected TRL runs now use verified 2 x 256 training tensors with the
   same data, checkpoint, optimizer, batch 64 and microbatch two. Their median
   rate is 16.9460 examples/s versus the retained native 56.2287: **3.3181x**.
   Twenty warm updates exclude first-use steps 1 and 12 around resume. The
   original invalid 9.005x comparison is retained only in correction history;
   active evidence, figures, tables and prose use the corrected runs. The
   native summary's configured maximum was not its actual execution length;
   recorded token capacities establish 256. Both TPU frameworks still use
   2,048-token tensors. `[P]` qualifies short real inputs, not an unresolved
   shape mismatch; neither backend demonstrates long-reasoning training.
2. **GPU V-JEPA:** global batch one, 24 videos; TPU global batch 128, 2,816 videos.
   Matching batches between frameworks on each hardware allocation is the
   comparison requirement; identical batches across GPU and TPU are not.
   No rerun is required merely for this difference. Retain the actual batch
   sizes and do not interpret absolute GPU/TPU rates as fixed-work scaling.
3. **Historical paired CLIP:** four presentations of each image's first caption.
   The later native CLIP adaptation uses five distinct captions per image.
4. **Seeded timing repetitions:** some pipelines keep a deterministic data order.
   Five seeds do not imply five independent shuffled scientific trajectories.
5. **Late interaction:** separate held-out score similarity from equivalence of
   learning dynamics. Historical GPU NanoMSMARCO nDCG@10 is 0.71234 initially
   and 0.71286 mean final for Representax, versus 0.70889 and 0.70785 for PyLate
   (five seeds). These endpoints are similar on a 50-query panel, but the
   original scorers use different precisions and the logged loss discrepancy
   is not established to be merely a constant reduction factor. Repeated
   query-positive pairs also create false negatives under diagonal labels.
   Do not describe these trajectories as proven equivalent. The separate
   revised native hard-negative learning run regresses from 0.7104 to 0.6925;
   that is a recipe result, not a paired framework-quality comparison.
6. **TPU input/cache effects:** image and predictive-video first seeds are noisy;
   several other reference windows also have high step-time variance. The
   paper calls these bounded warm timing measurements, not demonstrated
   long-run stationarity. All per-run coefficients of variation are preserved.
7. **Startup audit, 2026-09-16:** verified original metric and summary hashes
   for all 265 frozen runs. Recovered first and second optimizer-step intervals
   for all 135 references (65 GPU, 65 TPU, five Inductor controls), plus setup
   and first-use records for all 130 native runs. The table had searched only
   for a native metric name. Corrected tables distinguish the native first-use
   sum/event count from reference step 1/2 intervals, which include input wait.
   GPU late interaction has seven native first-use events, not one long initial
   compilation. The per-seed records and source hashes are in `analysis.json`.
   No reruns are needed to fill these entries. Pure compilation and controlled
   cold-cache end-to-end startup are not recoverable from these intervals.
8. **Adaptation recipes:** multimodal strategies differ in learning rates and
   sampled sources. They demonstrate useful recipes, not an isolated causal
   ablation. Dense TREC DL 2019 and Natural Questions initial-checkpoint
   evaluations are now running on GPUs 0 and 1 under
   `/raid/representax-paper/11-dense-retrieval-convergence/initial-transfer/`.
   They reuse the full corpora and original scoring settings; one shared
   pretrained baseline serves the three trained seeds. Final scores are not
   overwritten. Flickr30k already has initial and final scores. No convergence-equivalence,
   universal speedup, SOTA, production-readiness or FSDP-capacity claim is made.

## Artifact Boundaries

- `evidence.json` replaces only five GPU process-reward reference records and
  their aggregate. Its `corrections` entry preserves the superseded runs,
  aggregate and new launch provenance. All other measured runs are unchanged.
  `methods.json` records the resulting 265 paired-run settings and environments
  and 693 source hashes, including the corrected reference settings.
- `report.py` reconstructs eight numeric tables, five vector/PNG figures and
  `analysis.json` from the snapshot; no GPU, network or model weights required.
- Natural Questions corpus/query counts were checked locally with `wc -l`:
  2,681,468 documents and 3,452 queries. This is analysis, not a new experiment.
- Four additional manuscript tables describe capabilities, paired workload
  shapes, multimodal data and selected checkpoint revisions.
- The arXiv tarball is the **typesetting source**, not the complete research
  artifact package. Publish the reviewed paper directory and experiment code
  separately before claiming all new artifacts are available remotely.
- Review-mode PDF removes the author block and displayed repository URL.
  Do not assume that a source archive or repository is thereby anonymized.

## Before Upload

Review the body, reference-integration descriptions and disclosure. Decide
whether the qualified process-reward and small-batch V-JEPA rows should remain
in the main throughput figure or move entirely to the appendix. They are
currently included transparently, without a pooled framework speedup.
Confirm current venue formatting and the final public repository state.
No submission, push or commit is performed by the paper build.
