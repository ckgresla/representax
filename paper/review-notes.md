# Author Review Handoff

The complete manuscript is `paper.org`. The approved abstract is unchanged.
Author: Chris Kerwell Gresla, Independent Researcher. The main structure is
toolkit design, reference throughput, useful native adaptation, strong scaling,
and limitations, with detailed recipes and reproduction records in appendices.
The initial writing pass launched no training. The subsequent padding correction
reran five GPU TRL references; no core library/training code was changed.

## Evidence Qualifications

The writing audit reads historical artifacts, not today's launcher defaults:

**2026-09-16 full paired audit supersedes the earlier interpretation below.**
See `fairness-audit.md` for all 13 recipes, 265 runs and 693 verified source
hashes, diagnostic evidence and the repair/rerun matrix. Matched-training
ratios are now withheld for late interaction, outcome/process reward and
audio/video pending corrections. This does not delete historical measurements.
The process-reward padding correction is still valid; its newly discovered
head-initialization issue is separate. The complete GPU process-reward replacement
has now passed and is promoted (55.1760 versus 16.7875 examples/s, 3.2867x).
The TPU process replacement is also complete (85.2563 versus 9.93935 examples/s,
8.5777x), with all sixteen reference replicas verified for each seed.
All 50 corrected GPU cells are now complete and promoted: late interaction
3.9776x, outcome reward 1.4388x, process reward 3.2867x, audio-text 2.1262x,
and video-text 3.4395x. TPU late interaction has completed ten jobs but is held
for a real-batch/loss-reporting check; the other TPU panels remain in progress.
The native learning studies and A100
strong-scaling measurements are not invalidated by these paired-run findings.

1. **Process reward (2026-09-16 correction):** the shared preparation selected
   1,920 examples containing 102--256 real tokens. All five native GPU runs
   actually used 256-token buckets; the original TRL runs padded to 2,048.
   Five corrected TRL runs now use verified 2 x 256 training tensors with the
   same data, checkpoint, optimizer, batch 64 and microbatch two. Their median
   rate is 16.9460 examples/s versus the retained native 56.2287: **3.3181x**.
   Twenty warm updates exclude first-use steps 1 and 12 around resume. The
   original invalid 9.005x comparison is retained only in correction history;
   that padding-only panel is also now superseded by ten freshly paired runs
   with shared scalar-head initialization. The
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
   evaluations completed on GPUs 0 and 1 under
   `/raid/representax-paper/11-dense-retrieval-convergence/initial-transfer/`.
   They reuse the full corpora and original scoring settings; one shared
   pretrained baseline serves the three trained seeds. Initial nDCG@10 is
   0.01701635 and 0.00322931, respectively; mean final scores remain 0.54043131
   and 0.29216390. Source snapshot/patch, model weights, data manifest and original
   run configuration hashes were verified before importing both reports.
   Final scores are not overwritten. Flickr30k already has initial and final scores. No convergence-equivalence,
   universal speedup, SOTA, production-readiness or FSDP-capacity claim is made.
9. **TPU audio execution audit, 2026-09-16:** all five seeds use global batch
   48, three examples per chip. Native rematerialized GradCache uses chunk two
   plus full layer checkpointing; the reference uses direct MNR plus XLA
   layer checkpointing. Historical source `5b0d216` pads native encoder inputs
   from three to four slots per chip, dropping padding before the loss.
   Commit `576d281` switched the reference to direct MNR after replay
   compatibility/memory work; the native chunk-two setting was inherited
   from the earlier GPU preflight. The measured medians are 11.2468 versus
   13.4512 examples/s. Native warm input-queue waits are below 0.1 ms on
   average per seed, but placement/enqueue averages 2.75--2.93 s; this can
   include asynchronous backpressure and is not isolated transfer time.
   Compilation is excluded. Padding/replay are plausible costs, not an
   established cause of the entire gap. The author requested a TPU rerun:
   check direct MNR and chunk-three capacity, then rerun the five-seed pair
   on one matching slice. The historical direct path ignored local-negative
   scope; a user-approved isolated patch reuses the cached path's per-device
   loss without encoder replay. Ten CPU-mesh tests check loss, gradients and
   optimizer updates against explicitly grouped MNR, including three pairs
   per chip and two-/four-device Auto/Explicit meshes. The unpatched direct
   path fails these checks. This does not establish real-model TPU capacity
   or speed; those remain canary gates. Preserve historical records until reviewed
  replacement evidence exists; do not present this as a tuned-backend verdict.

## Artifact Boundaries

- `evidence.json` preserves both the first five-reference padding correction
  and the subsequent ten-run GPU process-reward replacement under `corrections`.
  Complete additional panels are promoted individually after validation;
  pending historical ratios remain withheld. `methods.json` retains the earlier
  265-run methods snapshot with 693 source hashes; replacement run records carry
  their own source, configuration/artifact locations and metric hashes.
- `report.py` reconstructs nine numeric tables, five vector/PNG figures and
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

Review the body, reference-integration descriptions and disclosure. The main
framework comparison is now a four-column table with absolute rates and
relative ratios; bold denotes the higher of the two reported medians per
workload/hardware group, not significance. The compiled dense reference and
per-seed plot are in the appendix.
Qualified process-reward and small-batch V-JEPA rows remain visible, without a
pooled framework speedup.
Confirm current venue formatting and the final public repository state.
No submission, push or commit is performed by the paper build.
