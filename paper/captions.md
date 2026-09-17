# Historical Caption Drafts

Superseded editorial working material, retained for history. Current captions
are in `paper.org` and generated `tables/*.org`; those alone are exported.
All affected paired panels and transfer baselines are now complete. Descriptions
below of pending corrections, missing baselines and old plot encodings are not
the current manuscript state.

## Framework Throughput Table

Median examples/s across five seeds, with separate GPU and TPU panels and
Representax/reference ratios. Bold marks the higher of the two reported medians
within each workload/hardware group, not statistical significance. GPU
references use eager execution; the dense TorchInductor control is reported
in the compiled-GPU appendix subsection, not in sparsely populated columns.
Hardware allocations and batches can differ across panels, not between the
frameworks within a paired workload. Historical rows [L], [O], [I] and [M]
retain absolute rates but withhold ratios/winner styling while awaiting pairing
corrections. Complete validated replacements restore those comparisons.
Qualifications [P], [C], [G] remain.

## Framework Throughput Plot (Appendix)

Thirteen measured recipe rows, five seeds per framework and accelerator panel.
Diamonds are the ratio of median per-seed examples/s; dots show individual
paired-seed ratios, not confidence intervals. Rates include input waiting within
complete optimizer-step intervals and exclude compilation/first use. GPU uses
one RTX 4090 per run; TPU uses the full 16-chip multi-host v5e slice. `[L]` marks
historical late interaction with different negative pools/batches and cached
gradient normalization. `[O]` marks outcome-reward master/state precision;
`[M]` marks audio/video adapter precision and initialization; `[I]` marks
process-reward initialization. Uncorrected cells in these categories have no
plotted ratio; validated replacements do.
`[P]` marks short process-reward inputs, with matched
padding of 256 on GPU and 2,048 on TPU; the GPU row uses corrected references.
`[C]` marks TPU rows with pronounced cold-filesystem order effects
and preprocessing outside the reference timed loop. All eligible seeds remain visible.
The dense Inductor control is reported separately in the appendix numeric table.

## Cross-Accelerator

Ratio of median Representax/reference examples/s on GPU versus TPU. The dashed
lines indicate equal framework rates. Numbers identify recipes in the adjacent
key. This is a relative-performance map, not an equal-resource accelerator
comparison or a Pareto frontier. Hardware allocations, negative pools and
some batch capacities differ across accelerator panels. Historical and
cold-cache qualifications from the throughput figure also apply here.

## Held-Out Learning

Initial and mean final nDCG@10 over three seeds. Open circles indicate initial
scores; filled circles indicate final scores with sample standard deviations.
CLIP directions are evaluated on Flickr30k following COCO adaptation; dense and
late interaction use NanoMSMARCO. Include the late-interaction regression rather
than implying every recipe improves. These are finite-budget adaptation results,
not measurements of asymptotic convergence. Transfer scores without an initial
evaluation are reported in the table, not drawn as improvements.

## Multimodal Adaptation

Initial to final nDCG@10 for Jina v5 Omni Nano adaptation, three seeds per
strategy and 2,000 updates per run. Each strategy uses its own initial scores.
Whiskers show sample standard deviations. Lines join endpoints; they are not
interpolated learning curves. Connector-only samples media-text sources
uniformly and excludes text-only training; the other strategies also train on
text-text batches. Media gains and text retention therefore reflect complete
adaptation recipes, not an isolated parameter-selection ablation. Dataset
panels must not be averaged into an unqualified overall quality score.

## Strong Scaling

Controlled 505M ModernBERT masked-language modeling on 1/2/4/8 A100 SXM4 GPUs,
with 131,072 input tokens per update and sequence length 512. The local
microbatch is 16, with accumulation counts 16/8/4/2. Left: mean tokens/s and
sample SD across three seeds, versus ideal linear scaling from the measured
one-GPU baseline. Right: each seed's parallel efficiency; labels are mean
within-seed speedups. Each run contributes 20 warm updates after one first
update. Input preparation and placement are included; offline tokenization,
compilation and cache loading are not. This is replicated DDP strong scaling,
not weak scaling, FSDP capacity or converged pretraining. Tinybox and A100
results are not a controlled interconnect-only ablation.
