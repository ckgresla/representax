# Figure Revision Plan

Freeze the measured data, not the first visual encoding. These revisions need
no training runs. Do not select axes, rows or ranges to conceal regressions.
Keep the current figures as a reproducible first pass until the replacements
have been reviewed.

## Three Questions for the Main Text

| Question | Main figure | Supporting table |
|---|---|---|
| How does the shared toolkit perform against references? | Aligned horizontal speedup dot plots, separate GPU and TPU panels, grouped by task family | Absolute examples/s and optimizer steps/s; model, batch, shape, reference and timing exclusions |
| Can it learn useful representations? | Per-dataset changes in held-out retrieval, including multimodal text retention | Initial and final scores, sample SD, sample counts and training budget |
| Does the same scientific workload scale? | Speedup versus GPU count, measured seeds and ideal `y=x` line | Tokens/s, seconds/update, efficiency, microbatch/accumulation and memory |

## Concrete Changes

1. **Comparison plot:** keep a log-ratio axis centered on parity and show all
   paired seed ratios. Group task families rather than displaying thirteen
   undifferentiated rows. Add dense eager and Inductor controls explicitly;
   otherwise the main image can imply that the eager reference is the best
   available GPU execution. Historical late interaction should be visually
   separated, with cache-qualified TPU rows visibly marked. A shared absolute
   examples/s axis across text, audio and video would obscure task differences.
2. **Cross-accelerator scatter:** move to the appendix or remove if it does not
   add insight beyond the aligned panels. It repeats the same ratios and needs
   a lookup key. A paired GPU/TPU ratio plot by recipe may communicate the
   backend reversal more directly. This is not a Pareto frontier: no common
   cost/resource axis was measured, and the allocations differ.
3. **Learning plot:** do not imply that a 0.1 gain on two different evaluation
   sets means equal learning progress. Use separate dataset panels and tables
   for exact initial/final values. Dense transfer scores lack initial
   measurements; show finals only. Keep the late-interaction negative result
   in an explicitly labeled appendix panel, not among the positive examples.
4. **Multimodal plot:** replace endpoint-joining lines with per-seed change in
   nDCG@10, strategy on the categorical axis and a visible zero line, faceted by
   held-out dataset. This foregrounds gains versus forgetting and does not look
   like a two-point learning curve. Retain each arm's own baseline and report
   differing source mixtures. No composite modality score.
5. **Scaling plot:** use speedup versus 1/2/4/8 GPUs as the single main panel,
   with ideal linear scaling. Put efficiency and absolute rates in a compact
   table: the existing throughput and efficiency panels encode almost the
   same information. Use true numeric spacing, a baseline-inclusive axis and
   all three seeds; do not zoom the axis to make tiny variance look large.
6. **Startup costs:** keep compilation/cache-load and complete-step rates
   separately in an appendix table. Distinguish known cold compilation,
   cached loads and compilation-plus-first-step measurements. Do not plot
   missing reference instrumentation as zero startup cost. Break-even plots
   require compatible timing boundaries; not every current row supplies them.

## Examples Worth Borrowing From

- **Scikit-learn:** the paper places a concrete cross-library timing table
  alongside a description of interface consistency and implementation
  tradeoffs. This supports our toolkit-first narrative; the useful lesson is
  not to make a compiler benchmark the whole paper.
  [Sections 4-5 and Table 1](https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf)
- **PyTorch Distributed:** separates latency breakdown, tuning and scalability
  into distinct figures. Borrow the one-question-per-figure structure, not its
  whole grid of diagnostics. Our interconnect diagnosis belongs in supporting
  material unless it is necessary to explain the principal result.
  [Section 5](https://arxiv.org/html/2006.15704v1#S5)
- **Megatron-LM:** pairs scaling curves with an explicit configuration table.
  Borrow that adjacency: readers should immediately see what is held fixed.
  Do not compare our strong-scaling percentages with its model weak-scaling
  percentages as if they were matched results.
  [Table 1 and Figure 5](https://arxiv.org/pdf/1909.08053)

These are editorial proposals from the existing data, not a new experiment
plan. Abstract selection is the immediate priority; revise and visually
inspect the figures after agreeing on that narrative.
