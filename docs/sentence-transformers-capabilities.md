# Sentence Transformers capability ledger

Representax pins Sentence Transformers 5.6.1 as a repository-only acceptance
oracle. This ledger maps its 29 released dense loss classes to scientific and
execution contracts; it is not a promise to reproduce upstream class names.
Loss coverage is only one layer of the dense system. Data, sampling, trainer,
evaluation, inference, and final-artifact gaps are tracked separately in the
[`dense-system audit`](dense-system-audit.md).

| Scientific contract | Sentence Transformers 5.6.1 classes | Representax status |
|---|---|---|
| In-batch negative ranking | `MultipleNegativesRankingLoss`, `CachedMultipleNegativesRankingLoss`, `MultipleNegativesSymmetricRankingLoss`, `CachedMultipleNegativesSymmetricRankingLoss` | Native: direct/cached, symmetric, graded positives, local/global negatives, and distributed GradCache |
| Labeled-pair regression and ranking | `CosineSimilarityLoss`, `CoSENTLoss`, `AnglELoss` | Native: fixed-shape pair batches, explicit validity, task-owned routes, and pinned value/representation-gradient parity |
| Pairwise contrastive learning | `ContrastiveLoss`, `OnlineContrastiveLoss` | Native: cosine, Euclidean, and Manhattan distance; ordinary and online mining share one scientific loss configuration |
| Explicit and label-mined triplets | `TripletLoss`, `BatchAllTripletLoss`, `BatchHardTripletLoss`, `BatchHardSoftMarginTripletLoss`, `BatchSemiHardTripletLoss` | Native: explicit routed triplets plus all, hard, hard soft-margin, and semi-hard mining over class-labeled batches; pinned value/representation-gradient parity |
| Embedding and score distillation | `MSELoss`, `EmbedDistillLoss`, `MarginMSELoss`, `DistillKLDivLoss` | Native: broadcast or per-column teacher embeddings with MSE, L2, or cosine matching; positive-minus-negative score regression; temperature-scaled distribution KL; pinned value/representation-gradient parity |
| Dimension and layer composition | `MatryoshkaLoss`, `Matryoshka2dLoss`, `AdaptiveLayerLoss` | Native: registered loss modifiers reuse one representation or layerwise encoder pass; Matryoshka composes with cached MNR/GIST, and BERT, MPNet, ModernVBERT, and serialized sentence chains expose one-scan layerwise outputs |
| Guide-model negative selection | `GISTEmbedLoss`, `CachedGISTEmbedLoss` | Native: offline guide representations, absolute/relative filtering, explicit negatives, direct and bounded replay, and score-row chunking |
| Contrastive tension | `ContrastiveTensionLoss`, `ContrastiveTensionLossInBatchNegatives` | Native: explicit dual-encoder state, aligned-pair BCE, symmetric in-batch CE, cosine/dot similarity, and trainable temperature |
| Classification head training | `SoftmaxLoss` | Native: explicit encoder-plus-head model state and configurable pair features under a classification task |
| Orthogonal regularization | `GlobalOrthogonalRegularizationLoss` | Native: modality-neutral representation batches, named mean/second-moment terms, and mean/sum aggregation |
| Denoising autoencoding | `DenoisingAutoEncoderLoss` | Native: explicit encoder/causal-decoder composition, damaged-input batches, shifted clean-token targets, and padding-aware cross entropy |
| Mega-batch mining | `MegaBatchMarginLoss` | Native: direct hardest-negative margins plus bounded candidate mining and gradient replay as an execution policy |

“Native” means more than accepting a configuration: the formula, model-facing
batch contract, compiled training path, and pinned upstream numerical gate are
implemented. Performance comparisons remain a separate systems gate so a
correct implementation cannot acquire a speed claim without matched evidence.

Every native row is closed by
[`tests/tasks/test_sentence_transformers_parity.py`](../tests/tasks/test_sentence_transformers_parity.py).
Its inventory assertion requires all 29 upstream classes to appear in 39
same-tensor value and relevant-gradient cases. The separate performance
case measures synchronized, warmed forward and backward work per class and
warns—rather than invalidating numerical correctness—if a native objective is
slower on an uncontrolled device.

On the August 16, 2026 RTX 4090 acceptance run (batch 48, representation
dimension 128), every native loss-plus-representation-backward program was
faster than its paired Sentence Transformers 5.6.1 class: the observed range
was 3.00x to 84.69x, with native compilation between 0.078 and 1.030 seconds.
This deliberately measures objective math over already-computed embeddings; it
is not an encoder, input-pipeline, or full optimizer-step speed claim.

Guide encodings are batch artifacts rather than a live hidden model. Likewise,
classification heads, contrastive-tension encoder branches and temperature,
and reconstruction decoders remain explicit Equinox model state. Mega-batch
candidate selection is an execution schedule, while its margins remain the
scientific loss. These boundaries keep every trainable parameter visible to
Optax and every memory-changing policy visible to the job configuration.

Sentence Transformers may place an optional learned projection inside
`MSELoss` or `EmbedDistillLoss`. Representax keeps trainable state in the model:
compose a projection head before these losses when student and teacher
dimensions differ. The loss then receives equal projected dimensions, avoiding
optimizer-visible parameters hidden inside a task closure.

Representax uses positive Euclidean and Manhattan distances for explicit
triplets, consistent with the released loss documentation and margin equation.
The pinned Sentence Transformers 5.6.1 callbacks return negative similarities
for those two explicit-triplet options, reversing that equation; acceptance
therefore exercises its cosine option rather than preserving the defect. The
four in-batch implementations match their released default Euclidean formulas.
