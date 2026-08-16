# Sentence Transformers capability ledger

Representax pins Sentence Transformers 5.6.1 as a repository-only acceptance
oracle. This ledger maps its 28 released dense loss classes to scientific and
execution contracts; it is not a promise to reproduce upstream class names.

| Scientific contract | Sentence Transformers 5.6.1 classes | Representax status |
|---|---|---|
| In-batch negative ranking | `MultipleNegativesRankingLoss`, `CachedMultipleNegativesRankingLoss`, `MultipleNegativesSymmetricRankingLoss`, `CachedMultipleNegativesSymmetricRankingLoss` | Native: direct/cached, symmetric, graded positives, local/global negatives, and distributed GradCache |
| Labeled-pair regression and ranking | `CosineSimilarityLoss`, `CoSENTLoss`, `AnglELoss` | Native: fixed-shape pair batches, explicit validity, task-owned routes, and pinned value/representation-gradient parity |
| Pairwise contrastive learning | `ContrastiveLoss`, `OnlineContrastiveLoss` | Native: cosine, Euclidean, and Manhattan distance; ordinary and online mining share one scientific loss configuration |
| Explicit and label-mined triplets | `TripletLoss`, `BatchAllTripletLoss`, `BatchHardTripletLoss`, `BatchHardSoftMarginTripletLoss`, `BatchSemiHardTripletLoss` | Native: explicit routed triplets plus all, hard, hard soft-margin, and semi-hard mining over class-labeled batches; pinned value/representation-gradient parity |
| Embedding and score distillation | `MSELoss`, `EmbedDistillLoss`, `MarginMSELoss`, `DistillKLDivLoss` | Planned |
| Dimension and layer composition | `MatryoshkaLoss`, `Matryoshka2dLoss`, `AdaptiveLayerLoss` | Partial: MNR supports weighted Matryoshka dimensions; general loss and layer composition remain |
| Guide-model negative selection | `GISTEmbedLoss`, `CachedGISTEmbedLoss` | Planned |
| Contrastive tension | `ContrastiveTensionLoss` | Planned |
| Classification head training | `SoftmaxLoss` | Planned as a classification task, not a retrieval special case |
| Orthogonal regularization | `GlobalOrthogonalRegularizationLoss` | Planned as a composable regularizer |
| Denoising autoencoding | `DenoisingAutoEncoderLoss` | Planned as a reconstruction task |
| Mega-batch mining | `MegaBatchMarginLoss` | Planned as an execution and sampling policy |

“Native” means more than accepting a configuration: the formula, model-facing
batch contract, compiled training path, and pinned upstream numerical gate are
implemented. Performance comparisons remain a separate systems gate so a
correct implementation cannot acquire a speed claim without matched evidence.

Representax uses positive Euclidean and Manhattan distances for explicit
triplets, consistent with the released loss documentation and margin equation.
The pinned Sentence Transformers 5.6.1 callbacks return negative similarities
for those two explicit-triplet options, reversing that equation; acceptance
therefore exercises its cosine option rather than preserving the defect. The
four in-batch implementations match their released default Euclidean formulas.
