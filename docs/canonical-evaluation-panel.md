# Canonical first-paper evaluation panel

This is a research-orientation artifact for choosing the finite Representax
paper panel. It is not a redistribution of third-party datasets and its rows
must not become packaged test fixtures. Dataset adapters should pin the source
revision, preserve the upstream splits, and stream or cache the original
artifacts according to their licenses.

The panel deliberately measures the main representation geometries rather than
copying all of MTEB. System-performance comparisons and model-quality
evaluations are separate contracts: quality datasets establish that training
semantics are non-inferior, while paired jobs on identical hardware establish
throughput, utilization, capacity, and time-to-quality claims.

## Proposed compact panel

| Signal | Evaluator | Initial datasets | Primary metric |
|---|---|---|---|
| Graded symmetric similarity | `SimilarityEvaluator` | STSBenchmark.v2, SICK-R | cosine Spearman |
| Pair decisions | `PairClassificationEvaluator` | Sprint Duplicate Questions | maximum average precision across declared similarity functions |
| Frozen linear separability | `ClassificationProbeEvaluator` | Banking77.v2 | accuracy, macro F1 secondary |
| Unsupervised global geometry | `ClusteringEvaluator` | Twenty Newsgroups.v2 | V-measure |
| Text retrieval | `InformationRetrievalEvaluator` | SciFact, NFCorpus, ArguAna | nDCG@10; MRR, MAP, recall secondary |
| Multilingual retrieval | `InformationRetrievalEvaluator` | MIRACL | macro language nDCG@10 plus per-language scores |
| Cross-lingual alignment | ordinary IR/qrels adapter | FLORES-200 | F1 or retrieval accuracy |
| Image-text retrieval | `InformationRetrievalEvaluator` | Flickr30k | recall@1/5/10 and nDCG@10 |
| Audio-text retrieval | `InformationRetrievalEvaluator` | AudioCaps | hit-rate@5 and recall@1/5/10 |
| Video-text retrieval | `InformationRetrievalEvaluator` | MSR-VTT | recall@1/5/10 and nDCG@10 |
| JEPA representation quality | `JEPARepresentationEvaluator` plus frozen probe | ImageNet-1K | frozen linear-probe top-1; k-NN and collapse diagnostics secondary |

SICK-R is retained beside STS-B because its controlled compositional and
entailment-derived pairs probe a different failure mode. MIRACL and FLORES are
not optional decorations: multilingual and cross-lingual alignment are core
embedding workloads. FLORES does not require a special “translation” evaluator;
aligned sentences can be expressed through the ordinary query/corpus/qrels
contract.

## What frozen-embedding classification means

The encoder is frozen. Representax embeds the train, validation, and test
examples once; fits only a lightweight linear classifier on the train
embeddings; chooses probe hyperparameters on validation data; and reports the
held-out labels. This measures how linearly accessible task information is in
the representation, not how well the encoder can be fine-tuned.

The contract must pin feature normalization, classifier family, regularization
grid, class weighting, optimizer or solver tolerance, maximum iterations, seed,
and selection split. The same extracted embeddings must produce matching probe
results in Representax and the maintained reference evaluator.

## Dataset previews

The excerpts below are intentionally short. They show the data geometry and
label meaning; the pinned upstream artifact remains the source of truth.

### STSBenchmark.v2 — graded similarity

- Source: [`mteb/STSBenchmarkv2`](https://huggingface.co/datasets/mteb/STSBenchmarkv2), revision `93b628c3969a75e76727db2b7ee252e53e96268d`.
- Rows: 5,749 train, 1,485 validation, 1,362 test after duplicate removal.
- Preview: “A girl is styling...” / “A girl is brushing...” → `2.5`.
- Preview: “Men play soccer...” / “Boys are playing...” → `3.6`.
- Preview: “Woman measuring an ankle.” / “Woman measures an ankle.” → `5.0`.

The original 2017 STS Benchmark contains 8,628 pairs: 5,749 train, 1,500
development, and 1,379 test. By genre it contains 4,299 news, 3,250 image
caption, and 1,079 forum pairs. MTEB's recommended v2 adapter removes 32
duplicate evaluation pairs, hence the smaller 8,596-row total.

### SICK-R — compositional similarity

- Source: [`mteb/sickr-sts`](https://huggingface.co/datasets/mteb/sickr-sts), revision `20a6d6f312dd54037fe07a32d58e5e168867909d`.
- Rows: 9,927 test pairs; license `CC BY-NC-SA 3.0`.
- Preview: “Kids playing in a yard...” / “Boys playing in a yard...” → `4.5`.
- Preview: “Children playing inside...” / “Kids playing in a yard...” → `3.2`.
- Preview: “Boys playing outdoors...” / “Kids outdoors near a man...” → `4.7`.

### Sprint Duplicate Questions — pair classification

- Source: [`mteb/sprintduplicatequestions-pairclassification`](https://huggingface.co/datasets/mteb/sprintduplicatequestions-pairclassification), revision `d66bd1f72af766a5cc4b0ca5e00c162f89e8cc46`.
- Test payload: 101,000 pairs, of which 1,000 are positive; upstream license is unspecified.
- Preview: “USB modem signal strength” / “Does my modem have a weak signal?” → duplicate.
- Preview: “DuraMax GPS on/off” / “Turn GPS on or off” → duplicate.
- Preview: “USB modem signal strength” / “Check Nexus software updates” → not duplicate.

### Banking77.v2 — frozen classification probe

- Source: [`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77), revision `18072d2685ea682290f7b8924d94c62acc19c0b2`; MIT.
- Rows: 9,993 train and 3,076 test across 77 intents.
- Preview: “How do I locate my card?” → `card_arrival`.
- Preview: “My new card has not arrived.” → `card_arrival`.
- Preview: “I ordered a card; help.” → `card_arrival`.

### Twenty Newsgroups.v2 — clustering

- Source: [`mteb/twentynewsgroups-clustering`](https://huggingface.co/datasets/mteb/twentynewsgroups-clustering), revision `6125ec4e24fa026cec8a478383ee943acfbd5449`.
- Evaluation payload: ten deterministic 1,000-document clustering samples; upstream license is unspecified.
- Preview: “Motorola MC143150 and MC143120” → `sci.electronics`.
- Preview: “Windows 3.1 for sale” → `misc.forsale`.
- Preview: “Gospel Dating” → `alt.atheism`.

### SciFact — scientific claim retrieval

- Source: [`mteb/scifact`](https://huggingface.co/datasets/mteb/scifact), revision `d56462d0e63a25450459c4f213e49ffdb866f7f9`; `CC BY-NC 4.0`.
- Size: 1,109 queries, 5,183 abstracts, and 339 test qrels.
- Preview: “0-dimensional biomaterials show inductive properties.” → nanotechnology/stem-cell abstract `31715818`.
- Preview: “1,000 Genomes maps rare variants...” → rare-variant abstract `14717500`.
- Preview: “1/2000 have abnormal PrP positivity.” → prion survey `13734012`.

### NFCorpus — graded biomedical retrieval

- Source: [`mteb/nfcorpus`](https://huggingface.co/datasets/mteb/nfcorpus), revision `ec0fa4fe99da2ff19ca1214b7966684033a58814`.
- Size: 3,237 queries, 3,633 documents, and 12,334 test qrels; upstream license is unspecified.
- Preview: “Do statins cause breast cancer?” → cholesterol-raft cancer paper `MED-2427`.
- Preview: “Exploiting autophagy to live longer” → caloric-restriction paper `MED-2513`.
- Preview: “Reduce dietary alkylphenols” → nonylphenol paper `MED-2644`.

### ArguAna — counterargument retrieval

- Source: [`mteb/arguana`](https://huggingface.co/datasets/mteb/arguana), revision `c22ab2a51041ffd869aaddef7af8d8215647e41a`; `CC BY-SA 4.0`.
- Size: 1,406 queries and 8,674 candidate arguments.
- Preview: “Vegetarianism helps the environment” → opposing local-food argument.
- Preview: “It is immoral to kill animals” → opposing human-rights argument.
- Preview: “Vegetarianism is healthier” → opposing balanced-diet argument.

### MIRACL — multilingual retrieval

- Source: [`mteb/MIRACLRetrieval`](https://huggingface.co/datasets/mteb/MIRACLRetrieval), revision `9c09abc13478308c27598f350e31d8f06b9b5481`; `CC BY-SA 4.0`.
- Coverage: 18 languages with separate queries, corpora, and qrels.
- German preview: “Wo ist das Gebiet der Irokesen-Indianer in Kanada?”
- German preview: “Wie viele Arten von Jellyfish?”
- German preview: “Wie groß ist der größte Python?”

The pilot may use a few languages for rapid iteration, but the paper must report
per-language results and a declared macro aggregation rather than hiding
regressions inside one multilingual average.

### FLORES-200 — cross-lingual alignment

- Source: [`mteb/FloresBitextMining`](https://huggingface.co/datasets/mteb/FloresBitextMining), revision `2144d16cc15edd22d4a9237d12bff5f31f5c07fc`; `CC BY-SA 4.0`.
- Rows: 997 development and 1,012 devtest aligned records across 200 languages.
- Preview alignment: “We now have four-month-old mice...” ↔ German equivalent.
- Preview alignment: “Dr. Ehud Ur, professor of medicine...” ↔ Spanish equivalent.
- Preview alignment: “Like some other experts, he is skeptical...” ↔ German equivalent.

### Flickr30k — image/text retrieval

- Source: [`mteb/flickr30kt2i`](https://huggingface.co/datasets/mteb/flickr30kt2i), revision `e819702b287bfbe084e129a61f308a802b7c108e`; `CC BY-SA 4.0`.
- Size: 1,000 images, 5,000 captions, and 5,000 qrels.
- Three captions for image `d_s0000000`: “man with pierced ears...”; “beer-can crocheted hat”; “man with gauges and glasses...”.

### AudioCaps — audio/text retrieval

- Source: [`mteb/audiocaps_a2t`](https://huggingface.co/datasets/mteb/audiocaps_a2t), revision `acfbf827c27f81787800129443780c072dc8ae62`; MIT.
- Size: 883 audio queries, 4,411 captions, and 4,411 qrels.
- Caption previews: “rattling noise and sharp vibrations”; “rocket, explosion, and fire”; “humming, speech, and laughter”.

### MSR-VTT — video/audio/text retrieval

- Source: [`mteb/MSR-VTT`](https://huggingface.co/datasets/mteb/MSR-VTT), revision `4661603cee25c1fd370e5478a2953203cf37155b`; upstream license is unspecified.
- Size: 879 test videos with audio and captions.
- Preview: “a person is connecting something to system”.
- Preview: “a woman creating a fondant baby and flower”.
- Preview: “a boy plays grand theft auto 5”.

### ImageNet-1K — JEPA representation transfer

ImageNet access is gated and its samples must not be copied into this repository.
The canonical JEPA protocol freezes the pretrained backbone, extracts the
declared representation, trains a pinned linear probe on ImageNet-1K train, and
reports top-1 on validation. Representax should also report k-NN accuracy,
per-dimension variance, effective rank, covariance condition, and SIGReg terms
as diagnostics; those diagnostics do not replace downstream transfer quality.

## Historical and current interpretation

STS-B remains relevant in 2026 for historical continuity and as a cheap test of
graded symmetric similarity. Sentence-BERT, SimCSE, MTEB, and later embedding
work all report STS-B directly or through the MTEB STS category. Current model
papers increasingly emphasize MTEB/MMTEB category and aggregate results rather
than treating standalone STS-B as decisive. Representax should therefore retain
STSBenchmark.v2 but never infer retrieval, clustering, multilingual, or
multimodal quality from it.

The paired systems workloads are not universal values supplied by Sentence
Transformers. Sentence Transformers supplies maintained implementations,
evaluator semantics, and useful recipes. Representax must define and publish the
scientifically matched jobs, run both systems on the same machines, and measure
them afresh. Published model-card scores may validate quality plumbing; published
throughput from different hardware cannot establish our systems claim.

## Canonical references

- [STS Benchmark / SemEval 2017](https://aclanthology.org/S17-2001/)
- [Sentence-BERT](https://aclanthology.org/D19-1410/)
- [SimCSE](https://arxiv.org/abs/2104.08821)
- [MTEB](https://arxiv.org/abs/2210.07316)
- [MMTEB](https://arxiv.org/abs/2502.13595)
- [Improving Text Embeddings with Large Language Models](https://arxiv.org/abs/2401.00368)
- [Qwen3 Embedding](https://arxiv.org/abs/2506.05176)
- [jina-embeddings-v5-text](https://arxiv.org/abs/2602.15547)
- [LeJEPA](https://arxiv.org/abs/2511.08544)
