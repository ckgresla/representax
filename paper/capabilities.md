# Capability and Evidence Table

Execution means completed optimizer updates, not merely import or forward support.
Experiment 10 uses maintained reference code with the recorded TPU integrations;
it does not establish that every upstream trainer runs unchanged.

| Workload | Native system | GPU evidence | Multi-host TPU evidence | Reference | Quality evidence |
|---|---|---|---|---|---|
| Dense retrieval | MPNet; ETTIN in longer adaptation | Exp. 10, five seeds; eager + dense Inductor control | Exp. 10, five seeds; matched local negative pools | Sentence Transformers | Exp. 11, three seeds; held-out retrieval |
| Similarity / pair classification | MPNet Base and BERT Base | Exp. 10, five seeds each | Exp. 10, five seeds each | Sentence Transformers | Short trajectories only |
| Cross-encoder | Cross-encoder recipe | Exp. 10, five seeds | Exp. 10, five seeds | Sentence Transformers | Short trajectories only |
| Late interaction | GTE-ModernColBERT | Historical Exp. 10, qualified | Historical Exp. 10, qualified | PyLate | Exp. 12: retained negative held-out result |
| Outcome / process reward | Qwen3-0.6B | Exp. 10, five seeds each | Exp. 10, five seeds each | TRL | Short trajectories only |
| Image-text | CLIP ViT-B/32 | Exp. 10, five seeds | Exp. 10, five seeds; input-cache caveat | Sentence Transformers | Exp. 13, three seeds; COCO to Flickr30k |
| Audio-text / video-text | Omni 3B comparison recipe | Exp. 10, five seeds each | Exp. 10, five seeds each | Sentence Transformers | Exp. 14 uses a different model/recipe |
| V-JEPA 2.1 | Video predictive representation task | Exp. 10, five seeds | Exp. 10, five seeds; input-cache caveat | facebookresearch/vjepa2 | Short trajectories only |
| Text-to-any-modality | Jina v5 Omni Nano | Exp. 14, nine native runs | Not measured for this recipe | No paired training reference | Three adaptation strategies; held-out media/text |
| Masked language modeling | Controlled 505M ModernBERT | Exp. 15, 1/2/4/8 A100; three seeds | Not measured for this recipe | No paired training reference | Short real-text scaling workload, not full pretraining |

## Execution and Lifecycle

| Facility | Evidence boundary |
|---|---|
| Multi-host TPU | Four worker VMs, four local v5e chips each; native distributed JAX and reference PJRT ranks. This is multi-host TPU execution, not multi-node GPU execution. |
| Multi-GPU DDP | Twelve A100 runs; fixed work, deferred synchronization after accumulation. |
| FSDP / hybrid sharding | Implemented and probed historically; no accepted capacity-frontier result in this paper freeze. |
| Checkpoint/resume/export | Experiments 11 and 13 record midpoint resume and export/reload; do not extrapolate exact reload validation to all task/backend combinations. |
| Multimodal checkpoint/resume | Experiment 14 training/evaluation and earlier integration checks; final review did not rerun all exported artifacts for fresh reload parity. |
| GradCache | Rematerialized path and dense custom VJP; task/layout compatibility is not universal. |
| Bucketing / prefetch | Enabled per recipe; preserve actual input shapes and timing boundaries, not a blanket claim of matched preprocessing. |
| Evaluation | Shared native held-out evaluators for longer adaptation studies; short-run loss correlation is not a substitute for learned-quality equivalence. |

## Recorded Reference Revisions

| Project | Revision in the paired-panel record |
|---|---|
| Sentence Transformers | `7d3eb16a65f62045226e08082ade63cbc71c97a4` |
| PyLate | `346ea2b9c273256919809ac35b3a055970780c9c` |
| TRL | `a7be897f5c8d7b52161f9f8a47d8e6242456b898` |
| V-JEPA 2 | `204698b45b3712590f06245fbfba32d3be539812` |

These are historical revisions from the recorded experiment inventory, not
claims about current upstream behavior. Before release, reconcile them with
each run's environment/provenance files and publish the reproduction commands
and locks. In particular, global-negative Sentence Transformers TPU support is
a limitation of the tested integration, not a blanket claim about multi-node
Sentence Transformers or PyTorch.
