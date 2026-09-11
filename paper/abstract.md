# Representax: Scalable Representation Learning in JAX

Representation-learning research spans diverse objectives, modalities, and
accelerator environments, yet developing new methods often requires rebuilding
training infrastructure. We introduce Representax, a JAX toolkit that brings
model adaptation, composable objectives, memory-efficient training, multimodal
data processing, evaluation, and distributed execution into a shared training
interface. We evaluate the toolkit through replicated framework comparisons
on GPUs and multi-host TPUs, end-to-end adaptation studies, and fixed-batch
multi-GPU scaling. Five-seed comparisons show workload-dependent throughput
advantages alongside regressions, with compilation reported separately from
steady-state optimizer-step time. Three-seed studies demonstrate improved
held-out dense and image-text retrieval. Text-anchored multimodal adaptation
improves image, audio, and video retrieval while exposing tradeoffs in text
retention across connector tuning, low-rank adaptation, and full fine-tuning.
For a 505-million-parameter masked-language-model workload, Representax achieves
3.81x and 6.87x strong-scaling speedups on four and eight A100 GPUs, respectively,
while holding the global batch fixed. These results position Representax as a
practical foundation for representation-learning research across modalities
and accelerators, rather than a new learning objective or a universally faster
execution backend.
