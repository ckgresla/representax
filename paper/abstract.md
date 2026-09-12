# Representax: Scalable Representation Learning in JAX

New representation-learning ideas should be easy to try and practical to scale.
As research crosses the boundaries between objectives, modalities, and hardware,
its training infrastructure should make those boundaries easier to cross.
We built Representax, a unified JAX foundation for research across text, images,
audio, and video. It connects composable objectives and model adaptation with
multimodal data processing, memory-efficient training, evaluation, and distributed
execution through a shared training interface. We evaluate this foundation along
three complementary axes: performance against reference frameworks, useful
learning on held-out tasks, and multi-GPU strong scaling. Five-seed GPU and
multi-host TPU comparisons show throughput advantages across many workloads,
while identifying settings where reference implementations remain faster;
compilation is reported separately from steady-state training. Three-seed
adaptation studies improve dense and image-text retrieval, while text-anchored
multimodal training improves media retrieval and reveals tradeoffs in text
retention. A 505-million-parameter masked-language model achieves 3.81x and
6.87x strong-scaling speedups on four and eight A100 GPUs relative to one GPU,
with the global training batch held fixed. Representax provides a common
experimental foundation on which researchers can build new representation-learning
methods and explore how far they can go.
