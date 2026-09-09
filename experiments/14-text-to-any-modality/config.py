"""Frozen nine-run Jina Nano text-to-any-modality recipe."""

from pathlib import Path

from representax.config import (
    BatchConfig,
    CheckpointConfig,
    ComponentConfig,
    DataConfig,
    EvaluationConfig,
    ExportConfig,
    GradCacheConfig,
    InformationRetrievalEvaluatorConfig,
    JobConfig,
    LoggingConfig,
    LoRAConfig,
    ModelConfig,
    OptimizationConfig,
    PrecisionConfig,
    TrainingConfig,
)
from representax.data import mix, source
from representax.tasks.modifiers import MatryoshkaModifierConfig
from representax.tasks.retrieval import MNRConfig, RetrievalConfig

SEEDS = (7, 42, 773)
LEARNING_RATES = {"connectors": 2e-4, "connectors-lora": 5e-5, "full": 1e-5}
STEPS = 2000
DATA_MODULE = "experiments.14-text-to-any-modality.data"


def training_job(manifest, *, strategy, seed, assets: Path):
    if strategy not in LEARNING_RATES or seed not in SEEDS:
        raise ValueError("strategy/seed is outside the frozen experiment")
    paths = manifest["seeds"][str(seed)]
    names = ("image", "audio", "video") + (
        () if strategy == "connectors" else ("text",)
    )
    connectors = r"^\.(vision\.merger|audio\.projection)\."
    return JobConfig(
        name=f"paper-14-{strategy}-seed-{seed}",
        model=ModelConfig(
            target="representax.models.jina_v5.load_jina_v5_omni",
            parameters={
                "model_name_or_path": str(assets / "jina-v5-omni-nano"),
                "revision": "b7287f6b6b562e25bc4a28b939d1f936484b4137",
                "local_files_only": True,
                "parameter_dtype": "float32",
                "compute_dtype": "bfloat16",
                "sequence_length_buckets": [128, 512, 1024],
                "patch_count_buckets": [
                    256,
                    512,
                    1024,
                    2048,
                    4096,
                    8192,
                    16384,
                    32768,
                    65536,
                ],
                "audio_chunk_count_buckets": [8],
                "audio_token_count_buckets": [256],
                "image_min_pixels": 256 * 256,
                "image_max_pixels": 256 * 256,
                "video_min_pixels": 224 * 224,
                "video_max_pixels": 224 * 224,
            },
        ),
        task=RetrievalConfig(),
        loss=MNRConfig(scale=50.0, symmetric=True),
        loss_modifiers=(
            MatryoshkaModifierConfig(dimensions=(32, 64, 128, 256, 512, 768)),
        ),
        optimization=OptimizationConfig(
            optimizer=ComponentConfig(
                target="optax.adamw", parameters={"weight_decay": 0.01}
            ),
            schedule=ComponentConfig(
                target="optax.warmup_cosine_decay_schedule",
                parameters={
                    "init_value": 0.0,
                    "peak_value": LEARNING_RATES[strategy],
                    "warmup_steps": round(0.06 * STEPS),
                    "decay_steps": STEPS,
                    "end_value": 0.0,
                },
            ),
            max_gradient_norm=1.0,
        ),
        data=DataConfig(
            distribution=mix(
                *[
                    source(
                        paths[name],
                        name=name,
                        map=f"{DATA_MODULE}.map_{name}"
                        if name != "text"
                        else "representax.data.identity",
                    )
                    for name in names
                ],
                seed=seed,
                shuffle=False,
                sampling_unit="batch",
            ),
            collate=ComponentConfig(
                target="representax.tasks.retrieval.RetrievalCollator"
            ),
            num_threads=2,
            prefetch_buffer_size=2,
        ),
        training=TrainingConfig(
            global_batch_size=32,
            max_steps=STEPS,
            seed=seed,
            batch=BatchConfig(micro_batch_size=32),
            grad_cache=GradCacheConfig(implementation="custom_vjp", micro_batch_size=2),
            activation_rematerialization="full",
            adapter=LoRAConfig(rank=8, alpha=16, target_pattern=".*")
            if strategy == "connectors-lora"
            else None,
            trainable_pattern=".*"
            if strategy == "full"
            else connectors
            + (r"|\.lora_[ab]$" if strategy == "connectors-lora" else ""),
            trainable_embedding_rows={}
            if strategy == "full"
            else {".text.token_embedding": (128257, 128258)},
            precision=PrecisionConfig.bfloat16_mixed(),
        ),
        evaluation=EvaluationConfig(
            data=DataConfig(
                distribution=mix(
                    *[
                        source(
                            panel["path"], name=name, map="representax.data.identity"
                        )
                        for name, panel in manifest["evaluation"].items()
                    ],
                    shuffle=False,
                ),
                num_threads=2,
                prefetch_buffer_size=2,
            ),
            batch_size=2,
            every_steps=500,
            on_start=True,
            on_end=True,
            save_best=False,
            primary_metric="valid/flickr30k/cosine_ndcg@10",
            primary_metric_mode="max",
            evaluators=tuple(
                InformationRetrievalEvaluatorConfig(
                    name=name,
                    relevant_documents=panel["relevant_documents"],
                    accuracy_at_k=(1, 5, 10),
                    precision_recall_at_k=(1, 5, 10),
                )
                for name, panel in manifest["evaluation"].items()
            ),
        ),
        checkpointing=CheckpointConfig(every=1000, keep=2, asynchronous=False),
        logging=LoggingConfig(timing=True, accelerator=True, console_every=10),
        export=ExportConfig(enabled=True),
    )
