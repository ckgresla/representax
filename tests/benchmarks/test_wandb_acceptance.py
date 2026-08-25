from benchmarks.wandb_acceptance import evaluation_cases, training_cases

from representax.tasks.registry import BUILTIN_TASKS


def test_wandb_acceptance_matrix_covers_every_builtin_task():
    assert set(training_cases()) == set(BUILTIN_TASKS.definitions)


def test_wandb_acceptance_matrix_covers_the_evaluator_inventory():
    assert set(evaluation_cases()) == {
        "classification",
        "information_retrieval",
        "jepa",
        "loss",
        "mse",
        "paraphrase_mining",
        "reranking",
        "reward",
        "similarity",
        "triplet",
    }
