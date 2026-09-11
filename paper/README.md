# Paper Working Set

- `abstract.md`: proposed abstract, ready for author review, not submitted.
- `draft.md`: claim-bounded manuscript scaffold and current results narrative.
- `capabilities.md`: capability/evidence boundaries and recorded reference pins.
- `evidence.json`: frozen compact evidence, raw metric records and source hashes.
- `results.md`: generated framework table; `figures/`: generated PDF/PNG figures.
- `captions.md`: statistical conventions and figure-specific limitations.

No training code or measured artifacts are changed by this directory. The
roadmap and remaining submission tasks stay in `../todo.org`.

Rebuild figures without checkpoints, model downloads or access to `/raid`:

```bash
experiments/.venv/bin/python paper/build.py render
experiments/.venv/bin/python -m unittest discover -s paper -p test_build.py
```

The renderer requires Matplotlib and NumPy. Inter is vendored under its OFL
license in `assets/`; source: `https://github.com/rsms/inter`,
`docs/font-files/InterVariable.ttf`, downloaded 2026-09-11. The font bytes are
hashed in the evidence manifest. Colors use the author's named light palette.

The evidence was captured with:

```bash
experiments/.venv/bin/python paper/build.py freeze
```

`freeze` refuses to overwrite an existing snapshot. It verifies historical
paired metric/summary hashes and recomputes rates from complete step intervals;
it also checks the A100 per-step token/mask hashes and library-source equality.
The snapshot is not a checkpoint archive. It preserves sufficient data for these
figures, while the original `/raid/representax-paper` run trees retain larger
logs, environments, source patches and model artifacts. The current lock is
identified separately from historical run environments. No schema versioning
or new experiment dispatcher is introduced.

## Claim-to-Evidence Map

| Claim | Source | Limit |
|---|---|---|
| Workload-dependent framework throughput | Exp. 10 GPU/TPU `results.json`, per-run metrics and summaries | Short windows; local negative pools on TPU; cache/late-interaction qualifications |
| Useful dense learning | Exp. 11 summary and evaluation history | NanoMSMARCO improvement; transfer finals without initial transfer baselines |
| Useful image-text learning | Exp. 13 summary | COCO fine-tuning, not CLIP pretraining from scratch |
| Multimodal adaptation tradeoffs | Exp. 14 summary with per-arm evaluation history | Different mixtures; text-to-any, not any-to-any |
| Negative late-interaction result | Exp. 12 hard-negative per-seed reports | Small held-out panel; cause unresolved |
| 3.81x / 6.87x strong scaling | Exp. 15 A100 summary and 12 raw results | One workload; DDP; short window; not FSDP capacity |

Figure readiness is not submission readiness. Primary-source citations,
venue-format/anonymity checks, author review and public artifact packaging
remain before submission. Nothing here is uploaded or submitted automatically.
