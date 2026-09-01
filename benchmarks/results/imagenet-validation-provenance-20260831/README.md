# ImageNet-1K validation-label provenance

Verdict: **PASS — official-quality ImageNet-1K validation evaluation is
admissible for this local tree.**

The filename-to-synset mapping was derived directly from the official
`ILSVRC2012_devkit_t12` `meta.mat` and validation ground-truth file. The local
50,000-image tree exactly matches that mapping (1,000 classes, 50 images each).
The pre-existing local validation tar matches the canonical archive MD5, and
every extracted local JPEG is byte-identical to its member in that archive.

Reproduce from the repository root (downloads only the public 2.5 MB devkit):

```bash
.venv/bin/python scripts/verify_imagenet_validation.py \
  --val-root /raid/datasets/imagenet/val \
  --val-archive /raid/datasets/imagenet/ILSVRC2012_img_val.tar \
  --output-dir /raid/representax/paper-campaign-v1/preflight/imagenet-validation-provenance-20260831
```

Use `--devkit /path/to/ILSVRC2012_devkit_t12.tar.gz` for an offline rerun.

Sources and access details are recorded in `result.json`. The official ImageNet
download page requires login for original images; this run used the existing
local archive and did not access credentials. The devkit's bundled `COPYING`
text is fingerprinted in `result.json`; this evidence redistributes neither the
devkit archive nor ImageNet images.

The generated 50,000-row mapping is kept under that `/raid` artifact root, not
duplicated in Git. Key SHA-256 hashes:

- `filename-to-synset.tsv`: `b5b25a74f93140f3e3febc504cd1e77411d38604e61da4801e2cde971771ba54`
- `result.json`: `7e0c7d3ff8fa20969a7927cf04727aff178c94bd79118a5d7f331ac23db5062b`
- local image content manifest: `12c5bc17eb80a04dddb6ddc2957254171111d905c4b9c4e71e7963784764cb87`
