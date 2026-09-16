"""Capture historical methods metadata without loading models or training data."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/raid/representax-paper')
ASSETS = Path('/raid/representax-paper-assets')


def collect():
    sources = {}

    def read(path, expected=None):
        path = Path(path)
        raw = path.read_bytes()
        digest = 'sha256:' + hashlib.sha256(raw).hexdigest()
        if expected is not None and expected != digest:
            raise ValueError(f'Historical hash mismatch: {path}')
        sources[str(path)] = {'sha256': digest, 'bytes': len(raw)}
        return json.loads(raw)

    def compact_manifest(value):
        if isinstance(value, dict):
            return {
                k: compact_manifest(v) for k, v in value.items()
                if k not in {'relevant_documents', 'probe_rows', 'probe_scores'}
            }
        if isinstance(value, list):
            return [compact_manifest(v) for v in value]
        return value

    evidence = read(HERE / 'evidence.json')
    records, manifests, environments = [], {}, {}
    for platform, panel in evidence['panels'].items():
        for run in panel['runs']:
            directory = Path(run['resolved_directory'])
            summary = read(directory / 'summary.json', run['summary_sha256'])
            manifest_path = Path(run.get('data_manifest_path', directory / 'data-manifest.json'))
            manifest_hash = run.get('data_manifest_sha256') or run['run']['data_manifest_sha256']
            if manifest_hash not in manifests:
                manifests[manifest_hash] = compact_manifest(read(manifest_path, manifest_hash))
            env_path = directory / 'environment.json'
            if env_path.exists():
                env = read(env_path)
                env_hash = sources[str(env_path)]['sha256']
            else:
                saved_run = read(directory / 'run.json')
                env = saved_run['environment']
                env_hash = 'sha256:' + hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()
            environments[env_hash] = env
            native = sorted((directory / 'run').glob('**/run.json'))
            config = None
            if run['framework'] == 'representax' and native:
                saved = read(native[0])
                config = {k: v for k, v in saved['config'].items()
                          if k not in {'evaluation', 'logging', 'checkpointing', 'export'}}
            fields = {'global_batch_size', 'batch_size', 'maximum_length',
                      'execution_sequence_length', 'precision', 'micro_batch_size',
                      'grad_cache_micro_batch_size', 'gradient_accumulation_steps',
                      'local_batch_size', 'sequence_length_buckets', 'steps',
                      'device_count', 'process_count', 'negative_scope'}
            capacities = {row['metrics']['perf/token_capacity'] / row['metrics']['perf/examples']
                          for row in run['metrics']
                          if 'perf/token_capacity' in row.get('metrics', {})}
            records.append({
                'platform': platform, 'recipe': run['recipe'], 'seed': run['seed'],
                'framework': run['framework'], 'source_commit': run['source_commit'],
                'manifest': manifest_hash, 'environment': env_hash,
                'summary': {k: v for k, v in summary.items() if k in fields},
                'config': config, 'contract': run['contract'],
                'source': run.get('source', run['run'].get('source')),
                'observed_token_capacity_per_example': sorted(capacities),
            })
    learning = {}
    for name in ['11-dense-retrieval-convergence', '13-image-text-convergence']:
        learning[name] = read(ROOT / name / 'summary.json')
    late_root = ROOT / '12-late-interaction-convergence/hard-negatives/runs'
    learning['late'] = {
        str(s): {'report': read(late_root / f'seed-{s}/report.json'),
                 'launch': read(late_root / f'seed-{s}/launch.json')}
        for s in (7, 42, 773)
    }
    omni = {}
    for arm in ('connectors', 'connectors-lora', 'full'):
        for seed in (7, 42, 773):
            path = ROOT / f'14-text-to-any-modality/runs/{arm}/seed-{seed}'
            saved = read(path / 'run/run.json')
            cfg = {k: v for k, v in saved['config'].items()
                   if k not in {'evaluation', 'logging'}}
            omni[f'{arm}/{seed}'] = {'config': cfg,
                                     'result': read(path / 'result.json'),
                                     'data_contract': saved['data_contract']}
    additional = {}
    for path in [
        ASSETS / 'dense-msmarco-full-unique/manifest.json',
        ASSETS / 'dense-transfer-evaluation/manifest.json',
        ASSETS / 'image-text-convergence/manifest.json',
        ASSETS / 'late-interaction-hard-negatives/manifest.json',
        ROOT / '14-text-to-any-modality/training-data/manifest.json',
    ]:
        additional[path.parent.name] = compact_manifest(read(path))
    return {'paired_runs': records, 'data_manifests': manifests,
            'environments': environments, 'learning': learning, 'omni': omni,
            'additional_manifests': additional, 'sources': sources}


if __name__ == '__main__':
    output = HERE / 'methods.json'
    if output.exists():
        raise SystemExit('methods.json already exists; historical snapshot is not overwritten')
    data = collect()
    output.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    print(f"Captured {len(data['paired_runs'])} runs and {len(data['sources'])} source hashes")
