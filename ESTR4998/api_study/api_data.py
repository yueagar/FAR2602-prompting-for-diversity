"""Development-only provenance and reuse of canonical local diagnostics."""
import importlib.util
import json
from pathlib import Path
import sys
from urllib import request

import api_runner as op
import runner_foundation_v1 as foundation

HERE = Path(__file__).resolve().parent
LOCAL = HERE.parent / 'local_model_study'
SPLIT = LOCAL / 'data/narrative_split_v0_4.json'
RECORDS = LOCAL / 'results/v0_5_decoding.jsonl.gz'
EMBEDDING = {'model_id': 'sentence-transformers/all-MiniLM-L6-v2',
             'revision': '1110a243fdf4706b3f48f1d95db1a4f5529b4d41', 'window': 256,
             'partition_threshold': .8, 'rke_bandwidth': .5}


def sha(path):
    return op.hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_editor_tool():
    raise FileNotFoundError("This authoring operation requires the original development fixture editor and records.")


def development_tasks():
    # Do not expose a general split loader: this path cannot select held-out.
    texts = load_editor_tool().development_texts()
    op.require(len(texts) == 10 and all(isinstance(t, str) and t for t in texts.values()), 'Expected ten frozen development tasks')
    return [{'id': key, 'text': text, 'source': 'frozen_development', 'text_sha256': op.hashlib.sha256(text.encode()).hexdigest(),
             'split_sha256': sha(SPLIT), 'records_sha256': sha(RECORDS)} for key, text in sorted(texts.items())]


def local_modules():
    if str(LOCAL / 'src') not in sys.path:
        sys.path.insert(0, str(LOCAL / 'src'))
    from localstudy import constraints, metrics
    return constraints, metrics


def text_diagnostics(task_id, text, finish):
    constraints, metrics = local_modules()
    checks = constraints.check_prompt(task_id, text)
    flags = metrics.degeneracy_flags(text)
    flags['refusal'] = flags['refusal'] or finish == 'content_filter'
    return {'flags': flags, 'constraint_checks': checks,
            'constraint_satisfied': constraints.satisfied(checks), 'truncated': finish == 'length',
            'constraint_version': constraints.CONSTRAINTS_VERSION,
            'flag_version': metrics.METRIC_VERSIONS['degeneracy_flags'] + '+api-content-filter-v1'}


def embedding_metrics(texts, embedder=None):
    """Full responses, canonical MiniLM windows; no substituted fake vectors."""
    from dataclasses import asdict
    _, metrics = local_modules()
    embedder = embedder or metrics.Embedder(EMBEDDING['model_id'], EMBEDDING['revision'], device='cpu', window=256)
    vectors = embedder.encode(texts)
    result = asdict(metrics.set_metrics(texts, vectors, .8, rke_bandwidth=.5))
    return {'values': result, 'embedding': EMBEDDING, 'metric_versions': metrics.METRIC_VERSIONS}


def fetch_snapshot(route, destination):
    """Public metadata GET only. Snapshots are not proof of live compatibility."""
    model = route['model']
    op.require('/' in model and all(c.isalnum() or c in '/_.:-' for c in model), 'Invalid model route')
    url = f'https://openrouter.ai/api/v1/models/{model}/endpoints'
    with request.build_opener(op.NoRedirect()).open(url, timeout=30) as response:
        raw = json.loads(response.read())
    matches = [e for e in raw['data']['endpoints'] if e.get('tag') == route['provider']]
    op.require(len(matches) == 1, 'Pin exactly one complete endpoint tag from the endpoint metadata')
    snapshot = {'fetched_utc': op.utcnow(), 'url': url, 'model': model, 'provider': route['provider'],
                'endpoint': matches[0], 'raw': raw}
    op.write_json(Path(destination), snapshot)
    return snapshot


def endpoint_check(route, parameters, base, execute=False):
    path = route.get('endpoint_snapshot')
    if not path:
        op.require(not execute, f"Snapshot missing for {route['label']}")
        return None
    path = (Path(base) / path).resolve()
    snapshot = json.loads(path.read_text())
    op.require(snapshot['model'] == route['model'] and snapshot['provider'] == route['provider'], 'Endpoint snapshot route differs')
    endpoint = snapshot['endpoint']
    # Runtime reservations need saved prices and both token ceilings. A missing
    # catalogue field is not permission to guess a price or unlimited context.
    # The shared reservation routine validates the same rates and enforces a
    # plain-text/no-tools payload before any client call. Media/search rates
    # remain in the snapshot; reasoning/cache-write rates enter its bound.
    foundation.text_price_tiers(endpoint.get('pricing'))
    op.require(isinstance(parameters, dict) and set(parameters) <= op.PARAMETERS,
               'Unsupported parameters outside the plain-text request contract')
    for key in ('context_length', 'max_completion_tokens'):
        op.require(op.integer(endpoint.get(key)), 'Snapshot requires explicit token ceilings')
    op.require(endpoint['max_completion_tokens'] <= endpoint['context_length'], 'Snapshot token ceilings are inconsistent')
    op.require(endpoint.get('tag') == route['provider'], 'Pin the complete endpoint tag')
    supported = set(endpoint['supported_parameters'])
    # Routing's structured_outputs capability corresponds to response_format.
    required = set(parameters)
    if route.get('json_schema'):
        op.require('structured_outputs' in supported, 'Endpoint does not advertise structured outputs')
        required.add('response_format')
    op.require(required <= supported, f"Unsupported parameters for {route['label']}: {sorted(required-supported)}")
    cap = endpoint.get('max_completion_tokens')
    op.require(cap is None or parameters['max_tokens'] <= cap, 'Configured output cap exceeds endpoint limit')
    return {'path': str(path), 'sha256': sha(path), 'snapshot': snapshot}


def runtime_route(route, parameters, base):
    checked = endpoint_check(route, parameters, base, True)
    snapshot = checked['snapshot']
    endpoint = snapshot['endpoint']
    canonical = snapshot.get('raw', {}).get('data', {}).get('canonical_slug')
    source = 'raw.data.canonical_slug' if canonical else 'requested_model'
    if not canonical:
        # Endpoint descriptors explicitly bind a provider display name to its
        # serving model ID. Accept only that exact saved mapping, never an
        # inferred date/name alias from the returned completion itself.
        prefix = str(endpoint.get('provider_name', '')) + ' | '
        name = endpoint.get('name')
        if endpoint.get('provider_name') and isinstance(name, str) and name.startswith(prefix):
            candidate = name[len(prefix):]
            if '/' in candidate and not any(c.isspace() or c == '|' for c in candidate):
                canonical, source = candidate, 'endpoint.name'
    return endpoint | {
        'canonical_slug': canonical or route['model'],
        'canonical_slug_source': source,
        'snapshot_sha256': checked['sha256'],
    }
