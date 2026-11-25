"""Read the study's local key without shell evaluation or secret output."""
import os
from pathlib import Path


def load_env(path=None):
    path = Path(path) if path is not None else Path(__file__).with_name('.env')
    if not path.exists():
        return
    # Deliberately support just the one credential; do not execute shell text,
    # expand substitutions, or let a local file replace a supplied environment.
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name, separator, value = line.partition('=')
        if not separator or name.strip() != 'OPENROUTER_API_KEY':
            raise ValueError('Invalid .env entry; use OPENROUTER_API_KEY=value')
        value = value.strip()
        if value[:1] in ('"', "'"):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError('Unclosed quote in .env')
            value = value[1:-1]
        if any(c.isspace() for c in value):
            raise ValueError('API key must not contain whitespace')
        os.environ.setdefault('OPENROUTER_API_KEY', value)


def api_key():
    load_env()
    key = os.environ.get('OPENROUTER_API_KEY', '').strip()
    if not key:
        raise ValueError('Fill in OPENROUTER_API_KEY in the study .env or environment')
    return key
