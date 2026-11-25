"""Research transport with allowlisted diagnostics; smoke client stays intact."""
import json
import re
from urllib import error, request

import api_runner as op


def failure_diagnostic(exc):
    # Preserve the supplied foundation's exact-message allowlist. Arbitrary
    # exception messages can contain provider text or credentials.
    result = {'category': 'unclassified_failure', 'http_status': None}
    message = exc.args[0] if type(exc) is RuntimeError and len(exc.args) == 1 and type(exc.args[0]) is str else None
    match = re.fullmatch(r'OpenRouter HTTP ([1-5][0-9]{2}); no automatic retry', message or '')
    if match:
        result.update(category='http_error', http_status=int(match[1]))
    else:
        categories = {
            'API transport/JSON error; billing may be unknown; no automatic retry': 'transport_or_json_error',
            'API timeout; billing may be unknown': 'timeout',
            'API transport error; billing may be unknown': 'transport_error',
            'API invalid JSON; billing may be unknown': 'invalid_json',
            'Refusing an API redirect': 'redirect_refused',
        }
        result['category'] = categories.get(message, result['category'])
    return result


def diagnostic_context(exc, attempt, stage, existing=None):
    """Add fixed phase/billing facts without copying exception/provider text.

    Diagnosis is descriptive. Retry eligibility still depends on the saved
    response, known billing, setup flag and the phase's explicit retry limit.
    """
    stages = {'transport', 'route_validation', 'response_validation',
              'judgment_parse', 'fixture_validation', 'billing'}
    op.require(stage in stages, 'Unknown diagnostic stage')
    base = existing or failure_diagnostic(exc)
    categories = {'unclassified_failure', 'http_error', 'transport_or_json_error',
                  'timeout', 'transport_error', 'invalid_json', 'redirect_refused',
                  'api_error', 'routing_mismatch', 'response_validation_error',
                  'judgment_parse_error', 'fixture_validation_error',
                  'finish_reason_error', 'unknown_billing'}
    category = base.get('category')
    category = category if category in categories else 'unclassified_failure'
    raw = attempt.get('response')
    choices = raw.get('choices') if isinstance(raw, dict) else None
    raw_finish = (choices[0].get('finish_reason') if isinstance(choices, list)
                  and len(choices) == 1 and isinstance(choices[0], dict) else None)
    finish = raw_finish if raw_finish in ('stop', 'length', 'content_filter') else (
        'unrecognized' if raw_finish is not None else None)
    if category == 'unclassified_failure':
        category = {'response_validation': 'response_validation_error',
                    'judgment_parse': 'judgment_parse_error',
                    'fixture_validation': 'fixture_validation_error',
                    'billing': 'unknown_billing'}.get(stage, category)
        if stage == 'response_validation' and finish in ('length', 'content_filter', 'unrecognized'):
            category = 'finish_reason_error'
    status = base.get('http_status')
    return {'category': category,
            'http_status': status if type(status) is int and 400 <= status <= 599 else None,
            'stage': stage, 'response_received': 'response' in attempt,
            'finish_reason': finish,
            'billing_status': 'known' if attempt.get('reported_cost_usd') is not None else 'unknown'}


class Client(op.Client):
    def complete(self, body):
        req = request.Request(op.ENDPOINT, data=op.canonical(body).encode(), headers={
            'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json',
            'X-OpenRouter-Title': 'ESTR OpenRouter diversity study',
            'X-OpenRouter-Metadata': 'enabled'}, method='POST')
        try:
            with request.build_opener(op.NoRedirect()).open(req, timeout=self.timeout) as response:
                return json.loads(response.read())
        except error.HTTPError as exc:
            status = exc.code
            exc.close()
            raise RuntimeError(f'OpenRouter HTTP {status}; no automatic retry') from None
        except TimeoutError:
            raise RuntimeError('API timeout; billing may be unknown') from None
        except error.URLError as exc:
            category = 'timeout' if isinstance(exc.reason, TimeoutError) else 'transport error'
            raise RuntimeError(f'API {category}; billing may be unknown') from None
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise RuntimeError('API invalid JSON; billing may be unknown') from None
