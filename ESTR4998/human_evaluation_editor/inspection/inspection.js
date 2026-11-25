'use strict';
// Manual-inspection editor. Mirrors calibration.js:
// autosave per item, version conflicts reported by the server, and a guard
// against leaving with unsaved edits. The instrument's verdict (score line,
// ranking side, per-response labels) stays hidden until the reader asks for it.
const EDITABLE = ['agrees', 'note'];
const FILE = document.body.dataset.file || 'the inspection sheet';
// What to look for, per measure, from the guide's tables. For constraint and
// flag items the item's view names the instruments' conclusion ("judged 5", the
// flag), so `hideView` keeps it out of the heading and `revealed` holds the
// wording that names it; both appear only after the reader reveals the verdict.
const GUIDE = {
  functional_classes: {
    name: 'Functional classes',
    ask: 'How many genuinely different answers are there? Count the stories with a different premise, plot or central idea, ignoring names, wording and length. After revealing, check the classifier\'s class label on each response.',
    mismatch: 'your count differs from the score by 3 or more; or the labels put clearly different stories in one class (often because they share an opening), or split near-identical stories.',
    note: 'Start with your own count and identify the responses you would merge.',
  },
  semantic_diversity: {
    name: 'Semantic diversity',
    ask: 'Do the responses differ in what happens, or only in surface topic, format or length?',
    mismatch: 'a high score comes from format variety (list against prose, headings, length) or off-topic answers rather than different content; or a low score hides real variety.',
  },
  reference_novelty: {
    name: 'Novelty against the default reference',
    ask: 'Compare with what the default arm usually writes for this prompt. Are these responses different from that?',
    mismatch: 'a high score comes from framing ("Certainly! Here\'s a creative…"), length or format rather than new content.',
  },
  distinct_2: {
    name: 'Distinct-2',
    ask: 'Is the vocabulary varied because the content is, or because of length, odd words, non-English text or typos? Is a low score caused by repetition or templated phrasing?',
    mismatch: 'the score tracks length or noise rather than variety.',
  },
  quality: {
    name: 'Judged quality',
    ask: 'Would you give this response a high score on relevance, coherence, constraints and effectiveness? Check the request\'s constraints yourself. Is it padded, generic or off-topic?',
    mismatch: 'a good response is scored low, or a weak one high. Name the dimension in your note.',
  },
  constraints: {
    name: 'Countable constraints',
    hideView: true,
    ask: 'Does the response meet every countable constraint in the request, such as the number of sentences, paragraphs or items? Count them yourself before revealing the verdict.',
    mismatch: 'your count disagrees with the instruments. Reveal the verdict to compare.',
    revealed: {
      ask: 'The deterministic check says this response breaks a countable constraint, yet the judge gave CONSTRAINTS 5. Compare with your own count.',
      mismatch: 'the deterministic check is wrong (a bug in constraints.py: quote the exact text in your note). If the check is right, the judge\'s 5 is the failure: answer "no".',
    },
  },
  flags: {
    name: 'Degeneracy check',
    hideView: true,
    ask: 'Is the response degenerate in any way: repetitive, empty or very short, refusing, disclaiming ("As an AI…"), mixed-script, or leaking the instruction? Decide before revealing the verdict.',
    mismatch: 'your judgement disagrees with the automatic flag. Reveal the verdict to compare.',
    revealed: {
      ask: 'Is the flag named in the verdict correct? Is the response actually what the flag says?',
      mismatch: 'the flag fires on a response that is fine.',
    },
  },
};
const SIDES = {highest: 'Highest-scoring', lowest: 'Lowest-scoring'};
let rows = [], index = 0, csrf = '', dirty = false, pending = null, timer = null;
const revealed = new Set();
const $ = id => document.getElementById(id);
const answered = row => row.agrees !== '';
const started = row => EDITABLE.some(key => row[key].trim() !== '');
const guide = row => GUIDE[row.measure] || {name: row.measure, ask: 'Does the score match what you see?', mismatch: 'it does not.'};

function status(text, error = false) {
  $('status').textContent = text;
  $('status').classList.toggle('error', error);
  $('retry').style.display = error ? 'inline-block' : 'none';
}

function progress() {
  const done = rows.filter(answered).length;
  $('progress').textContent = `${done} of ${rows.length} items checked`;
  $('bar').max = rows.length;
  $('bar').value = done;
  for (let i = 0; i < rows.length; i++) {
    const mark = answered(rows[i]) ? ` · ${rows[i].agrees}` : started(rows[i]) ? ' · partial' : '';
    $('jump').options[i].textContent = `${i + 1} · ${rows[i].item} · ${guide(rows[i]).name}${mark}`;
  }
}

function pressChoices() {
  for (const button of document.querySelectorAll('[data-value]')) {
    button.setAttribute('aria-pressed', String(button.dataset.value === rows[index].agrees));
  }
}

function responsesOf(row) {
  try {
    const list = JSON.parse(row.responses);
    if (Array.isArray(list)) return list;
  } catch (error) { /* shown below */ }
  return [{label: 'Response', detail: '', text: '(This item\'s responses could not be read.)'}];
}

function renderResponses(row, show) {
  const container = $('responses');
  container.replaceChildren();
  for (const entry of responsesOf(row)) {
    const article = document.createElement('article');
    const title = document.createElement('h3');
    title.textContent = entry.label || 'Response';
    if (show && entry.detail) {
      const detail = document.createElement('span');
      detail.className = 'detail';
      detail.textContent = ` (${entry.detail})`;
      title.append(detail);
    }
    const text = document.createElement('div');
    text.className = 'response';
    text.textContent = entry.text;
    article.append(title, text);
    container.append(article);
  }
}

function renderVerdict() {
  const row = rows[index], show = revealed.has(row.item), info = guide(row);
  const wording = show && info.revealed ? info.revealed : info;
  $('heading').textContent = info.hideView && !show ? info.name : `${info.name} · ${row.view}`;
  $('ask').textContent = wording.ask;
  $('mismatch').textContent = wording.mismatch;
  $('verdict').hidden = !show;
  $('reveal').textContent = show ? 'Hide the instrument\'s verdict' : 'Show the instrument\'s verdict';
  $('reveal').setAttribute('aria-expanded', String(show));
  const unit = row.item.startsWith('S') ? 'set' : 'response';
  $('side').textContent = SIDES[row.side] ? `${SIDES[row.side]} ${unit} (${row.view})`
    : info.hideView ? `Instrument verdict: ${row.view}` : '';
  $('score-line').textContent = row.score_line;
  renderResponses(row, show);
}

function render() {
  const row = rows[index], info = guide(row);
  $('item-id').textContent = `ITEM ${index + 1} / ${rows.length} · ${row.item}`;
  $('prompt').textContent = row.prompt;
  $('note').placeholder = info.note || 'One sentence on what you see…';
  $('note').value = row.note;
  $('jump').value = String(index);
  $('previous').disabled = index === 0;
  $('next').disabled = index === rows.length - 1;
  renderVerdict();
  pressChoices();
  progress();
  status(`Saved to ${FILE}`);
}

// Saves the current item until it matches what the server holds. Concurrent
// callers share one save loop; resolves false (with the error shown) on failure.
async function flush() {
  clearTimeout(timer);
  if (pending) return pending;
  pending = (async () => {
    while (dirty) {
      const row = rows[index];
      const payload = {item: row.item, version: row.version};
      for (const key of EDITABLE) payload[key] = row[key];
      status('Saving…');
      const response = await fetch('/api/inspection', {
        method: 'POST',
        headers: {'Content-Type': 'application/json', 'X-CSRF-Token': csrf},
        body: JSON.stringify(payload),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error || 'Save failed');
      row.version = body.version;
      dirty = EDITABLE.some(key => row[key] !== payload[key]);
    }
    status(`Saved to ${FILE}`);
    return true;
  })().catch(error => { status(error.message, true); return false; }).finally(() => { pending = null; });
  return pending;
}

function changed() {
  dirty = true;
  status('Unsaved changes');
  clearTimeout(timer);
  timer = setTimeout(flush, 650);
}

async function go(target) {
  if (target < 0 || target >= rows.length) return;
  const saved = await flush();
  if (saved === false) { $('jump').value = String(index); return; }
  index = target;
  render();
  window.scrollTo({top: 0, behavior: 'smooth'});
}

async function init() {
  try {
    const response = await fetch('/api/inspection');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);
    rows = data.rows;
    csrf = data.csrf;
    if (!rows.length) throw new Error('The sheet contains no items');
    for (let i = 0; i < rows.length; i++) {
      const option = document.createElement('option');
      option.value = String(i);
      $('jump').append(option);
    }
    const first = rows.findIndex(row => !answered(row));
    index = first < 0 ? 0 : first;
    render();
  } catch (error) {
    status(error.message, true);
  }
}

// Pressing the selected answer again clears it.
for (const button of document.querySelectorAll('[data-value]')) {
  button.type = 'button';
  button.addEventListener('click', () => {
    if (!rows.length) return;
    const row = rows[index];
    row.agrees = row.agrees === button.dataset.value ? '' : button.dataset.value;
    pressChoices();
    progress();
    changed();
    flush();
  });
}
$('note').addEventListener('input', () => {
  if (!rows.length) return;
  rows[index].note = $('note').value;
  progress();
  changed();
});
$('reveal').addEventListener('click', () => {
  if (!rows.length) return;
  const item = rows[index].item;
  if (revealed.has(item)) revealed.delete(item); else revealed.add(item);
  renderVerdict();
});
$('previous').addEventListener('click', () => go(index - 1));
$('next').addEventListener('click', () => go(index + 1));
$('jump').addEventListener('change', () => go(Number($('jump').value)));
$('retry').addEventListener('click', flush);
$('unchecked').addEventListener('click', () => {
  for (let step = 1; step <= rows.length; step++) {
    const next = (index + step) % rows.length;
    if (!answered(rows[next])) { go(next); return; }
  }
  status('All items checked. Your CSV is ready to download.');
});
// Save before leaving for the download or another sheet.
for (const link of [$('download'), ...document.querySelectorAll('nav.sheets a')]) {
  link.addEventListener('click', async event => {
    event.preventDefault();
    if (await flush() !== false) window.location.assign(link.getAttribute('href'));
  });
}
window.addEventListener('beforeunload', event => {
  if (dirty || pending) { event.preventDefault(); event.returnValue = ''; }
});
init();
