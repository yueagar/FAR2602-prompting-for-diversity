'use strict';
// Quality-sheet editor. Mirrors pair/pair.js: autosave per item, version conflicts
// reported by the server, and a guard against leaving with unsaved edits.
const SCORES = ['relevance', 'coherence', 'constraints', 'effectiveness'];
const EDITABLE = [...SCORES, 'rater_notes'];
const FILE = document.body.dataset.file || 'the quality sheet';
let rows = [], index = 0, csrf = '', dirty = false, pending = null, timer = null;
const $ = id => document.getElementById(id);
const scored = row => SCORES.every(key => row[key] !== '');
const started = row => SCORES.some(key => row[key] !== '');

function status(text, error = false) {
  $('status').textContent = text;
  $('status').classList.toggle('error', error);
  $('retry').style.display = error ? 'inline-block' : 'none';
}

function progress() {
  const done = rows.filter(scored).length;
  $('progress').textContent = `${done} of ${rows.length} items scored`;
  $('bar').max = rows.length;
  $('bar').value = done;
  for (let i = 0; i < rows.length; i++) {
    const mark = scored(rows[i]) ? ' · scored' : started(rows[i]) ? ' · partial' : '';
    $('jump').options[i].textContent = `${i + 1} · ${rows[i].item_id}${mark}`;
  }
}

function pressScores() {
  const row = rows[index];
  for (const button of document.querySelectorAll('[data-score]')) {
    button.setAttribute('aria-pressed', String(button.dataset.score === row[button.dataset.dimension]));
  }
}

function render() {
  const row = rows[index];
  $('item-id').textContent = `ITEM ${index + 1} / ${rows.length} · ${row.item_id}`;
  $('prompt').textContent = row.prompt;
  $('response').textContent = row.response;
  $('notes').value = row.rater_notes;
  $('jump').value = String(index);
  $('previous').disabled = index === 0;
  $('next').disabled = index === rows.length - 1;
  pressScores();
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
      const payload = {item_id: row.item_id, version: row.version};
      for (const key of EDITABLE) payload[key] = row[key];
      status('Saving…');
      const response = await fetch('/api/score', {
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
    const response = await fetch('/api/items');
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
    const first = rows.findIndex(row => !scored(row));
    index = first < 0 ? 0 : first;
    render();
  } catch (error) {
    status(error.message, true);
  }
}

// Five toggle buttons per dimension; pressing the selected score again clears it.
for (const scale of document.querySelectorAll('.scale')) {
  for (let score = 1; score <= 5; score++) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = String(score);
    button.dataset.dimension = scale.dataset.dimension;
    button.dataset.score = String(score);
    button.setAttribute('aria-pressed', 'false');
    button.setAttribute('aria-label', `${scale.dataset.name} ${score}`);
    button.addEventListener('click', () => {
      if (!rows.length) return;
      const row = rows[index], key = button.dataset.dimension;
      row[key] = row[key] === button.dataset.score ? '' : button.dataset.score;
      pressScores();
      progress();
      changed();
      flush();
    });
    scale.append(button);
  }
}

$('notes').addEventListener('input', () => {
  if (!rows.length) return;
  rows[index].rater_notes = $('notes').value;
  changed();
});
$('previous').addEventListener('click', () => go(index - 1));
$('next').addEventListener('click', () => go(index + 1));
$('jump').addEventListener('change', () => go(Number($('jump').value)));
$('retry').addEventListener('click', flush);
$('unscored').addEventListener('click', () => {
  for (let step = 1; step <= rows.length; step++) {
    const next = (index + step) % rows.length;
    if (!scored(rows[next])) { go(next); return; }
  }
  status('All items scored. Your CSV is ready to download.');
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
