'use strict';
// Calibration-sheet editor. Mirrors quality.js: autosave per item, version
// conflicts reported by the server, and a guard against leaving with unsaved edits.
const CHOICES = ['same_content', 'style_affected'];
const EDITABLE = [...CHOICES, 'content_a', 'content_b', 'answer', 'rater_notes'];
const LABELS = {yes: 'Equivalent', no: 'Not equivalent'};
const FILE = document.body.dataset.file || 'the calibration sheet';
let rows = [], index = 0, csrf = '', dirty = false, pending = null, timer = null;
const $ = id => document.getElementById(id);
const answered = row => row.kind === 'question'
  ? row.answer.trim() !== ''
  : row.content_a.trim() !== '' && row.content_b.trim() !== '' && CHOICES.every(key => row[key] !== '');
const started = row => EDITABLE.some(key => row[key].trim() !== '');

function status(text, error = false) {
  $('status').textContent = text;
  $('status').classList.toggle('error', error);
  $('retry').style.display = error ? 'inline-block' : 'none';
}

function progress() {
  const done = rows.filter(answered).length;
  $('progress').textContent = `${done} of ${rows.length} items answered`;
  $('bar').max = rows.length;
  $('bar').value = done;
  for (let i = 0; i < rows.length; i++) {
    const mark = answered(rows[i]) ? ' · done' : started(rows[i]) ? ' · partial' : '';
    $('jump').options[i].textContent = `${i + 1} · ${rows[i].item_id}${mark}`;
  }
}

function pressChoices() {
  const row = rows[index];
  for (const button of document.querySelectorAll('[data-value]')) {
    const key = button.closest('[data-choice]').dataset.choice;
    button.setAttribute('aria-pressed', String(button.dataset.value === row[key]));
  }
}

function render() {
  const row = rows[index], pair = row.kind === 'pair';
  $('item-id').textContent = `ITEM ${index + 1} / ${rows.length} · ${row.item_id}`;
  $('request-title').textContent = pair ? 'Request' : 'Question';
  $('prompt').textContent = row.prompt;
  $('proposed').textContent = !pair && row.proposed ? `Proposed: ${row.proposed}` : '';
  $('responses').hidden = !pair;
  $('pair-fields').hidden = !pair;
  $('question-fields').hidden = pair;
  $('response-a').textContent = row.response_a;
  $('response-b').textContent = row.response_b;
  $('round1').textContent = pair
    ? `Your round-1 label: ${LABELS[row.your_round1_label] || 'not recorded'}` : '';
  for (const field of document.querySelectorAll('[data-field]')) field.value = row[field.dataset.field];
  $('jump').value = String(index);
  $('previous').disabled = index === 0;
  $('next').disabled = index === rows.length - 1;
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
      const payload = {item_id: row.item_id, version: row.version};
      for (const key of EDITABLE) payload[key] = row[key];
      status('Saving…');
      const response = await fetch('/api/calibration', {
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
    const response = await fetch('/api/calibration');
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
    const row = rows[index], key = button.closest('[data-choice]').dataset.choice;
    row[key] = row[key] === button.dataset.value ? '' : button.dataset.value;
    pressChoices();
    progress();
    changed();
    flush();
  });
}
for (const field of document.querySelectorAll('[data-field]')) {
  field.addEventListener('input', () => {
    if (!rows.length) return;
    rows[index][field.dataset.field] = field.value;
    progress();
    changed();
  });
}
$('previous').addEventListener('click', () => go(index - 1));
$('next').addEventListener('click', () => go(index + 1));
$('jump').addEventListener('change', () => go(Number($('jump').value)));
$('retry').addEventListener('click', flush);
$('unanswered').addEventListener('click', () => {
  for (let step = 1; step <= rows.length; step++) {
    const next = (index + step) % rows.length;
    if (!answered(rows[next])) { go(next); return; }
  }
  status('All items answered. Your CSV is ready to download.');
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
