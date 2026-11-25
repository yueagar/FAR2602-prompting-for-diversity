'use strict';
// Judge-fixture checking editor (judge-selection plan, sections 2a, 2b, 2d).
// Mirrors inspection.js: autosave per item, version conflicts reported by the
// server, and a guard against leaving with unsaved edits. Each row carries its
// own questions; a row with one question leaves question_2 and answer_2 blank.
const EDITABLE = ['answer_1', 'answer_2', 'note'];
const FILE = document.body.dataset.file || 'the fixture sheet';
const ASSIGNMENT = {
  both: 'Checked by both members: the set is accepted only if both say yes.',
  primary: 'You are the primary checker for this item.',
  'cross-check': 'Cross-check: the other member is the primary checker; you answer independently.',
};
const KIND = {outlines: 'Outlines', response: 'Response', pair: 'Quality pair'};
let rows = [], index = 0, csrf = '', dirty = false, pending = null, timer = null;
const $ = id => document.getElementById(id);
const current = row => row.status !== 'superseded';
const finished = row => row.answer_1 !== '' && (!row.question_2 || row.answer_2 !== '');
const started = row => EDITABLE.some(key => row[key].trim() !== '');

function status(text, error = false) {
  $('status').textContent = text;
  $('status').classList.toggle('error', error);
  $('retry').style.display = error ? 'inline-block' : 'none';
}

function progress() {
  const live = rows.filter(current), done = live.filter(finished).length;
  $('progress').textContent = `${done} of ${live.length} items checked`;
  $('bar').max = live.length;
  $('bar').value = done;
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const answers = [row.answer_1, row.answer_2].filter(Boolean).join('/');
    const mark = !current(row) ? ' · superseded' : finished(row) ? ` · ${answers}`
      : started(row) ? ' · partial' : '';
    $('jump').options[i].textContent = `${i + 1} · ${row.item} · ${KIND[row.kind] || row.kind}${mark}`;
  }
}

function pressChoices() {
  for (const group of document.querySelectorAll('[data-answer]')) {
    const value = rows[index][group.dataset.answer];
    for (const button of group.querySelectorAll('[data-value]')) {
      button.setAttribute('aria-pressed', String(button.dataset.value === value));
    }
  }
}

function entriesOf(json) {
  try {
    const list = JSON.parse(json);
    if (Array.isArray(list)) return list;
  } catch (error) { /* shown below */ }
  return [{label: 'Text', text: '(This text could not be read.)'}];
}

function renderList(id, json) {
  const container = $(id);
  container.replaceChildren();
  for (const entry of entriesOf(json)) {
    const article = document.createElement('article');
    const title = document.createElement('h3');
    title.textContent = entry.label;
    const text = document.createElement('div');
    text.className = 'response';
    text.textContent = entry.text;
    article.append(title, text);
    container.append(article);
  }
}

function render() {
  const row = rows[index];
  $('item-id').textContent = `ITEM ${index + 1} / ${rows.length} · ${row.item}`;
  $('heading').textContent = row.heading;
  $('assignment').textContent = ASSIGNMENT[row.assignment] || '';
  $('superseded').hidden = current(row);
  $('task').textContent = row.task;
  renderList('context', row.context);
  renderList('texts', row.texts);
  $('question-1').textContent = row.question_1;
  $('question-2').textContent = row.question_2;
  $('second').hidden = !row.question_2;
  $('note').value = row.note;
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
      const payload = {item: row.item, version: row.version};
      for (const key of EDITABLE) payload[key] = row[key];
      status('Saving…');
      const response = await fetch('/api/fixtures', {
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
    const response = await fetch('/api/fixtures');
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
    const first = rows.findIndex(row => current(row) && !finished(row));
    index = first < 0 ? 0 : first;
    render();
  } catch (error) {
    status(error.message, true);
  }
}

// Pressing the selected answer again clears it.
for (const group of document.querySelectorAll('[data-answer]')) {
  for (const button of group.querySelectorAll('[data-value]')) {
    button.type = 'button';
    button.addEventListener('click', () => {
      if (!rows.length) return;
      const row = rows[index], key = group.dataset.answer;
      row[key] = row[key] === button.dataset.value ? '' : button.dataset.value;
      pressChoices();
      progress();
      changed();
      flush();
    });
  }
}
$('note').addEventListener('input', () => {
  if (!rows.length) return;
  rows[index].note = $('note').value;
  progress();
  changed();
});
$('previous').addEventListener('click', () => go(index - 1));
$('next').addEventListener('click', () => go(index + 1));
$('jump').addEventListener('change', () => go(Number($('jump').value)));
$('retry').addEventListener('click', flush);
$('unchecked').addEventListener('click', () => {
  for (let step = 1; step <= rows.length; step++) {
    const next = (index + step) % rows.length;
    if (current(rows[next]) && !finished(rows[next])) { go(next); return; }
  }
  status('All current items checked. Your CSV is ready to download.');
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
