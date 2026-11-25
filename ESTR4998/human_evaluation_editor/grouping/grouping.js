'use strict';
// Each row is an independent complete-set judgment. No suggested partition is
// loaded, and a saved draft is not counted as a completed human judgment.
const EDITABLE = ['groups', 'status', 'exposure', 'note'];
const FILE = document.body.dataset.file || 'your grouping sheet';
const $ = id => document.getElementById(id);
let rows = [], index = 0, csrf = '', dirty = false, pending = null, timer = null;
let ready = false, loading = false;

function status(text, error = false) {
  $('status').textContent = text;
  $('status').classList.toggle('error', error);
  $('retry').style.display = error ? 'inline-block' : 'none';
}

function responses(row) { return JSON.parse(row.responses); }
function assignments(row) { return JSON.parse(row.groups); }
function allAssigned(row) {
  const groups = assignments(row);
  return responses(row).every(item => Number.isInteger(groups[item.id]) && groups[item.id] > 0);
}
function canComplete(row) { return allAssigned(row) && row.exposure !== ''; }
function completed(row) { return row.status === 'complete' && canComplete(row); }

function updateProgress() {
  const complete = rows.filter(completed).length;
  const unsure = rows.filter(row => row.status === 'unsure').length;
  $('progress').textContent = `${complete} of ${rows.length} sets complete${unsure ? ` · ${unsure} need discussion` : ''}`;
  $('bar').max = rows.length || 1;
  $('bar').value = complete;
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const state = completed(row) ? 'complete' : row.status === 'unsure' ? 'needs discussion' : 'draft';
    $('jump').options[i].textContent = `${i + 1} · ${row.item} · ${state}`;
  }
}

function updateSummary() {
  const row = rows[index], entries = responses(row), groups = assignments(row);
  const preview = new Map();
  for (const entry of entries) {
    const group = groups[entry.id];
    if (!group) continue;
    if (!preview.has(group)) preview.set(group, []);
    preview.get(group).push(entry.id);
  }
  const assigned = [...preview.values()].reduce((sum, ids) => sum + ids.length, 0);
  $('assignment-summary').textContent = `${assigned} of ${entries.length} responses assigned · ${preview.size} groups so far`;
  $('group-preview').replaceChildren();
  for (const [group, ids] of [...preview.entries()].sort((a, b) => a[0] - b[0])) {
    const chip = document.createElement('span');
    chip.className = 'group-chip';
    chip.textContent = `Group ${group}: ${ids.join(', ')}`;
    $('group-preview').append(chip);
  }
  $('complete').disabled = !canComplete(row) || row.status === 'complete';
  $('draft').disabled = row.status === 'draft';
  $('judgment-status').textContent = row.status === 'complete' ? 'Marked complete'
    : row.status === 'unsure' ? 'Needs discussion — unresolved' : 'Draft — not yet marked complete';
  $('completion-hint').textContent = row.status === 'complete'
    ? 'You can revise this judgment. Changing a group or exposure answer returns it to draft.'
    : !allAssigned(row) ? 'Assign every response to a group before marking complete.'
    : row.exposure === '' ? 'Record your earlier exposure before marking complete.'
    : 'Review your groups, then mark this set complete or flag an unresolved judgment.';
  updateProgress();
}

function renderResponses(row) {
  const entries = responses(row), groups = assignments(row);
  $('responses').replaceChildren();
  for (const entry of entries) {
    const article = document.createElement('article');
    const title = document.createElement('h3');
    title.textContent = `Response ${entry.id}`;
    const text = document.createElement('div');
    text.className = 'response';
    text.textContent = entry.text;
    if (!entry.text.trim()) {
      text.className = 'response empty-response';
      text.textContent = entry.text === '' ? '[Empty response]' : '[Whitespace-only response]';
    }
    const control = document.createElement('div');
    control.className = 'group-control';
    const label = document.createElement('label');
    label.htmlFor = `group-${entry.id}`;
    label.textContent = `Group for ${entry.id}`;
    const select = document.createElement('select');
    select.id = `group-${entry.id}`;
    select.dataset.response = entry.id;
    for (let number = 0; number <= entries.length; number++) {
      const option = document.createElement('option');
      option.value = number ? String(number) : '';
      option.textContent = number ? `Group ${number}` : 'Unassigned';
      select.append(option);
    }
    select.value = groups[entry.id] ? String(groups[entry.id]) : '';
    select.addEventListener('change', () => {
      const current = rows[index], next = assignments(current);
      if (select.value === '') delete next[entry.id];
      else next[entry.id] = Number(select.value);
      current.groups = JSON.stringify(next);
      if (current.status === 'complete') current.status = 'draft';
      updateSummary();
      changed();
      return flush();
    });
    control.append(label, select);
    article.append(title, text, control);
    $('responses').append(article);
  }
}

function render() {
  const row = rows[index];
  $('item-id').textContent = `SET ${index + 1} / ${rows.length} · ${row.item}`;
  $('task').textContent = row.task;
  $('exposure').value = row.exposure;
  $('note').value = row.note;
  $('jump').value = String(index);
  $('previous').disabled = index === 0;
  $('next').disabled = index === rows.length - 1;
  renderResponses(row);
  updateSummary();
  status(`Saved to ${FILE}`);
}

// Keep edits made while a request is in flight. Navigation waits for the same
// loop, so a failed or conflicting save cannot silently discard a judgment.
async function flush() {
  clearTimeout(timer);
  if (pending) return pending;
  pending = (async () => {
    while (dirty) {
      const row = rows[index], payload = {item: row.item, version: row.version};
      for (const key of EDITABLE) payload[key] = row[key];
      status('Saving…');
      const response = await fetch('/api/grouping', {
        method: 'POST',
        headers: {'Content-Type': 'application/json', 'X-CSRF-Token': csrf},
        body: JSON.stringify(payload),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error || 'Save failed. Your edits are still on this page.');
      row.version = body.version;
      dirty = EDITABLE.some(key => row[key] !== payload[key]);
    }
    if (ready) status(rows.length ? `Saved to ${FILE}` : 'Waiting for fresh responses.');
    return true;
  })().catch(error => { status(error.message, true); return false; })
    .finally(() => { pending = null; });
  return pending;
}

function changed() {
  dirty = true;
  status('Unsaved changes');
  clearTimeout(timer);
  timer = setTimeout(flush, 650);
}

async function go(target) {
  if (!ready || target < 0 || target >= rows.length) return;
  if (await flush() === false) { $('jump').value = String(index); return; }
  index = target;
  render();
  window.scrollTo({top: 0, behavior: 'smooth'});
}

function mark(value) {
  if (!rows.length) return;
  const row = rows[index];
  if (value === 'complete' && !canComplete(row)) return;
  if (value === 'unsure' && !row.note.trim()) {
    status('Add a note describing the uncertainty, then choose Needs discussion.', true);
    $('note').focus();
    return;
  }
  row.status = value;
  updateSummary();
  changed();
  return flush();
}

async function init() {
  if (loading) return;
  loading = true;
  $('download').hidden = true;
  status('Loading sheet…');
  try {
    const response = await fetch('/api/grouping');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not load the grouping sheet.');
    if (!Array.isArray(data.rows)) throw new Error('The grouping sheet could not be read.');
    // Do not make an unreadable response set look like a blank judgment.
    for (const row of data.rows) {
      const entries = responses(row), groups = assignments(row);
      if (!Array.isArray(entries) || !entries.length || !groups || Array.isArray(groups)
          || typeof groups !== 'object') throw new Error('The grouping sheet could not be read.');
    }
    rows = data.rows;
    csrf = data.csrf;
    ready = true;
    $('jump').replaceChildren();
    for (let i = 0; i < rows.length; i++) {
      const option = document.createElement('option');
      option.value = String(i);
      $('jump').append(option);
    }
    $('workspace').hidden = rows.length === 0;
    $('empty-state').hidden = rows.length !== 0;
    $('download').hidden = rows.length === 0;
    for (const id of ['jump', 'unchecked', 'exposure', 'note', 'unsure']) $(id).disabled = rows.length === 0;
    if (!rows.length) {
      updateProgress();
      status('Waiting for fresh responses.');
      return;
    }
    const first = rows.findIndex(row => !completed(row));
    index = first < 0 ? 0 : first;
    render();
  } catch (error) {
    ready = false;
    status(error.message, true);
  } finally { loading = false; }
}

$('exposure').addEventListener('change', () => {
  if (!rows.length) return;
  const row = rows[index];
  row.exposure = $('exposure').value;
  if (row.status === 'complete') row.status = 'draft';
  updateSummary(); changed();
  return flush();
});
$('note').addEventListener('input', () => {
  if (!rows.length) return;
  const row = rows[index];
  row.note = $('note').value;
  if (row.status === 'unsure' && !row.note.trim()) row.status = 'draft';
  updateSummary(); changed();
});
$('complete').addEventListener('click', () => mark('complete'));
$('unsure').addEventListener('click', () => mark('unsure'));
$('draft').addEventListener('click', () => mark('draft'));
$('previous').addEventListener('click', () => go(index - 1));
$('next').addEventListener('click', () => go(index + 1));
$('jump').addEventListener('change', () => go(Number($('jump').value)));
$('retry').addEventListener('click', () => ready ? flush() : init());
$('unchecked').addEventListener('click', () => {
  for (let step = 1; step <= rows.length; step++) {
    const next = (index + step) % rows.length;
    if (!completed(rows[next])) return go(next);
  }
  status('All sets marked complete. Keep judgments private until reconciliation.');
});
for (const link of [$('download'), ...document.querySelectorAll('nav.sheets a')]) {
  link.addEventListener('click', async event => {
    event.preventDefault();
    if (link === $('download') && (!ready || !rows.length)) return;
    if (await flush() !== false) window.location.assign(link.getAttribute('href'));
  });
}
window.addEventListener('beforeunload', event => {
  if (dirty || pending) { event.preventDefault(); event.returnValue = ''; }
});
init();
