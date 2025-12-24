const ui = {
  status: document.querySelector('#status'),
  input: document.querySelector('#input'),
  output: document.querySelector('#output'),
  log: document.querySelector('#log'),
  stats: document.querySelector('#stats'),
  fields: document.querySelector('#fields'),
  syncScroll: document.querySelector('#sync-scroll'),
  venueNormalize: document.querySelector('#venue-normalize'),
  autoMerge: document.querySelector('#auto-merge'),
  trimBlankLines: document.querySelector('#trim-blank-lines'),
  showDiff: document.querySelector('#show-diff'),
  diffPanel: document.querySelector('#diff-panel'),
  diff: document.querySelector('#diff'),
  highlights: {
    input: document.querySelector('#input-highlight'),
    output: document.querySelector('#output-highlight'),
  },
  buttons: {
    sample: document.querySelector('#sample-btn'),
    clear: document.querySelector('#clear-btn'),
    selectDefault: document.querySelector('#select-default'),
    selectAll: document.querySelector('#select-all'),
    copyOutput: document.querySelector('#copy-output'),
    downloadOutput: document.querySelector('#download-output'),
    copyLog: document.querySelector('#copy-log'),
    downloadLog: document.querySelector('#download-log'),
    resolveDuplicates: document.querySelector('#resolve-duplicates'),
  },
  mergeModal: document.querySelector('#merge-modal'),
  mergeGroups: document.querySelector('#merge-groups'),
  mergeApply: document.querySelector('#merge-apply'),
  mergeCancel: document.querySelector('#merge-cancel'),
  mergeClose: document.querySelector('#merge-modal-close'),
};

const FALLBACK_ALL_FIELDS = [
  'author','title','year','journal','booktitle','volume','number','pages','publisher',
  'doi','url','editor','month','organization','school','institution','series','edition',
  'chapter','isbn','issn','type','howpublished','note','keywords','abstract','address',
  'crossref'
];
const FALLBACK_DEFAULT_FIELDS = ['title','author','journal','year','booktitle'];
const CUSTOM_FIELD_ORDER = FALLBACK_ALL_FIELDS;

const sampleBib = `@inproceedings{mildenhall2020nerf,
  title={nerf: Representing scenes as neural Radiance Fields for View Synthesis},
  author={Mildenhall, Ben and Srinivasan, Pratul P and Tancik, Matthew and Barron, Jonathan T and Ramamoorthi, Ravi and Ng, Ren},
  booktitle={European conference on computer vision},
  year={2020},
  organization={Springer}
}

@article{kerbl20233d,
  title={3d Gaussian splatting for real-time radiance field rendering.},
  author={Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  journal={ACM trans. Graph.},
  volume={42},
  number={4},
  pages={139--1},
  year={2023}
}

@misc{easyocr,
  author = {Jaided AI},
  title = {EasyOCR},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\\url{https://github.com/JaidedAI/EasyOCR}},
}`;

const state = {
  pyodidePromise: null,
  processingTimer: null,
  allFields: FALLBACK_ALL_FIELDS.slice(),
  defaultFields: FALLBACK_DEFAULT_FIELDS.slice(),
  modifiedKeys: new Set(),
  lastProcessed: '',
  isSyncing: false,
  duplicates: [],
  mergePlan: null,
};

setFieldLists(FALLBACK_ALL_FIELDS, FALLBACK_DEFAULT_FIELDS);
ui.output.value = '';
ui.log.textContent = '';
wireEvents();
state.pyodidePromise = loadRuntime();
loadSampleOnStart();

function escapeHtml(str) {
  return (str ?? '').toString().replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function setStatus(text, tone = 'info') {
  ui.status.textContent = text;
  ui.status.dataset.tone = tone;
}

function setDiffVisible(visible) {
  if (ui.diffPanel) ui.diffPanel.classList.toggle('hidden', !visible);
  if (ui.log) ui.log.classList.toggle('hidden', visible);
}

function renderDiff(diffText) {
  if (!ui.diff) return;
  if (!diffText || !diffText.trim()) {
    ui.diff.innerHTML = '<span class="diff-meta">No differences.</span>';
    return;
  }
  ui.diff.innerHTML = buildDiffHTML(diffText);
  ui.diff.scrollTop = 0;
}

function buildDiffHTML(diffText) {
  const lines = (diffText || '').split('\n');
  return lines
    .map((line) => {
      let cls = 'diff-line diff-meta';
      if (line.startsWith('---') || line.startsWith('+++')) cls = 'diff-line diff-meta';
      else if (line.startsWith('@@')) cls = 'diff-line diff-hunk';
      else if (line.startsWith('+')) cls = 'diff-line diff-add';
      else if (line.startsWith('-')) cls = 'diff-line diff-del';
      else cls = 'diff-line';
      return `<span class="${cls}">${escapeHtml(line)}</span>`;
    })
    .join('\n');
}

async function loadDataFiles(pyodide) {
  const baseFiles = [
    { path: 'data/all_bibtex_fields.txt', url: './data/all_bibtex_fields.txt' },
    { path: 'data/lowercase_words.txt', url: './data/lowercase_words.txt' },
    { path: 'data/preserve_terms.txt', url: './data/preserve_terms.txt' },
    { path: 'data/protected_terms.txt', url: './data/protected_terms.txt' },
  ];
  const venueFiles = ['cs', 'stat', 'math', 'eess', 'physics', 'q-bio', 'econ'].map((name) => ({
    path: `data/venues/${name}.yml`,
    url: `./data/venues/${name}.yml`,
  }));
  const files = [...baseFiles, ...venueFiles];

  try {
    const dataDir = pyodide.FS.analyzePath('data');
    if (!dataDir.exists) {
      pyodide.FS.mkdir('data');
    }
    const venuesDir = pyodide.FS.analyzePath('data/venues');
    if (!venuesDir.exists) {
      pyodide.FS.mkdir('data/venues');
    }
  } catch (err) {
    console.warn('Unable to prepare data directory; falling back to defaults.', err);
    return;
  }

  for (const file of files) {
    try {
      const res = await fetch(file.url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      pyodide.FS.writeFile(file.path, text);
    } catch (err) {
      console.warn(`Could not load ${file.url}; falling back to defaults.`, err);
    }
  }
}

async function loadRuntime() {
  try {
    setStatus('Loading Pyodide runtime…');
    const pyodide = await loadPyodide({ indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.25.1/full/' });

    const [coreCode] = await Promise.all([
      fetch('./process_core.py').then((res) => res.text()),
      loadDataFiles(pyodide),
    ]);
    pyodide.FS.writeFile('process_core.py', coreCode);
    pyodide.runPython('import sys; sys.path.append(\".\")');
    pyodide.runPython('import process_core');

    try {
      const pyFields = pyodide.runPython('sorted(process_core.ALL_BIBTEX_FIELDS)');
      const pyDefaults = pyodide.runPython('sorted(process_core.DEFAULT_KEEP_FIELDS)');
      setFieldLists(
        Array.from(pyFields.toJs({ create_proxies: false })),
        Array.from(pyDefaults.toJs({ create_proxies: false }))
      );
      pyFields.destroy();
      pyDefaults.destroy();
    } catch (err) {
      console.warn('Could not read field list from Python; using defaults.', err);
    }

    setStatus('Runtime ready — paste or upload BibTeX.', 'success');
    return pyodide;
  } catch (err) {
    console.error(err);
    setStatus('Runtime failed to load. Check access to cdn.jsdelivr.net.', 'warn');
    throw err;
  }
}

function wireEvents() {
  ui.buttons.sample.addEventListener('click', () => {
    loadSampleOnStart(true);
  });

  ui.buttons.clear.addEventListener('click', () => {
    ui.input.value = '';
    ui.output.value = '';
    ui.log.textContent = '';
    if (ui.showDiff) ui.showDiff.checked = false;
    setDiffVisible(false);
    renderDiff('');
    ui.stats.textContent = 'Not processed yet';
    state.lastProcessed = '';
    state.modifiedKeys.clear();
    resetMergeState(true);
    clearHighlights();
    setStatus('Cleared.');
  });

  ui.buttons.copyOutput.addEventListener('click', () => {
    copyText(getOutputForActions(), 'Processed output copied.');
  });

  ui.buttons.copyLog.addEventListener('click', () => {
    if (ui.showDiff?.checked) {
      copyText(ui.diff?.textContent || '', 'Diff copied.');
      return;
    }
    copyText(ui.log.textContent, 'Log copied.');
  });

  ui.buttons.downloadOutput.addEventListener('click', () => {
    const text = getOutputForActions();
    if (!text.trim()) {
      setStatus('No output to download.', 'warn');
      return;
    }
    downloadText('bibliography_cleaned.bib', text);
    setStatus('Downloaded bibliography_cleaned.bib');
  });

  ui.buttons.downloadLog.addEventListener('click', () => {
    if (ui.showDiff?.checked) {
      const text = ui.diff?.textContent || '';
      if (!text.trim()) {
        setStatus('No diff to download.', 'warn');
        return;
      }
      downloadText('bibtex_diff.txt', text);
      setStatus('Downloaded diff.');
      return;
    }
    if (!ui.log.textContent.trim()) {
      setStatus('No log to download.', 'warn');
      return;
    }
    downloadText('processing_log.txt', ui.log.textContent);
    setStatus('Downloaded log.');
  });

  ui.buttons.selectDefault.addEventListener('click', () => setFields(new Set(state.defaultFields)));
  ui.buttons.selectAll.addEventListener('click', () => setFields(new Set(state.allFields)));

  document.querySelectorAll('input[name="title-mode"]').forEach((el) => {
    el.addEventListener('change', () => scheduleProcessing());
  });
  document.querySelectorAll('input[name="venue-mode"]').forEach((el) => {
    el.addEventListener('change', () => scheduleProcessing());
  });
  if (ui.venueNormalize) {
    ui.venueNormalize.addEventListener('change', () => {
      updateVenueNormalizeState();
      scheduleProcessing();
    });
  }
  document.querySelectorAll('input[name="venue-standardize"]').forEach((el) => {
    el.addEventListener('change', () => scheduleProcessing());
  });
  updateVenueNormalizeState();

  if (ui.showDiff) {
    ui.showDiff.addEventListener('change', () => {
      const enabled = ui.showDiff.checked;
      setDiffVisible(enabled);
      if (!enabled) renderDiff('');
      if (enabled && ui.diff) ui.diff.innerHTML = '<span class="diff-meta">Computing diff…</span>';
      if (enabled) scheduleProcessing();
    });
  }

  ui.autoMerge.addEventListener('change', () => {
    resetMergeState();
    refreshMergeButton();
    scheduleProcessing();
  });

  ui.input.addEventListener('input', () => {
    resetMergeState();
    setOverlayHTML('input', ui.input.value);
    updateOverlayScroll('input');
    scheduleProcessing();
  });
  ui.input.addEventListener('scroll', onInputScroll);

  ui.output.addEventListener('scroll', onOutputScroll);
  ui.output.addEventListener('input', () => {
    setOverlayHTML('output', ui.output.value);
  });

  ui.fields.addEventListener('change', (e) => {
    if (e.target.name === 'keep-field') scheduleProcessing();
  });

  document.querySelector('#file-input').addEventListener('change', handleFileUpload);

  if (ui.buttons.resolveDuplicates) {
    ui.buttons.resolveDuplicates.addEventListener('click', openMergeModal);
  }
  if (ui.mergeApply) ui.mergeApply.addEventListener('click', applyMergeSelections);
  if (ui.mergeCancel) ui.mergeCancel.addEventListener('click', closeMergeModal);
  if (ui.mergeClose) ui.mergeClose.addEventListener('click', closeMergeModal);
  if (ui.mergeModal) {
    ui.mergeModal.addEventListener('click', (e) => {
      if (e.target === ui.mergeModal || e.target.classList.contains('modal__backdrop')) {
        closeMergeModal();
      }
    });
  }
}

function loadSampleOnStart(clearModal = false) {
  resetMergeState(clearModal);
  ui.input.value = sampleBib.trim();
  setStatus('Sample loaded, processing…');
  scheduleProcessing();
}

function setFieldLists(allFields, defaultFields) {
  const incomingAll = Array.from(allFields || FALLBACK_ALL_FIELDS);
  const orderSet = new Set(CUSTOM_FIELD_ORDER);
  const ordered = CUSTOM_FIELD_ORDER.filter((f) => incomingAll.includes(f));
  const remaining = incomingAll.filter((f) => !orderSet.has(f)).sort();
  state.allFields = ordered.concat(remaining);
  state.defaultFields = Array.from(defaultFields || FALLBACK_DEFAULT_FIELDS);
  renderFieldCheckboxes();
}

function renderFieldCheckboxes() {
  ui.fields.innerHTML = '';
  const defaultSet = new Set(state.defaultFields);
  state.allFields.forEach((field) => {
    const label = document.createElement('label');
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.name = 'keep-field';
    checkbox.value = field;
    checkbox.checked = defaultSet.has(field);
    label.appendChild(checkbox);
    label.appendChild(document.createTextNode(field));
    ui.fields.appendChild(label);
  });
}

function getSelectedFields() {
  return Array.from(document.querySelectorAll('input[name="keep-field"]:checked')).map((el) => el.value);
}

function setFields(selection) {
  document.querySelectorAll('input[name="keep-field"]').forEach((box) => {
    box.checked = selection.has(box.value);
  });
}

function safeParseJSON(payload, fallback = null) {
  try {
    return payload ? JSON.parse(payload) : fallback;
  } catch {
    return fallback;
  }
}

function resetMergeState(closeModal = false) {
  state.duplicates = [];
  state.mergePlan = null;
  refreshMergeButton();
  if (closeModal) closeMergeModal();
}

function refreshMergeButton() {
  const btn = ui.buttons?.resolveDuplicates;
  if (!btn) return;
  const hasDupes = Array.isArray(state.duplicates) && state.duplicates.length > 0;
  btn.disabled = !(ui.autoMerge.checked && hasDupes);
}

function getTitleMode() {
  return document.querySelector('input[name="title-mode"]:checked')?.value || 'title';
}

function getVenueMode() {
  return document.querySelector('input[name="venue-mode"]:checked')?.value || 'format';
}

function getVenueStandardizeMode() {
  if (!ui.venueNormalize || !ui.venueNormalize.checked) return 'none';
  return document.querySelector('input[name="venue-standardize"]:checked')?.value || 'full';
}

function updateVenueNormalizeState() {
  const enabled = ui.venueNormalize?.checked;
  document.querySelectorAll('input[name="venue-standardize"]').forEach((el) => {
    el.disabled = !enabled;
  });
  if (!enabled) {
    // Reset to default radio when disabled to keep state predictable.
    const defaultRadio = document.querySelector('input[name="venue-standardize"][value="full"]');
    if (defaultRadio) defaultRadio.checked = true;
  }
}

function scheduleProcessing() {
  if (state.processingTimer) clearTimeout(state.processingTimer);
  state.processingTimer = setTimeout(() => {
    state.processingTimer = null;
    processNow();
  }, 400);
}

async function processNow() {
  const rawText = ui.input.value;
  if (!rawText.trim()) {
    ui.output.value = '';
    ui.log.textContent = '';
    setDiffVisible(false);
    renderDiff('');
    ui.stats.textContent = 'Not processed yet';
    resetMergeState();
    clearHighlights();
    setStatus('Paste or upload BibTeX first.', 'warn');
    return;
  }

  setStatus('Processing…');
  let pyResult = null;
  try {
    const pyodide = await state.pyodidePromise;
    const keepFields = getSelectedFields();

    pyodide.globals.set('raw_text_js', rawText);
    pyodide.globals.set('keep_fields_js', keepFields);
    pyodide.globals.set('title_mode_js', getTitleMode());
    pyodide.globals.set('venue_mode_js', getVenueMode());
    pyodide.globals.set('venue_standardize_mode_js', getVenueStandardizeMode());
    pyodide.globals.set('auto_merge_js', ui.autoMerge.checked);
    pyodide.globals.set('merge_plan_json', state.mergePlan === null ? null : JSON.stringify(state.mergePlan));
    pyodide.globals.set('want_diff_js', Boolean(ui.showDiff?.checked));

    pyResult = pyodide.runPython(`
import json
import difflib
from process_core import process_bibtex_content, DEFAULT_KEEP_FIELDS
keep_fields = set(keep_fields_js) if keep_fields_js else set(DEFAULT_KEEP_FIELDS)
merge_plan = json.loads(merge_plan_json) if merge_plan_json else None
processed_output, log_obj = process_bibtex_content(
    raw_text_js,
    keep_fields,
    title_mode=title_mode_js,
    venue_mode=venue_mode_js,
    venue_standardize_mode=venue_standardize_mode_js,
    auto_merge_duplicates=bool(auto_merge_js),
    merge_plan=merge_plan,
)
report = log_obj.generate_report()
changed_keys = sorted({c['entry'] for c in log_obj.changes})
diff_text = ''
if want_diff_js:
    diff_text = '\\n'.join(difflib.unified_diff(
        raw_text_js.splitlines(),
        processed_output.splitlines(),
        fromfile='input',
        tofile='cleaned',
        n=200,
        lineterm='',
    ))
(processed_output, report, log_obj.total_entries, log_obj.modified_entries, changed_keys, json.dumps(log_obj.duplicates_pre_merge), diff_text)
`);

    const [processed, report, totalEntries, modifiedEntries, changedKeys, duplicatesJson, diffText] = pyResult.toJs({ create_proxies: false });
    state.lastProcessed = processed;
    state.modifiedKeys = new Set(changedKeys || []);
    state.duplicates = safeParseJSON(duplicatesJson, []);
    refreshMergeButton();
    renderOutput(rawText, processed, state.modifiedKeys);
    ui.log.textContent = report;
    setDiffVisible(Boolean(ui.showDiff?.checked));
    renderDiff(diffText || '');
    ui.stats.textContent = totalEntries
      ? `Total ${totalEntries}, modified ${modifiedEntries}`
      : 'No valid entries found';
    setStatus('Auto-processed.', 'success');
  } catch (err) {
    console.error(err);
    setStatus(`Processing failed: ${err.message || err}`, 'warn');
    state.duplicates = [];
    refreshMergeButton();
    const showDiff = Boolean(ui.showDiff?.checked);
    setDiffVisible(showDiff);
    if (showDiff && ui.diff) {
      ui.diff.innerHTML = '<span class="diff-meta">Diff unavailable (processing failed).</span>';
    } else {
      renderDiff('');
    }
  } finally {
    if (pyResult && pyResult.destroy) pyResult.destroy();
  }
}

function handleFileUpload(event) {
  const file = event.target.files?.[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    resetMergeState(true);
    ui.input.value = e.target.result;
    setStatus(`Loaded file: ${file.name}. Processing…`);
    scheduleProcessing('upload');
  };
  reader.onerror = () => setStatus('File read failed.', 'warn');
  reader.readAsText(file);
}

async function copyText(text, message) {
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    setStatus(message, 'success');
  } catch {
    setStatus('Copy failed, please select text manually.', 'warn');
  }
}

function downloadText(filename, text) {
  const blob = new Blob([text], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function openMergeModal() {
  if (!ui.autoMerge.checked) {
    setStatus('Enable auto-merge to choose masters.', 'warn');
    return;
  }
  if (!state.duplicates || !state.duplicates.length) {
    setStatus('No duplicates detected to resolve.', 'warn');
    return;
  }
  renderMergeModal(state.duplicates);
  if (ui.mergeModal) {
    ui.mergeModal.classList.remove('hidden');
    ui.mergeModal.setAttribute('aria-hidden', 'false');
  }
}

function closeMergeModal() {
  if (ui.mergeModal) {
    ui.mergeModal.classList.add('hidden');
    ui.mergeModal.setAttribute('aria-hidden', 'true');
  }
}

function renderMergeModal(groups) {
  if (!ui.mergeGroups) return;
  ui.mergeGroups.innerHTML = '';
  if (!groups || !groups.length) {
    const msg = document.createElement('p');
    msg.className = 'hint';
    msg.textContent = 'No duplicate groups were found.';
    ui.mergeGroups.appendChild(msg);
    return;
  }

  groups.forEach((group, idx) => {
    const card = document.createElement('div');
    card.className = 'merge-card';
    card.dataset.groupIndex = `${idx}`;
    const header = renderMergeCardHeader(group, idx);
    card.appendChild(header);
    card.appendChild(renderMergeOptions(group, idx));
    const toggle = header.querySelector('input[type="checkbox"][data-merge-toggle]');
    if (toggle) {
      const updateCardState = () => setMergeGroupEnabled(card, toggle.checked);
      toggle.addEventListener('change', updateCardState);
      updateCardState();
    }
    ui.mergeGroups.appendChild(card);
  });
}

function renderMergeCardHeader(group, idx) {
  const header = document.createElement('div');
  header.className = 'merge-card__header';

  const titleWrap = document.createElement('div');
  const title = document.createElement('h4');
  title.className = 'merge-card__title';
  const primaryTitle =
    group.entries?.[0]?.title || group.group_norm || group.normalized_title || 'Duplicate group';
  title.textContent = primaryTitle;
  const subtitle = document.createElement('p');
  subtitle.className = 'merge-card__subtitle';
  const keys = (group.entries || []).map((e) => e.key).filter(Boolean);
  subtitle.textContent = `Entries: ${keys.length} • Keys: ${keys.join(', ')}`;
  titleWrap.appendChild(title);
  titleWrap.appendChild(subtitle);

  const toggleLabel = document.createElement('label');
  toggleLabel.className = 'toggle';
  const checkbox = document.createElement('input');
  checkbox.type = 'checkbox';
  checkbox.checked = getExistingMergeState(group)?.merge !== false;
  checkbox.dataset.mergeToggle = `${idx}`;
  const span = document.createElement('span');
  span.textContent = 'Merge this group';
  toggleLabel.appendChild(checkbox);
  toggleLabel.appendChild(span);

  header.appendChild(titleWrap);
  header.appendChild(toggleLabel);
  return header;
}

function renderMergeOptions(group, idx) {
  const optionsWrap = document.createElement('div');
  optionsWrap.className = 'merge-options';
  const existing = getExistingMergeState(group);
  const defaultMaster = existing?.master || group.entries?.[0]?.key;
  const defaultMasterIdx = existing?.master_idx ?? group.entries?.[0]?.occurrence ?? 0;

  (group.entries || []).forEach((entry, entryIdx) => {
    const option = document.createElement('div');
    option.className = 'merge-option';

    const radio = document.createElement('input');
    radio.type = 'radio';
    radio.name = `merge-master-${idx}`;
    radio.value = entry.key;
    radio.dataset.occurrence = entry.occurrence ?? entryIdx;
    radio.dataset.order = entry.order ?? entryIdx;
    const radioId = `merge-${idx}-${entryIdx}`;
    radio.id = radioId;
    radio.checked =
      entry.key === defaultMaster
        ? (entry.occurrence ?? 0) === defaultMasterIdx
        : (!defaultMaster && entryIdx === 0);

    const label = document.createElement('label');
    label.htmlFor = radioId;
    const key = document.createElement('span');
    key.className = 'entry-key';
    key.textContent = entry.key || '(no key)';
    const title = document.createElement('span');
    title.className = 'entry-title';
    title.textContent = entry.title || '(no title)';
    const body = document.createElement('pre');
    body.className = 'entry-body';
    body.textContent = entry.entry || '(no entry text)';
    label.appendChild(key);
    label.appendChild(title);
    label.appendChild(body);

    option.appendChild(radio);
    option.appendChild(label);
    optionsWrap.appendChild(option);
  });

  return optionsWrap;
}

function setMergeGroupEnabled(card, enabled) {
  if (!card) return;
  card.classList.toggle('merge-disabled', !enabled);
  card.querySelectorAll('input[type="radio"]').forEach((radio) => {
    radio.disabled = !enabled;
  });
}

function getExistingMergeState(group) {
  if (!state.mergePlan || !Array.isArray(state.mergePlan)) return null;
  const groupId = group.group_norm || group.normalized_title;
  return state.mergePlan.find((item) => item.group_norm === groupId) || null;
}

function gatherMergePlan() {
  if (!ui.mergeGroups) return null;
  const plan = [];
  const groups = state.duplicates || [];

  groups.forEach((group, idx) => {
    const card = ui.mergeGroups.querySelector(`[data-group-index="${idx}"]`);
    if (!card) return;
    const mergeToggle = card.querySelector('input[type="checkbox"][data-merge-toggle]');
    const shouldMerge = mergeToggle ? mergeToggle.checked : true;
    const masterInput = card.querySelector(`input[name="merge-master-${idx}"]:checked`);
    const master = masterInput?.value;
    const masterIdx = masterInput
      ? parseInt(masterInput.dataset.occurrence || masterInput.dataset.index || '0', 10) || 0
      : 0;
    const masterOrder = masterInput
      ? parseInt(masterInput.dataset.order || masterInput.dataset.index || '0', 10) || 0
      : 0;
    const mergeKeys = Array.from(
      new Set((group.entries || []).map((e) => e.key).filter(Boolean))
    );
    const mergeItems = (group.entries || [])
      .filter((e) => {
        const occ = e.occurrence ?? 0;
        return !(e.key === master && occ === masterIdx);
      })
      .map((e) => ({
        key: e.key || '',
        occurrence: e.occurrence ?? 0,
        order: e.order ?? 0,
      }));
    plan.push({
      group_norm: group.group_norm || group.normalized_title || '',
      merge: shouldMerge,
      master: master || (group.entries?.[0]?.key || ''),
      master_idx: masterIdx,
      master_order: masterOrder,
      merge_keys: mergeKeys,
      merge_items: mergeItems,
    });
  });

  return plan;
}

function applyMergeSelections() {
  const plan = gatherMergePlan();
  state.mergePlan = plan;
  closeMergeModal();
  setStatus('Applying merge choices…');
  processNow();
}

function getOutputForActions() {
  const text = ui.output.value || state.lastProcessed || '';
  if (!text.trim()) return text;
  return ui.trimBlankLines?.checked ? stripExtraBlankLines(text) : text;
}

function stripExtraBlankLines(text) {
  const lines = text.split('\n');
  const kept = [];
  let blankRun = 0;

  for (const line of lines) {
    const isBlank = !line.trim();
    if (isBlank) {
      blankRun += 1;
      if (blankRun > 1) continue;
    } else {
      blankRun = 0;
    }
    kept.push(line);
  }

  while (kept.length && !kept[kept.length - 1].trim()) {
    kept.pop();
  }

  return kept.join('\n');
}

function splitEntries(text) {
  const matches = Array.from(text.matchAll(/@\w+\s*\{/g));
  const entries = [];
  if (!matches.length) return entries;
  for (let i = 0; i < matches.length; i++) {
    const start = matches[i].index;
    const end = i + 1 < matches.length ? matches[i + 1].index : text.length;
    entries.push(text.slice(start, end));
  }
  return entries;
}

function extractKey(entry) {
  const m = entry.match(/@\w+\s*\{\s*([^,\s]+)/);
  return m ? m[1] : null;
}

function computeLineCounts(rawInput) {
  const matches = Array.from(rawInput.matchAll(/@\w+\s*\{/g));
  const counts = [];
  if (!matches.length) return { leading: 0, counts };

  const leadingText = rawInput.slice(0, matches[0].index);
  const leading = (leadingText.match(/\n/g) || []).length;

  for (let i = 0; i < matches.length; i++) {
    const start = matches[i].index;
    const end = i + 1 < matches.length ? matches[i + 1].index : rawInput.length;
    const block = rawInput.slice(start, end);
    counts.push(block.split('\n').length);
  }

  return { leading, counts };
}

function padEntry(entryText, targetLines) {
  const lineCount = entryText.split(/\r?\n/).length;
  const diff = targetLines - lineCount;
  if (diff <= 0) return entryText;
  return entryText + '\n'.repeat(diff);
}

function blankLines(lineCount) {
  if (!lineCount || lineCount <= 0) return '';
  return Array(lineCount).fill('').join('\n');
}

function ensureLeadingBlankLines(rawInput, processedText) {
  const { leading } = computeLineCounts(rawInput);
  if (!leading) return processedText;
  const outputLeading = (processedText.match(/^\n+/) || [''])[0].length;
  if (outputLeading >= leading) return processedText;
  return '\n'.repeat(leading - outputLeading) + processedText;
}

function buildPaddedOutput(rawInput, processedText) {
  const processedEntries = splitEntries(processedText);
  if (!processedEntries.length) return processedText;
  const { leading, counts } = computeLineCounts(rawInput);
  const rawTotalLines = rawInput.split('\n').length;

  const parts = [];
  if (leading > 0) {
    parts.push('\n'.repeat(leading));
  }

  const total = Math.max(counts.length, processedEntries.length);
  for (let i = 0; i < total; i++) {
    const targetLines = counts[i] || 0;
    const entry = processedEntries[i];
    if (entry === undefined) {
      if (targetLines > 0) parts.push(blankLines(targetLines));
      continue;
    }
    const padded = padEntry(entry, targetLines || entry.split(/\r?\n/).length);
    parts.push(padded);
  }

  const combined = parts.join('');
  const targetTotalLines = rawTotalLines;
  return normalizeLineCount(combined, targetTotalLines);
}

function renderOutput(rawInput, processedText, modifiedSet) {
  const rawCounts = computeLineCounts(rawInput).counts;
  const processedEntries = splitEntries(processedText);
  // If entry counts match, we can safely pad to align; otherwise keep backend spacing (preserves master position)
  const textForDisplay = ensureLeadingBlankLines(
    rawInput,
    processedEntries.length === rawCounts.length
      ? buildPaddedOutput(rawInput, processedText)
      : ensureLeadingBlankLines(rawInput, processedText)
  );
  ui.output.value = textForDisplay;
  // After content update, align scroll positions if sync is enabled
  syncScroll(ui.input, ui.output);
  renderHighlights(rawInput, textForDisplay);
}

function renderHighlights(rawInput, processedText) {
  setOverlayHTML('input', rawInput);
  setOverlayHTML('output', processedText);
}

function setOverlayHTML(side, text) {
  const overlay = ui.highlights[side];
  const editor = side === 'input' ? ui.input : ui.output;
  if (!overlay || !editor) return;

  if (!text) {
    overlay.innerHTML = '';
    overlay.style.height = '0px';
    overlay.style.width = '0px';
    return;
  }

  // Insert a guard span so the browser doesn't drop the first leading newline
  overlay.innerHTML = `<pre class="overlay-text ${side}"><span class="pre-guard"></span>${buildOverlayHTML(text, state.modifiedKeys, side)}</pre>`;
  overlay.style.height = `${editor.scrollHeight}px`;
  overlay.style.width = `${editor.scrollWidth}px`;
  updateOverlayScroll(side);
}

function buildOverlayHTML(text, modifiedSet, side) {
  const matches = Array.from(text.matchAll(/@\w+\s*\{/g));
  if (!matches.length) return escapeHtml(text);
  const parts = [];
  let cursor = 0;

  for (let i = 0; i < matches.length; i++) {
    const start = matches[i].index;
    const end = i + 1 < matches.length ? matches[i + 1].index : text.length;
    if (start > cursor) {
      parts.push(escapeHtml(text.slice(cursor, start)));
    }
    const entry = text.slice(start, end);
    const key = extractKey(entry) || `entry_${i + 1}`;
    const cls = modifiedSet.has(key) ? `changed-text ${side}` : 'normal';
    const highlighted = highlightTitleColon(entry, side);
    parts.push(`<span class="${cls}">${highlighted}</span>`);
    cursor = end;
  }

  if (cursor < text.length) {
    parts.push(escapeHtml(text.slice(cursor)));
  }

  return parts.join('');
}

function highlightTitleColon(escapedEntry, side) {
  const spanStart = `<span class="colon-warn ${side}">`;
  const spanEnd = '</span>';
  const text = escapedEntry;

  const titleMatch = /title\s*=\s*([{"])/i.exec(text);
  if (!titleMatch) return escapeHtml(text);

  const openIdx = titleMatch.index + titleMatch[0].length - 1;
  const openChar = titleMatch[1];
  let valueStart = openIdx + 1;
  let valueEnd = -1;

  if (openChar === '{') {
    let depth = 1;
    for (let i = valueStart; i < text.length; i++) {
      if (text[i] === '{') depth++;
      else if (text[i] === '}') depth--;
      if (depth === 0) {
        valueEnd = i;
        break;
      }
    }
  } else {
    // quote case
    for (let i = valueStart; i < text.length; i++) {
      if (text[i] === '"') {
        valueEnd = i;
        break;
      }
    }
  }

  if (valueEnd === -1) return escapeHtml(text);

  const value = text.slice(valueStart, valueEnd);
  const colonIdx = value.indexOf(':');
  if (colonIdx === -1) return escapeHtml(text);

  const before = value.slice(0, colonIdx);
  const after = value.slice(colonIdx + 1);
  if (!before.trim()) return escapeHtml(text);

  const prefix = escapeHtml(text.slice(0, valueStart));
  const suffix = escapeHtml(text.slice(valueEnd));
  const highlightedValue = `${spanStart}${escapeHtml(before)}${spanEnd}:${escapeHtml(after)}`;

  return `${prefix}${highlightedValue}${suffix}`;
}

function clearHighlights() {
  ['input', 'output'].forEach((side) => {
    const overlay = ui.highlights[side];
    if (!overlay) return;
    overlay.innerHTML = '';
    overlay.style.height = '0px';
  });
}

function normalizeLineCount(text, targetLines) {
  if (!targetLines || targetLines <= 0) return text;
  const lines = text.split('\n');
  if (lines.length === targetLines) return text;
  if (lines.length > targetLines) {
    // Preserve trailing newline expectation: if targetLines < lines.length, cut exactly
    return lines.slice(0, targetLines).join('\n');
  }
  // lines.length < targetLines
  return text + '\n'.repeat(targetLines - lines.length);
}

function updateOverlayScroll(side) {
  const overlay = ui.highlights[side];
  if (!overlay) return;
  const editor = side === 'input' ? ui.input : ui.output;
  overlay.style.transform = `translate(${-editor.scrollLeft}px, ${-editor.scrollTop}px)`;
}

function syncScroll(source, target) {
  if (!ui.syncScroll.checked) return;
  const sourceMax = source.scrollHeight - source.clientHeight;
  const targetMax = target.scrollHeight - target.clientHeight;
  if (sourceMax <= 0 || targetMax <= 0) return;
  const ratio = source.scrollTop / sourceMax;
  state.isSyncing = true;
  target.scrollTop = ratio * targetMax;
  state.isSyncing = false;
  updateOverlayScroll(source === ui.input ? 'input' : 'output');
  updateOverlayScroll(target === ui.input ? 'input' : 'output');
}

function onInputScroll() {
  if (state.isSyncing) return;
  updateOverlayScroll('input');
  syncScroll(ui.input, ui.output);
  updateOverlayScroll('output');
}

function onOutputScroll() {
  if (state.isSyncing) return;
  updateOverlayScroll('output');
  syncScroll(ui.output, ui.input);
  updateOverlayScroll('input');
}
