// Command palette (⌘K): one keyboard-first entry point to every action, scenario, run, claim, and source.
'use strict';
import { $, api, emit, esc, store, toast, withKey } from './core.js';
import { openLens } from './lens.js';

const P = { items: [], filtered: [], sel: 0, open: false };

function score(text, query) {
  if (!query) return 1;
  const t = text.toLowerCase(); const q = query.toLowerCase().trim();
  if (!q) return 1;
  const idx = t.indexOf(q);
  if (idx !== -1) return 100 - Math.min(idx, 60) + (idx === 0 ? 20 : 0);
  let ti = 0; let hits = 0;
  for (const ch of q.replace(/\s+/g, '')) { ti = t.indexOf(ch, ti); if (ti === -1) return 0; hits++; ti++; }
  return hits ? 10 : 0;
}

async function buildItems(scenarios) {
  const runId = store.runId;
  const items = [
    { group: 'Navigate', icon: '◎', label: 'Live Demo', sub: '1', run: () => emit('navigate', { view: 'demo' }) },
    { group: 'Navigate', icon: '⌁', label: 'How it works: architecture explorer', sub: '2', run: () => emit('navigate', { view: 'explorer' }) },
    { group: 'Navigate', icon: '▤', label: 'Console', sub: '3', run: () => emit('navigate', { view: 'console' }) },
    { group: 'Navigate', icon: '◔', label: 'Insights for the current run', sub: 'charts', run: () => emit('navigate', { view: 'console', params: { tab: 'insights' } }) },
    { group: 'Navigate', icon: '💬', label: 'Agent group chat for the current run', sub: 'every hand-off between agents', run: () => emit('navigate', { view: 'console', params: { tab: 'chat' } }) },
    { group: 'Architecture', icon: '☰', label: 'Workflow: how a question becomes a verified report', sub: '11 steps', run: () => emit('navigate', { view: 'explorer', params: { xview: 'workflow' } }) },
    { group: 'Architecture', icon: '◇', label: 'Agents & their real system prompts', sub: 'prompts · tools · latency', run: () => emit('navigate', { view: 'explorer', params: { xview: 'agents' } }) },
    { group: 'Architecture', icon: '⌗', label: 'LangGraph state channels & data contracts', sub: 'reducers · schemas', run: () => emit('navigate', { view: 'explorer', params: { xview: 'state' } }) },
    { group: 'Architecture', icon: '▭', label: 'Run timeline (Gantt) of the current run', sub: 'node spans · Mesh calls', run: () => emit('navigate', { view: 'explorer', params: { xview: 'timeline' } }) },
    ...scenarios.map((s, i) => ({ group: 'Run a scenario', icon: String(i + 1), label: `${s.title}`, sub: s.tag, keywords: s.q, run: () => emit('demo-scenario', { index: i, run: true }) })),
    { group: 'Actions', icon: '◐', label: 'Toggle appearance (auto, light, dark)', sub: 'T', run: () => emit('theme-cycle') },
    { group: 'Actions', icon: '?', label: 'Keyboard shortcuts', sub: '?', run: () => emit('shortcuts') },
  ];
  if (runId) {
    items.push(
      { group: 'This run', icon: '⧉', label: 'Copy share link', sub: runId, run: () => copyLink() },
      { group: 'This run', icon: '⎙', label: 'Export PDF report', sub: 'PDF', run: () => window.open(withKey(`/api/runs/${runId}/report.pdf`), '_blank', 'noopener') },
      { group: 'This run', icon: '⇩', label: 'Export audit bundle (JSON)', sub: 'run, claims, sources, audit chain', run: () => { window.location.href = withKey(`/api/runs/${runId}/bundle`); } },
      { group: 'This run', icon: '⛓', label: 'Verify the audit hash chain', sub: 'tamper check', run: verifyChain },
      { group: 'This run', icon: '◉', label: 'Open the Evidence Lens', sub: 'claims ↔ sources', run: () => openLens({ runId }) },
    );
    try {
      const [claims, sources] = await Promise.all([api(`/api/runs/${runId}/claims`), api(`/api/runs/${runId}/sources`)]);
      claims.forEach((c) => items.push({ group: 'Claims', icon: c.verdict === 'SUPPORTED' ? '✓' : c.verdict === 'CONTRADICTED' ? '✕' : '⚠', label: c.claim_text, sub: `${c.claim_id}${c.included_in_report ? '' : ' · stopped'}`, run: () => openLens({ runId, claimId: c.claim_id }) }));
      if (claims.length) items.push({ group: 'This run', icon: '⧉', label: 'Copy link to the first stopped claim', sub: 'claim + evidence', run: () => { const c = claims.find((x) => !x.included_in_report) || claims[0]; copyLink({ lens: c.claim_id }); } });
      sources.forEach((s) => items.push({ group: 'Sources', icon: s.source_id.replace('SRC-', ''), label: s.title, sub: `${s.domain} · ${Number(s.authority_score).toFixed(2)}`, run: () => openLens({ runId, sourceId: s.source_id }) }));
    } catch (_) { /* run may still be starting */ }
  }
  try {
    const runs = await api('/api/runs?limit=30');
    runs.forEach((r) => items.push({ group: 'Recent runs', icon: r.status === 'blocked' ? '⛔' : r.status === 'completed' ? '●' : '◌', label: r.query, sub: `${r.status} · ${(r.created_at || '').slice(5, 16).replace('T', ' ')}`, run: () => emit('open-run', { runId: r.id }) }));
  } catch (_) { /* auth modal handles 401 */ }
  return items;
}

function render() {
  const list = $('#cmdkList');
  if (!P.filtered.length) { list.innerHTML = '<div class="cmdk-empty">No matches. Try a scenario name, a claim, or a domain.</div>'; return; }
  let html = ''; let group = null;
  P.filtered.forEach((it, i) => {
    if (it.group !== group) { group = it.group; html += `<div class="cmdk-group">${esc(group)}</div>`; }
    html += `<button class="cmdk-item" role="option" data-i="${i}" aria-selected="${i === P.sel}"><span class="cmdk-ico">${esc(it.icon)}</span><span class="cmdk-main">${esc(it.label)}</span><span class="cmdk-sub">${esc(it.sub || '')}</span></button>`;
  });
  list.innerHTML = html;
  list.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: 'nearest' });
}

function filter() {
  const q = $('#cmdkInput').value;
  const scored = P.items.map((it, order) => ({ it, s: Math.max(score(it.label, q), score(`${it.group} ${it.sub || ''} ${it.keywords || ''}`, q) * 0.6), order }))
    .filter((x) => x.s > 0);
  if (q.trim()) scored.sort((a, b) => b.s - a.s || a.order - b.order);
  const perGroup = {};
  P.filtered = scored.map((x) => x.it).filter((it) => (perGroup[it.group] = (perGroup[it.group] || 0) + 1) <= (q.trim() ? 8 : 6));
  if (q.trim()) {
    const order = [...new Set(P.filtered.map((it) => it.group))];
    P.filtered.sort((a, b) => order.indexOf(a.group) - order.indexOf(b.group));
  }
  P.sel = 0;
  render();
}

export async function openPalette(scenarios) {
  P.open = true;
  $('#cmdk').classList.remove('hidden');
  const input = $('#cmdkInput');
  input.value = '';
  input.focus();
  $('#cmdkList').innerHTML = '<div class="cmdk-empty">Loading…</div>';
  P.items = await buildItems(scenarios);
  if (P.open) filter();
}
export function closePalette() { P.open = false; $('#cmdk').classList.add('hidden'); }

function execute(i) {
  const it = P.filtered[i];
  if (!it) return;
  closePalette();
  try { it.run(); } catch (e) { toast(e.message); }
}

async function copyLink(extra = {}) {
  const url = new URL(location.href);
  if (store.runId) url.searchParams.set('run', store.runId);
  for (const [k, v] of Object.entries(extra)) url.searchParams.set(k, v);
  try { await navigator.clipboard.writeText(url.toString()); toast('Share link copied', 'ok'); }
  catch (_) { toast(url.toString(), 'info'); }
}

async function verifyChain() {
  try {
    const v = await api(`/api/runs/${store.runId}/audit/verify`);
    toast(v.valid ? `Audit chain intact · ${v.verified} events verified` : `Audit chain BROKEN at ${v.broken_at}`, v.valid ? 'ok' : 'error');
  } catch (e) { toast(e.message); }
}

export function initPalette() {
  $('#cmdkInput').addEventListener('input', filter);
  $('#cmdkInput').addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); P.sel = Math.min(P.filtered.length - 1, P.sel + 1); render(); }
    if (e.key === 'ArrowUp') { e.preventDefault(); P.sel = Math.max(0, P.sel - 1); render(); }
    if (e.key === 'Enter') { e.preventDefault(); execute(P.sel); }
    if (e.key === 'Escape') { e.preventDefault(); closePalette(); }
  });
  $('#cmdkList').addEventListener('click', (e) => { const b = e.target.closest('[data-i]'); if (b) execute(Number(b.dataset.i)); });
  $('#cmdkList').addEventListener('mousemove', (e) => {
    const b = e.target.closest('[data-i]');
    if (!b || Number(b.dataset.i) === P.sel) return;
    $('#cmdkList').querySelector('[aria-selected="true"]')?.setAttribute('aria-selected', 'false');
    b.setAttribute('aria-selected', 'true');
    P.sel = Number(b.dataset.i);
  });
  $('#cmdk').addEventListener('click', (e) => { if (e.target.id === 'cmdk') closePalette(); });
}
