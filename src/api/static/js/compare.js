// Plain AI vs DeepDive: one plain Mesh answer to the run's question, split into claims and checked
// by DeepDive's own Verifier against this run's evidence pool. Shared by the demo and the console.
import { ApiError, api, esc, fmt, pct, toast } from './core.js';

const C = {}; // runId -> { state: 'checking' | 'none' | 'loading' | 'ready' | 'error', data, error }
const listeners = new Set();

const changed = () => listeners.forEach((fn) => fn());
export function onCompareChange(fn) { listeners.add(fn); }

async function load(runId) {
  C[runId] = { state: 'checking' };
  try { C[runId] = { state: 'ready', data: await api(`/api/runs/${runId}/compare`) }; }
  catch (e) { C[runId] = e instanceof ApiError && e.status === 404 ? { state: 'none' } : { state: 'error', error: e.message }; }
  changed();
}

export async function runCompare(runId) {
  if (C[runId]?.state === 'loading') return;
  C[runId] = { state: 'loading' }; changed();
  try { C[runId] = { state: 'ready', data: await api(`/api/runs/${runId}/compare`, { method: 'POST' }) }; }
  catch (e) { C[runId] = { state: 'none' }; toast(e instanceof ApiError && e.status === 429 ? `Slow down: ${e.message}` : e.message); }
  changed();
}

// a click handler for any container that renders compareHtml
export function handleCompareClick(e) {
  const b = e.target.closest('[data-compare]');
  if (!b) return false;
  runCompare(b.dataset.compare);
  return true;
}

const STATUS = {
  supported: ['ok', (c) => `✓ traced to ${c.evidence_source_id}`],
  contradicted: ['bad', (c) => `✗ contradicted by ${c.evidence_source_id || 'its sources'}`],
  unsupported: ['warn', () => '✗ not stated by the closest sources'],
  not_found: ['warn', () => '· nothing in the evidence pool'],
  unchecked: ['muted', () => '? not checked (Mesh call failed)'],
};

const INTRO = 'One Mesh call answers the same question the way a typical chat assistant would: no search, no citations. DeepDive then splits that answer into claims with its own Verifier rules and checks each one against this run\'s sources, with the same guards it applies to itself.';

export function compareHtml(runId) {
  if (!runId) return '';
  const entry = C[runId];
  if (!entry) { load(runId); return ''; }
  if (entry.state === 'checking') return '';
  const head = '<div class="dsub">Plain AI vs DeepDive</div><h4 class="dcmp-title">Same question, same evidence, same checks</h4>';
  if (entry.state === 'error') return `<section class="dcmp">${head}<p class="dcmp-note">${esc(entry.error)}</p></section>`;
  if (entry.state === 'none' || entry.state === 'loading') {
    const busy = entry.state === 'loading';
    return `<section class="dcmp">${head}<p class="dcmp-note">${INTRO}</p>
      <button class="btn-primary dcmp-btn" data-compare="${esc(runId)}" ${busy ? 'disabled' : ''}>${busy ? 'Asking a plain AI and checking its claims…' : 'Ask a plain AI the same question'}</button></section>`;
  }
  const d = entry.data; const s = d.deepdive || {};
  const rate = d.traceable_rate;
  const reportCheck = s.report_sentences_checked ? `; ${s.report_sentences_supported}/${s.report_sentences_checked} report sentences re-verified` : '';
  const rows = [
    ['Claims you can trace to a verbatim source span', `<b>${rate === null ? '—' : pct(rate)}</b> <i>${d.supported} of ${d.checked} claims</i>`, `<b>100%</b> <i>${s.claims_published} published claims${reportCheck}</i>`],
    ['Claims with a specific figure', `<b>${d.with_figures ?? 0}</b> <i>of ${d.claims.length} claims</i>`, `<b>${s.claims_with_figures ?? 0}</b> <i>of ${s.claims_published} published, each with its quote</i>`],
    ['Claims the sources contradict', `<b>${d.contradicted}</b>`, `<b>0</b> <i>published; ${s.claims_stopped} stopped before publication</i>`],
    ['Citations a reader can check', '<b>none</b>', `<b>${s.sources_cited}</b> <i>sources cited</i>`],
    ['Conflicts between sources disclosed', '<b>none</b>', `<b>${s.contradictions_disclosed}</b>`],
  ];
  const bar = (v, color) => `<div class="dcmp-bar"><i style="width:${Math.round((v ?? 0) * 100)}%;background:${color}"></i></div>`;
  const claims = (d.claims || []).map((c) => {
    const [tone, label] = STATUS[c.status] || ['muted', () => c.status];
    const quote = c.evidence_quote ? `<div class="dcmp-quote">${c.status === 'supported' ? '' : `Closest span in ${esc(c.evidence_source_id || 'the sources')}: `}“${esc(c.evidence_quote)}”</div>` : '';
    return `<div class="dcmp-claim"><span class="dcmp-pill ${tone}">${esc(label(c))}</span><div class="text-[13px] text-slate-200">${esc(c.text)}</div>
      ${c.status !== 'supported' ? `<div class="dcmp-why">${esc(c.reason)}</div>` : ''}${quote}</div>`;
  }).join('');
  return `<section class="dcmp fade-in">${head}
    <div class="dcmp-grid"><div></div><div class="dcmp-col">Plain AI answer${bar(rate, 'rgb(var(--orange))')}</div><div class="dcmp-col">DeepDive report${bar(1, 'rgb(var(--green))')}</div>
      ${rows.map(([k, a, b]) => `<div class="dcmp-k">${k}</div><div class="dcmp-v">${a}</div><div class="dcmp-v">${b}</div>`).join('')}</div>
    <details class="dfold"><summary>The plain answer, claim by claim (${d.claims.length})</summary>
      <p class="dcmp-answer">${esc(d.answer)}</p>${claims}</details>
    <p class="dcmp-note">${esc(d.model)} · ${d.llm_calls} Mesh calls · ${fmt(d.total_tokens)} tokens, including the check. A claim the sources don't state may still be true; it just can't be traced, which is the problem with an uncited answer.${d.llm_reviewed ? '' : ' Some checks failed on the Mesh side and are marked not checked.'}</p>
  </section>`;
}
