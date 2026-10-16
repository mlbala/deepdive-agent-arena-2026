// Evidence Lens: a claim on the left, its cited source on the right with the exact verified span in focus.
'use strict';
import { $, api, esc, store, toast } from './core.js';

const SHIELD_TEXT = '[removed by DeepDive content shield: suspected prompt-injection text]';
const L = { runId: null, claims: [], index: -1, sourceId: null, filter: 'all', fallback: null, sources: {} };

async function loadClaims(runId) {
  const claims = await api(`/api/runs/${runId}/claims`);
  return claims;
}
async function loadSource(runId, sid) {
  const key = `${runId}:${sid}`;
  if (!L.sources[key]) L.sources[key] = await api(`/api/runs/${runId}/sources/${encodeURIComponent(sid)}`);
  return L.sources[key];
}

function publicationStatus(claim) {
  if (claim.included_in_report === undefined) return '';
  if (claim.included_in_report) return '<span class="text-emerald-300">● Published in the report</span>';
  const later = L.claims.filter((c) => c.sub_question_id === claim.sub_question_id && c.iteration > claim.iteration);
  if (later.length) {
    const it = Math.max(...later.map((c) => c.iteration));
    return `<span class="text-slate-400">● Superseded by the iteration-${it} re-draft of ${esc(claim.sub_question_id)}</span>`;
  }
  return claim.verdict === 'SUPPORTED'
    ? '<span class="text-amber-300">● Withheld: outweighed by a stronger source or below the authority floor</span>'
    : '<span class="text-amber-300">● Stopped before publication</span>';
}

function visibleClaims() {
  return L.filter === 'stopped' ? L.claims.filter((c) => !c.included_in_report) : L.claims;
}

export async function openLens({ runId = store.runId, claimId = null, sourceId = null, fallback = null } = {}) {
  if (!runId) return;
  try {
    if (L.runId !== runId) L.sources = {};
    L.runId = runId;
    L.claims = await loadClaims(runId);
    L.filter = 'all';
    L.fallback = null;
    L.index = claimId ? L.claims.findIndex((c) => c.claim_id === claimId) : -1;
    if (claimId && L.index < 0) L.fallback = fallback || { claim_id: claimId };
    L.sourceId = sourceId;
    await render();
    $('#lens').classList.remove('hidden');
    $('#lensBody').focus?.();
  } catch (e) { toast(e.message); }
}

export function closeLens() { $('#lens').classList.add('hidden'); }
const isOpen = () => !$('#lens').classList.contains('hidden');

function highlight(text, ranges, focus) {
  const all = [...ranges];
  if (focus) for (let i = all.length - 1; i >= 0; i--) if (all[i].start < focus.end && all[i].end > focus.start) all.splice(i, 1);
  if (focus) all.push({ ...focus, focus: true });
  all.sort((a, b) => a.start - b.start);
  let out = ''; let cursor = 0;
  for (const r of all) {
    if (r.start < cursor || r.start === null || r.end === null) continue;
    out += esc(text.slice(cursor, r.start));
    out += `<mark class="q-${esc(r.verdict)}${r.focus ? ' q-focus' : ''}" ${r.focus ? 'id="lensFocus"' : ''} title="${esc(r.claim_id || '')}">${esc(text.slice(r.start, r.end))}</mark>`;
    cursor = r.end;
  }
  out += esc(text.slice(cursor));
  return out.replaceAll(esc(SHIELD_TEXT), '<span class="shield-mark">removed by content shield: suspected prompt-injection text</span>');
}

async function render() {
  const list = visibleClaims();
  const claim = L.index >= 0 ? L.claims[L.index] : L.fallback;
  const sid = L.sourceId || claim?.evidence_source_id || claim?.cited_source_ids?.[0] || null;
  let src = null;
  if (sid) { try { src = await loadSource(L.runId, sid); } catch (_) { src = null; } }
  const pos = claim && L.index >= 0 ? list.findIndex((c) => c.claim_id === claim.claim_id) : -1;
  const focus = claim && src && claim.evidence_source_id === src.source_id && claim.evidence_char_start !== null && claim.evidence_char_start !== undefined
    ? { start: claim.evidence_char_start, end: claim.evidence_char_end, verdict: claim.verdict, claim_id: claim.claim_id } : null;
  const others = (src?.verified_quotes || []).filter((q) => !claim || q.claim_id !== claim.claim_id);
  const verdict = claim?.verdict;
  const why = claim ? (claim.downgrade_reason || claim.rationale || (verdict === 'SUPPORTED' ? 'The cited source states this claim.' : '')) : '';

  $('#lensBody').innerHTML = `
    <div class="lens-head">
      <div class="flex items-center gap-2"><span class="brand-mark" style="width:26px;height:26px;border-radius:8px"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="10.5" cy="10.5" r="6"/><path d="M20 20l-5-5"/></svg></span>
        <div><div class="text-[15px] font-semibold tracking-tight">Evidence Lens</div><div class="text-[11.5px] text-slate-500">Each claim next to the exact text it rests on</div></div></div>
      ${L.claims.length ? `<div class="seg ml-auto hidden sm:flex" style="min-width:200px"><button data-lens-filter="all" class="${L.filter === 'all' ? 'on' : ''}">All claims</button><button data-lens-filter="stopped" class="${L.filter === 'stopped' ? 'on' : ''}">Stopped</button></div>` : '<span class="ml-auto"></span>'}
      ${pos >= 0 ? `<span class="font-mono text-[11px] text-slate-500">${pos + 1} / ${list.length}</span>` : ''}
      <button class="icon-btn" data-lens-nav="-1" aria-label="Previous claim" ${pos <= 0 ? 'disabled' : ''}>‹</button>
      <button class="icon-btn" data-lens-nav="1" aria-label="Next claim" ${pos < 0 || pos >= list.length - 1 ? 'disabled' : ''}>›</button>
      <button class="icon-btn" data-lens-close aria-label="Close">✕</button>
    </div>
    <div class="lens-grid">
      <div class="lens-left scroll-thin">
        ${claim ? `
          <div class="flex flex-wrap items-center gap-2">${verdict ? `<span class="lens-verdict ${esc(verdict)}">${verdict === 'SUPPORTED' ? '✓' : verdict === 'CONTRADICTED' ? '✕' : '⚠'} ${esc(verdict.toLowerCase())}</span>` : ''}
            <span class="font-mono text-[11px] text-slate-500">${esc(claim.claim_id)}${claim.iteration ? ` · iteration ${esc(claim.iteration)}` : ''}${claim.sub_question_id ? ` · ${esc(claim.sub_question_id)}` : ''}</span></div>
          <p class="lens-claim mt-3">${esc(claim.claim_text || claim.claim || 'Claim details are sealed when the run finishes.')}</p>
          ${why ? `<div class="lens-why"><div class="panel-h mb-1">Why</div>${esc(why)}</div>` : ''}
          ${claim.confidence !== undefined ? `<div class="mt-4"><div class="flex text-[12px] text-slate-400"><span>Verifier confidence</span><b class="ml-auto text-slate-100">${Number(claim.confidence).toFixed(2)}</b></div><div class="dgauge"><div class="dgauge-fill" style="width:${Math.round(Number(claim.confidence) * 100)}%;background:rgb(var(--accent))"></div></div></div>` : ''}
          <div class="mt-4 text-[12px] text-slate-400">Cited sources</div>
          <div class="mt-1.5 flex flex-wrap gap-1.5">${(claim.cited_source_ids || []).map((s) => `<button class="chip" data-lens-source="${esc(s)}">${esc(s)}</button>`).join('') || '<span class="text-[12px] text-rose-300">none: an uncited claim is unsupported by definition</span>'}</div>
          <div class="mt-4 flex items-center gap-2 text-[12.5px]">${publicationStatus(claim)}</div>
          ${claim.quote_verified === false && verdict !== 'SUPPORTED' ? '<p class="mt-3 text-[12.5px] leading-relaxed text-slate-400">No verbatim span in the cited source supports this claim. That is exactly why DeepDive removed it.</p>' : ''}`
        : `<p class="lens-claim">Source view</p><p class="mt-2 text-[13px] text-slate-400">Verified quotes from this source are highlighted by verdict. Open a claim from the Claim Matrix or the Verify stage to put its exact span in focus.</p>`}
      </div>
      <div class="lens-right">
        ${src ? `<div class="lens-source-head"><div class="flex flex-wrap items-center gap-2"><span class="chip">${esc(src.source_id)}</span><span class="text-[14px] font-semibold tracking-tight">${esc(src.title)}</span></div>
          <div class="mt-1 flex flex-wrap items-center gap-2 text-[12px] text-slate-400"><span>${esc((src.authority_tier || '').replace('_', ' '))} · authority ${Number(src.authority_score).toFixed(2)}</span><span>·</span><span>${esc(src.retrieval_tier)}</span><span>·</span>
          <a class="link max-w-[420px] truncate" href="${esc(src.url.startsWith('http') ? src.url : '#')}" target="_blank" rel="noopener noreferrer nofollow">${esc(src.url)}</a></div>
          <div class="mt-1 text-[11.5px] text-slate-500">${focus ? 'The highlighted span is what the Verifier quoted. Its exact character offsets are stored in the audit trail.' : `${others.length} verified quote(s) highlighted by verdict`}</div></div>
          <div class="lens-doc scroll-thin">${highlight(src.content || '', others, focus)}</div>`
        : '<div class="grid h-full place-items-center p-10 text-center text-[14px] text-slate-500">No source to show for this claim.</div>'}
      </div>
    </div>`;
  requestAnimationFrame(() => document.getElementById('lensFocus')?.scrollIntoView({ block: 'center', behavior: 'smooth' }));
}

export function initLens() {
  $('#lens').addEventListener('click', async (e) => {
    if (e.target.id === 'lens' || e.target.closest('[data-lens-close]')) { closeLens(); return; }
    const nav = e.target.closest('[data-lens-nav]');
    if (nav && !nav.disabled) { move(Number(nav.dataset.lensNav)); return; }
    const s = e.target.closest('[data-lens-source]');
    if (s) { L.sourceId = s.dataset.lensSource; await render(); return; }
    const f = e.target.closest('[data-lens-filter]');
    if (f) { L.filter = f.dataset.lensFilter; const list = visibleClaims(); if (list.length) { L.index = L.claims.indexOf(list[0]); L.sourceId = null; } await render(); }
  });
  document.addEventListener('keydown', (e) => {
    if (!isOpen()) return;
    if (e.key === 'Escape') { e.stopPropagation(); closeLens(); }
    if (e.key === 'ArrowRight') { e.preventDefault(); move(1); }
    if (e.key === 'ArrowLeft') { e.preventDefault(); move(-1); }
  }, true);
}

function move(delta) {
  const list = visibleClaims();
  const claim = L.claims[L.index];
  const pos = list.indexOf(claim);
  const next = list[pos + delta];
  if (!next) return;
  L.index = L.claims.indexOf(next);
  L.sourceId = null;
  render();
}
