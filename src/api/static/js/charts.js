// Run insights: hero numbers + four charts (inline SVG). Colour roles follow the validated reference palette:
// series-1/series-2 for the two comparable measures, the fixed status palette (with icons) for verdicts,
// a single hue for magnitudes. Text always wears text tokens. Every mark has a hover tooltip; every chart
// has a table view.
'use strict';
import { esc, fmt, pct } from './core.js';

const NS = 'http://www.w3.org/2000/svg';

function hbar(x, y, w, h, r = 4) {
  if (w <= 0) return '';
  r = Math.min(r, w / 2, h / 2);
  return `M${x},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y + h - r} Q${x + w},${y + h} ${x + w - r},${y + h} H${x} Z`;
}
function vbar(x, y, w, h, r = 4) {
  if (h <= 0) return '';
  r = Math.min(r, w / 2, h / 2);
  return `M${x},${y + h} V${y + r} Q${x},${y} ${x + r},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y + h} Z`;
}
const tip = (lines) => esc(lines.filter(Boolean).join('\n'));

function card(title, sub, body, legend = '', table = '') {
  return `<section class="viz-card viz"><div class="viz-title">${esc(title)}</div><div class="viz-sub">${esc(sub)}</div>${legend}<div class="mt-3">${body}</div>${table}</section>`;
}
function table(headers, rows) {
  return `<details class="viz-table"><summary>View as table</summary><table><thead><tr>${headers.map((h) => `<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${esc(c)}</td>`).join('')}</tr>`).join('')}</tbody></table></details>`;
}

// -- chart A: grounding & coverage by iteration (grouped bars, one 0-100% axis, threshold refs)
function iterationChart(iters, thresholds) {
  const W = 560; const H = 230; const L = 36; const R = 104; const T = 16; const B = 30;
  const plotH = H - T - B; const plotW = W - L - R;
  const y = (v) => T + plotH * (1 - v);
  const groupW = plotW / Math.max(1, iters.length);
  const barW = Math.min(34, groupW / 3.2);
  let svg = '';
  for (const g of [0, 0.25, 0.5, 0.75, 1]) {
    svg += `<line class="grid-line" x1="${L}" x2="${W - R}" y1="${y(g)}" y2="${y(g)}"/><text class="axis-text" x="${L - 8}" y="${y(g) + 4}" text-anchor="end">${g * 100}%</text>`;
  }
  const refs = [['grounding bar', thresholds.grounding, -5], ['coverage bar', thresholds.coverage, 12]];
  const sameish = Math.abs(thresholds.grounding - thresholds.coverage) < 0.06;
  for (const [name, v, dy] of refs) svg += `<line class="ref-line" x1="${L}" x2="${W - R + 4}" y1="${y(v)}" y2="${y(v)}"/><text class="ref-text" x="${W - R + 8}" y="${y(v) + (sameish ? dy : 4)}">${name} ${pct(v)}</text>`;
  iters.forEach((it, i) => {
    const cx = L + groupW * (i + 0.5);
    const bars = [['Grounding', it.grounding_rate, 'var(--series-1)', `${it.supported} of ${it.total_claims} claims supported`], ['Coverage', it.coverage_score, 'var(--series-2)', 'sub-questions with a supported claim']];
    bars.forEach(([name, v, color, detail], j) => {
      const x = cx - barW - 1 + j * (barW + 2);
      const top = y(v);
      svg += `<path d="${vbar(x, top, barW, y(0) - top)}" style="fill:${color}" data-tip="${tip([`Iteration ${it.iteration} · ${name} ${pct(v)}`, detail, `bar ≥ ${pct(name === 'Grounding' ? thresholds.grounding : thresholds.coverage)}`])}"/>`;
      svg += `<text class="val-text" x="${x + barW / 2}" y="${top - 6}" text-anchor="middle">${pct(v)}</text>`;
    });
    svg += `<text class="axis-text" x="${cx}" y="${H - 8}" text-anchor="middle">Iteration ${it.iteration}${i > 0 ? ' (loop-back)' : ''}</text>`;
  });
  const legend = `<div class="viz-legend"><span><i class="viz-swatch" style="background:var(--series-1)"></i>Grounding: supported ÷ all claims</span><span><i class="viz-swatch" style="background:var(--series-2)"></i>Coverage: sub-questions answered</span></div>`;
  return card('Grounding & coverage by iteration', 'Each loop-back should raise the bars above the thresholds that end the loop.',
    `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Grounding and coverage by iteration">${svg}</svg>`, legend,
    table(['Iteration', 'Grounding', 'Coverage', 'Claims'], iters.map((it) => [it.iteration, pct(it.grounding_rate), pct(it.coverage_score), it.total_claims])));
}

// -- chart B: verdict mix per iteration (100% stacked bars, status palette + icons)
function verdictChart(iters) {
  const W = 520; const rowH = 26; const gap = 14; const L = 86; const R = 8;
  const H = iters.length * (rowH + gap) + 6;
  const plotW = W - L - R;
  const parts = [['Supported', 'supported', 'var(--good)', '✓'], ['Unsupported', 'unsupported', 'var(--warning)', '⚠'], ['Contradicted', 'contradicted', 'var(--critical)', '✕']];
  let svg = '';
  iters.forEach((it, i) => {
    const yy = 4 + i * (rowH + gap);
    const total = Math.max(1, it.total_claims);
    let x = L;
    svg += `<text class="axis-text" x="0" y="${yy + rowH / 2 + 4}">Iteration ${it.iteration}</text>`;
    parts.forEach(([name, key, color, icon], j) => {
      const n = it[key] || 0;
      if (!n) return;
      const w = Math.max(2, plotW * (n / total) - (j < parts.length - 1 ? 2 : 0));
      const last = parts.slice(j + 1).every(([, k]) => !(it[k] || 0));
      const d = last ? hbar(x, yy, w, rowH, 6) : `M${x},${yy} h${w} v${rowH} h${-w} Z`;
      svg += `<path d="${d}" style="fill:${color}" data-tip="${tip([`Iteration ${it.iteration} · ${icon} ${name}`, `${n} of ${it.total_claims} claims (${pct(n / total)})`])}"/>`;
      if (w > 34) svg += `<text x="${x + w / 2}" y="${yy + rowH / 2 + 4}" text-anchor="middle" style="fill:#fff;font-size:11px;font-weight:600">${icon} ${n}</text>`;
      x += w + 2;
    });
  });
  const legend = `<div class="viz-legend">${parts.map(([name, , color, icon]) => `<span><i class="viz-swatch" style="background:${color}"></i>${icon} ${name}</span>`).join('')}</div>`;
  return card('Claim verdicts', 'Atomic claims checked against their cited sources. Only the green ones can be published.',
    `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Claim verdicts per iteration">${svg}</svg>`, legend,
    table(['Iteration', 'Supported', 'Unsupported', 'Contradicted'], iters.map((it) => [it.iteration, it.supported, it.unsupported, it.contradicted])));
}

// -- chart C: source authority (single hue; tier boundaries + admissibility floor as references)
function authorityChart(sources, floor) {
  const rows = [...sources].sort((a, b) => b.score - a.score).slice(0, 12);
  if (!rows.length) return card('Source authority', 'No sources indexed.', '<div class="dwait">No sources.</div>');
  const W = 520; const rowH = 18; const gap = 9; const L = 150; const R = 40; const T = 18;
  const H = T + rows.length * (rowH + gap);
  const plotW = W - L - R;
  const x = (v) => L + plotW * v;
  let svg = '';
  for (const [v, label] of [[floor, 'floor'], [0.65, 'tier 2'], [0.85, 'tier 1']]) {
    svg += `<line class="ref-line" x1="${x(v)}" x2="${x(v)}" y1="${T - 6}" y2="${H}"/><text class="ref-text" x="${x(v)}" y="${T - 9}" text-anchor="middle">${label} ${v.toFixed(2)}</text>`;
  }
  rows.forEach((s, i) => {
    const yy = T + i * (rowH + gap);
    const name = (s.domain || s.title || s.source_id).replace(/^www\./, '');
    svg += `<text class="axis-text" x="${L - 10}" y="${yy + rowH / 2 + 4}" text-anchor="end">${esc(name.length > 22 ? `${name.slice(0, 21)}…` : name)}</text>`;
    svg += `<path d="${hbar(L, yy, plotW * s.score, rowH)}" style="fill:var(--series-1)" data-tip="${tip([`${s.source_id} · ${s.title}`, `${(s.tier || '').replace('_', ' ')} · authority ${s.score.toFixed(2)}`, s.cited ? 'cited in the report' : 'not cited in the report'])}"/>`;
    svg += `<text class="val-text" x="${x(s.score) + 6}" y="${yy + rowH / 2 + 4}">${s.score.toFixed(2)}${s.cited ? '<tspan class="cited-dot" dx="4">●</tspan>' : ''}</text>`;
  });
  return card('Source authority', 'Deterministic tier scores. Below the floor a source cannot support a claim. ● marks sources cited in the report.',
    `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Source authority scores">${svg}</svg>`, '',
    table(['Source', 'Domain', 'Tier', 'Authority', 'Cited'], rows.map((s) => [s.source_id, s.domain, s.tier, s.score.toFixed(2), s.cited ? 'yes' : 'no'])));
}

// -- chart D: Mesh tokens by agent (single hue)
function tokenChart(traces) {
  const per = {};
  for (const t of traces) {
    const a = (per[t.agent_name] ||= { tokens: 0, calls: 0, ms: 0 });
    a.tokens += t.total_tokens || 0; a.calls += 1; a.ms += t.latency_ms || 0;
  }
  const rows = Object.entries(per).sort((a, b) => b[1].tokens - a[1].tokens);
  if (!rows.length) return card('Mesh API usage by agent', 'No Mesh calls recorded.', '<div class="dwait">No calls.</div>');
  const max = Math.max(...rows.map(([, a]) => a.tokens), 1);
  const W = 520; const rowH = 18; const gap = 9; const L = 96; const R = 120;
  const H = rows.length * (rowH + gap);
  const plotW = W - L - R;
  let svg = '';
  rows.forEach(([agent, a], i) => {
    const yy = i * (rowH + gap);
    const w = plotW * (a.tokens / max);
    svg += `<text class="axis-text" x="${L - 10}" y="${yy + rowH / 2 + 4}" text-anchor="end">${esc(agent)}</text>`;
    svg += `<path d="${hbar(L, yy, Math.max(2, w), rowH)}" style="fill:var(--series-1)" data-tip="${tip([agent, `${fmt(a.tokens)} tokens · ${a.calls} call(s)`, `avg latency ${fmt(Math.round(a.ms / a.calls))} ms`])}"/>`;
    svg += `<text class="val-text" x="${L + Math.max(2, w) + 6}" y="${yy + rowH / 2 + 4}">${fmt(a.tokens)} tok · ${a.calls}×</text>`;
  });
  return card('Mesh API usage by agent', 'Every token went through the Mesh gateway, which enforces model allowlists and per-run budgets.',
    `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Mesh tokens by agent">${svg}</svg>`, '',
    table(['Agent', 'Tokens', 'Calls', 'Avg latency (ms)'], rows.map(([agent, a]) => [agent, fmt(a.tokens), a.calls, Math.round(a.ms / a.calls)])));
}

export function renderInsights(el, { run, events, traces, sources }) {
  const report = run?.report;
  const iters = events.filter((e) => e.event_type === 'metrics_computed').map((e) => e.payload);
  if (!iters.length) { el.innerHTML = '<div class="py-12 text-center text-sm text-slate-500">Insights appear once the Verifier has scored at least one iteration.</div>'; return; }
  const route = [...events].reverse().find((e) => e.event_type === 'route_decision')?.payload || {};
  const thresholds = { grounding: route.grounding_threshold ?? 0.75, coverage: route.coverage_threshold ?? 0.7 };
  const floor = 0.45;
  const first = iters[0]; const last = iters[iters.length - 1];
  const srcRows = (report?.sources || []).map((s) => ({ source_id: s.source_id, title: s.title, domain: s.domain, tier: s.tier, score: s.authority_score, cited: s.cited_in_report }));
  const fallbackSources = sources.map((s) => ({ source_id: s.source_id, title: s.title, domain: s.domain, tier: s.authority_tier, score: s.authority_score, cited: false }));
  const m = report?.metrics;
  const secs = run?.completed_at && run?.created_at ? Math.max(0, (Date.parse(run.completed_at) - Date.parse(run.created_at)) / 1000) : null;
  const delta = last.grounding_rate - first.grounding_rate;
  const tokens = traces.reduce((a, t) => a + (t.total_tokens || 0), 0);
  const heroes = [
    ['Final grounding', pct(last.grounding_rate), iters.length > 1 ? `${delta >= 0 ? '+' : ''}${Math.round(delta * 100)} pts after ${iters.length - 1} loop-back(s)` : 'first draft', delta > 0 ? 'up' : ''],
    ['Coverage', pct(last.coverage_score), `${iters.length} research pass(es)`, ''],
    ['Published / stopped', m ? `${m.claims_in_report} / ${m.claims_removed}` : '—', 'verified claims vs claims removed', ''],
    ['Mesh tokens', fmt(tokens), `${traces.filter((t) => t.operation !== 'embedding').length} model calls${secs !== null ? ` · ${secs.toFixed(1)} s` : ''}`, ''],
  ];
  el.innerHTML = `<div class="fade-in space-y-4">
    <div class="grid grid-cols-2 gap-3 lg:grid-cols-4">${heroes.map(([k, v, s, cls]) => `<div class="viz-hero"><span>${esc(k)}</span><b>${esc(v)}</b><i class="${cls}">${esc(s)}</i></div>`).join('')}</div>
    <div class="grid gap-4 xl:grid-cols-2">${iterationChart(iters, thresholds)}${verdictChart(iters)}${authorityChart(srcRows.length ? srcRows : fallbackSources, floor)}${tokenChart(traces)}</div>
  </div>`;
}

export function initChartTips() {
  const tipEl = document.getElementById('vizTip');
  document.addEventListener('mousemove', (e) => {
    const target = e.target.closest?.('.viz [data-tip]');
    if (!target) { tipEl.classList.add('hidden'); return; }
    const [head, ...rest] = target.getAttribute('data-tip').split('\n');
    tipEl.innerHTML = `<b>${esc(head)}</b>${rest.map((r) => `<div class="text-slate-400">${esc(r)}</div>`).join('')}`;
    tipEl.classList.remove('hidden');
    const x = Math.min(window.innerWidth - tipEl.offsetWidth - 12, e.clientX + 14);
    const y = Math.min(window.innerHeight - tipEl.offsetHeight - 12, e.clientY + 14);
    tipEl.style.left = `${x}px`; tipEl.style.top = `${y}px`;
  });
}
export { NS };
