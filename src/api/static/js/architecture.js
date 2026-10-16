// "How it works" deep-dive views: Workflow storyboard, Agents & prompts, State & contracts, Run timeline (Gantt).
// All content comes from /api/architecture (introspected from the running code) and live aggregates.
'use strict';
import { $, $$, api, esc, fmt, pct, store, toast } from './core.js';

const A = { arch: null, stats: null, view: 'graph', getRun: () => store.runId, timelineFor: null };
const KIND_COLOR = { guardrail: 'rgb(var(--orange))', agent: 'rgb(var(--accent))', decision: 'rgb(var(--purple))', tool: 'rgb(var(--cyan))', store: 'rgb(var(--teal))' };
const NODE_KIND = { input_guard: 'guardrail', output_guard: 'guardrail', router: 'decision', planner: 'agent', researcher: 'agent', drafter: 'agent', verifier: 'agent', synthesizer: 'agent', followup: 'agent' };
const LANE_LABEL = { input_guard: 'Input shield', planner: 'Planner', researcher: 'Researcher', drafter: 'Drafter', verifier: 'Verifier', synthesizer: 'Synthesizer', output_guard: 'Output shield', followup: 'Follow-up', embeddings: 'Mesh embeddings' };
const AGENT_LANE = { guardian: 'input_guard', planner: 'planner', researcher: 'researcher', drafter: 'drafter', verifier: 'verifier', synthesizer: 'synthesizer', followup: 'followup', embedder: 'embeddings' };

async function load(force = false) {
  if (!A.arch || force) A.arch = await api('/api/architecture');
  A.stats = await api('/api/architecture/stats');
}

const ms = (v) => (v === undefined || v === null ? '—' : v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`);
const pill = (kind) => `<span class="kind-pill" style="--c:${KIND_COLOR[kind] || 'rgb(var(--gray))'}">${esc(kind)}</span>`;

// ------------------------------------------------------------------ workflow storyboard
function renderWorkflow(el) {
  const { workflow, agents } = A.arch; const st = A.stats;
  const byNode = Object.fromEntries(agents.filter((a) => a.node).map((a) => [a.node, a]));
  const r = st.runs;
  const hero = [
    ['Runs recorded', fmt(r.total), Object.entries(r.by_status).map(([k, v]) => `${v} ${k}`).join(' · ') || 'none yet'],
    ['Average grounding', r.avg_grounding === null ? '—' : pct(r.avg_grounding), 'supported ÷ all claims, final pass'],
    ['Loop-back rate', r.loop_back_rate === null ? '—' : pct(r.loop_back_rate), 'runs that needed another research pass'],
    ['Blocked by guardrails', r.block_rate === null ? '—' : pct(r.block_rate), 'questions refused before any agent ran'],
  ];
  el.innerHTML = `<div class="fade-in space-y-5">
    <div class="grid grid-cols-2 gap-3 lg:grid-cols-4">${hero.map(([k, v, s]) => `<div class="viz-hero"><span>${esc(k)}</span><b>${esc(v)}</b><i>${esc(s)}</i></div>`).join('')}</div>
    <div class="wf">${workflow.map((step, i) => {
      const agent = step.node ? byNode[step.node] : null;
      const kind = NODE_KIND[step.node] || (step.id === 'audit' ? 'store' : 'tool');
      const nstat = step.node ? st.nodes[step.node] : null;
      const astat = agent?.llm ? st.agents.find((a) => a.agent === agent.id) : null;
      return `<section class="wf-step" style="--c:${KIND_COLOR[kind]}">
        <div class="wf-rail"><span class="wf-num">${i + 1}</span></div>
        <div class="wf-card">
          <div class="flex flex-wrap items-center gap-2"><h3 class="wf-title">${esc(step.title)}</h3>${pill(kind)}<span class="text-[12.5px] text-slate-400">${esc(step.actor)}</span>
            ${step.node ? `<button class="btn-ghost ml-auto" data-arch-focus="${esc(step.node)}">Show on graph</button>` : ''}</div>
          <p class="wf-what">${esc(step.what)}</p>
          <div class="wf-io"><span class="wf-io-k">In</span>${step.inputs.map((x) => `<code>${esc(x)}</code>`).join('')}<span class="wf-arrow">→</span><span class="wf-io-k">Out</span>${step.outputs.map((x) => `<code>${esc(x)}</code>`).join('')}</div>
          <div class="mt-3 flex flex-wrap gap-1.5">${step.checks.map((c) => `<span class="wf-check">✓ ${esc(c)}</span>`).join('')}</div>
          ${step.branches.length ? `<div class="mt-3 space-y-1.5">${step.branches.map((b) => `<div class="wf-branch ${b.to === 'end_blocked' ? 'danger' : b.to === 'retrieve' ? 'loop' : ''}"><b>${b.to === 'end_blocked' ? '⛔ End' : b.to === 'retrieve' ? '↩ Back to Retrieve' : `→ ${esc(b.to[0].toUpperCase() + b.to.slice(1))}`}</b> ${esc(b.when)}</div>`).join('')}</div>` : ''}
          ${nstat || astat ? `<div class="wf-stats">${nstat ? `<span>step time <b>${ms(nstat.avg_ms)}</b> avg · <b>${ms(nstat.p95_ms)}</b> p95 over ${nstat.runs} run(s)</span>` : ''}${astat ? `<span>model <b>${esc(agent.model)}</b> · ${astat.calls} calls · ${fmt(astat.avg_tokens)} tok/call · ${pct(astat.success_rate)} success</span>` : agent?.model ? `<span>model <b>${esc(agent.model)}</b></span>` : ''}</div>` : ''}
        </div></section>`;
    }).join('')}</div></div>`;
}

// ------------------------------------------------------------------ agents & prompts
function renderAgents(el) {
  const { agents } = A.arch; const st = A.stats;
  el.innerHTML = `<div class="fade-in grid gap-4 xl:grid-cols-2">${agents.map((a) => {
    const s = st.agents.find((x) => x.agent === a.id);
    const contracts = [a.llm_contract, ...(a.produces || [])].filter(Boolean);
    return `<section class="viz-card" style="--c:${KIND_COLOR[a.kind]}">
      <div class="flex flex-wrap items-center gap-2"><h3 class="text-[17px] font-semibold tracking-tight">${esc(a.name)}</h3>${pill(a.kind)}${a.model ? `<code class="arch-model">${esc(a.model)}</code>` : '<span class="text-[12px] text-slate-500">deterministic, no model</span>'}</div>
      <p class="mt-1.5 text-[13.5px] leading-relaxed text-slate-300">${esc(a.role)}</p>
      ${s ? `<div class="arch-stats"><div><span>calls</span><b>${fmt(s.calls)}</b></div><div><span>avg latency</span><b>${ms(s.avg_latency_ms)}</b></div><div><span>p95</span><b>${ms(s.p95_latency_ms)}</b></div><div><span>tokens / call</span><b>${fmt(s.avg_tokens)}</b></div><div><span>success</span><b>${pct(s.success_rate)}</b></div><div><span>JSON repaired</span><b>${fmt(s.repaired)}</b></div></div>` : ''}
      ${a.tools.length ? `<div class="arch-k">Tools</div><ul class="arch-list">${a.tools.map((t) => `<li>${esc(t)}</li>`).join('')}</ul>` : ''}
      <div class="arch-k">Guardrails & checks</div><ul class="arch-list">${a.guardrails.map((g) => `<li>${esc(g)}</li>`).join('')}</ul>
      ${contracts.length ? `<div class="arch-k">Contracts</div><div class="flex flex-wrap gap-1.5">${contracts.map((c, i) => `<button class="arch-contract" data-contract="${esc(c)}">${i === 0 && a.llm_contract ? 'returns ' : 'produces '}<b>${esc(c)}</b></button>`).join('')}</div>` : ''}
      ${a.prompts.map((p) => `<details class="arch-prompt"><summary>System prompt: ${esc(p.name)} <span class="text-slate-500">· ${fmt(p.text.length)} chars</span></summary><pre>${esc(p.text)}</pre></details>`).join('')}
      <div class="mt-3 font-mono text-[10.5px] text-slate-500">${esc(a.code)}</div>
    </section>`;
  }).join('')}</div>`;
}

// ------------------------------------------------------------------ state & contracts
function contractHtml(name) {
  const c = A.arch.contracts[name];
  if (!c) return `<p class="text-slate-500">${esc(name)} is described in the workflow.</p>`;
  return `<div class="flex items-center gap-2"><h3 class="text-[17px] font-semibold tracking-tight">${esc(name)}</h3>${pill(c.kind === 'llm_output' ? 'llm output' : 'domain')}</div>
    ${c.doc ? `<p class="mt-1 text-[13px] text-slate-400">${esc(c.doc)}</p>` : ''}
    <table class="arch-table mt-3"><thead><tr><th>Field</th><th>Type</th><th>Description</th></tr></thead><tbody>${c.fields.map((f) => `<tr><td><code>${esc(f.name)}</code>${f.required ? '<span class="req" title="required">●</span>' : ''}</td><td><code class="text-slate-400">${esc(f.type)}</code></td><td class="text-slate-400">${esc(f.description || '')}</td></tr>`).join('')}</tbody></table>`;
}

function renderState(el) {
  const { state, contracts } = A.arch;
  const reducerCls = { merge_dicts: 'r-merge', append_list: 'r-append' };
  const groups = { llm_output: [], domain: [] };
  Object.entries(contracts).forEach(([name, c]) => groups[c.kind].push(name));
  el.innerHTML = `<div class="fade-in space-y-5">
    <section class="viz-card"><div class="viz-title">LangGraph state channels</div>
      <div class="viz-sub">The shared memory every node reads from and returns partial updates to. The reducer decides how each update merges in: <b>merge_dicts</b> keeps verified sub-questions when a loop-back re-drafts others, and <b>append_list</b> makes the audit history append-only.</div>
      <div class="mt-3 overflow-x-auto"><table class="arch-table"><thead><tr><th>Channel</th><th>Type</th><th>Reducer</th><th>Written by</th><th>Meaning</th></tr></thead><tbody>
      ${state.map((c) => `<tr><td><code>${esc(c.name)}</code></td><td><code class="text-slate-400">${esc(c.type)}</code></td><td><span class="reducer ${reducerCls[c.reducer] || ''}">${esc(c.reducer)}</span></td><td>${c.written_by.map((n) => `<span class="node-chip">${esc(n)}</span>`).join(' ') || '<span class="text-slate-500">input</span>'}</td><td class="text-slate-400">${esc(c.description)}</td></tr>`).join('')}
      </tbody></table></div></section>
    <div class="grid gap-4 xl:grid-cols-2">
      <section class="viz-card"><div class="viz-title">What the models must return</div><div class="viz-sub">Every Mesh response is validated against one of these contracts. On failure, the gateway feeds the error back and the model repairs its output.</div>
        <div class="mt-3 flex flex-wrap gap-1.5">${groups.llm_output.map((n) => `<button class="arch-contract" data-contract="${esc(n)}"><b>${esc(n)}</b></button>`).join('')}</div></section>
      <section class="viz-card"><div class="viz-title">Domain records</div><div class="viz-sub">The auditable artefacts that flow through state, SQLite and the API.</div>
        <div class="mt-3 flex flex-wrap gap-1.5">${groups.domain.map((n) => `<button class="arch-contract" data-contract="${esc(n)}"><b>${esc(n)}</b></button>`).join('')}</div></section>
    </div>
    <section class="viz-card"><div class="viz-title">Mesh API gateway</div><div class="viz-sub">The only way out to a model.</div>
      <div class="arch-stats mt-3">${Object.entries(A.arch.gateway).map(([k, v]) => `<div><span>${esc(k.replace(/_/g, ' '))}</span><b class="text-[13px]">${esc(Array.isArray(v) ? v.join(', ') : v)}</b></div>`).join('')}</div></section>
  </div>`;
}

// ------------------------------------------------------------------ run timeline (Gantt)
function niceStep(totalMs) {
  const target = totalMs / 8;
  return [100, 250, 500, 1000, 2000, 5000, 10000, 15000, 30000, 60000].find((s) => s >= target) || 120000;
}

async function renderTimeline(el) {
  const runId = A.getRun();
  if (!runId) { el.innerHTML = '<div class="dwait">Pick a run in the Live graph tab (or run one) to see its timeline.</div>'; return; }
  el.innerHTML = '<div class="dwait">Loading timeline…</div>';
  let tl;
  try { tl = await api(`/api/runs/${runId}/timeline`); } catch (e) { el.innerHTML = `<div class="dwait">${esc(e.message)}</div>`; return; }
  if (!tl.spans.length) { el.innerHTML = '<div class="dwait">This run has no recorded node executions yet.</div>'; return; }
  const lanes = [...tl.lanes];
  if (tl.llm_calls.some((c) => c.agent === 'embedder')) lanes.push('embeddings');
  if (tl.llm_calls.some((c) => c.agent === 'followup') && !lanes.includes('followup')) lanes.push('followup');
  const W = 1100; const L = 128; const R = 20; const T = 34; const laneH = 40; const H = T + lanes.length * laneH + 26;
  const dur = Math.max(tl.duration_ms, 1);
  const x = (t) => L + ((W - L - R) * t) / dur;
  const laneY = (lane) => T + lanes.indexOf(lane) * laneH;
  const step = niceStep(dur);
  let svg = '';
  for (let t = 0; t <= dur + 1; t += step) svg += `<line class="grid-line" x1="${x(t)}" x2="${x(t)}" y1="${T - 6}" y2="${H - 22}"/><text class="axis-text" x="${x(t)}" y="${H - 6}" text-anchor="middle">${step >= 1000 ? `${(t / 1000).toFixed(t % 1000 ? 1 : 0)}s` : `${t}ms`}</text>`;
  lanes.forEach((lane, i) => {
    const y = T + i * laneH;
    if (i % 2 === 0) svg += `<rect x="${L}" y="${y}" width="${W - L - R}" height="${laneH}" style="fill:var(--fill)" opacity=".45" rx="6"/>`;
    svg += `<text class="axis-text" x="${L - 12}" y="${y + laneH / 2 + 4}" text-anchor="end" style="font-weight:600">${esc(LANE_LABEL[lane] || lane)}</text>`;
  });
  for (const sp of tl.spans) {
    const y = laneY(sp.node) + 7; const w = Math.max(3, x(sp.end_ms) - x(sp.start_ms));
    svg += `<rect x="${x(sp.start_ms)}" y="${y}" width="${w}" height="14" rx="4" style="fill:var(--series-1)" data-tip="${esc([`${LANE_LABEL[sp.node] || sp.node}${sp.iteration > 1 ? ` · iteration ${sp.iteration}` : ''}`, `${ms(sp.duration_ms)} (from ${ms(sp.start_ms)} to ${ms(sp.end_ms)})`, sp.message].join('\n'))}"/>`;
    if (sp.iteration > 1 && w > 26) svg += `<text x="${x(sp.start_ms) + 6}" y="${y + 11}" style="fill:#fff;font-size:10px;font-weight:700">×${sp.iteration}</text>`;
  }
  for (const c of tl.llm_calls) {
    const lane = AGENT_LANE[c.agent] || 'researcher';
    if (!lanes.includes(lane)) continue;
    const y = laneY(lane) + 25; const w = Math.max(2, x(c.end_ms) - x(c.start_ms));
    svg += `<rect x="${x(c.start_ms)}" y="${y}" width="${w}" height="7" rx="2" style="fill:var(--series-2)" data-tip="${esc([`Mesh call · ${c.agent}`, `${c.model} · ${c.status}`, `${ms(c.latency_ms)} · ${fmt(c.tokens)} tokens`].join('\n'))}"/>`;
  }
  for (const m of tl.markers) {
    const mx = x(m.t_ms);
    const color = m.type === 'loop_back' ? 'rgb(var(--orange))' : m.type === 'guardrail_blocked' || m.type === 'run_failed' || m.type === 'run_blocked' ? 'rgb(var(--red))' : m.type === 'run_completed' ? 'rgb(var(--green))' : 'rgb(var(--purple))';
    const label = { loop_back: 'loop-back', route_decision: m.route === 'researcher' ? 'decide: loop' : 'decide: publish', run_completed: 'published', guardrail_blocked: 'blocked', budget_guard: 'budget guard', run_blocked: 'blocked', run_failed: 'failed' }[m.type] || m.type;
    if (m.type === 'route_decision' || m.type === 'loop_back' || m.type === 'run_completed' || m.type === 'guardrail_blocked') {
      svg += `<line x1="${mx}" x2="${mx}" y1="${T - 8}" y2="${H - 22}" style="stroke:${color};stroke-width:1.25;stroke-dasharray:4 3" data-tip="${esc([label, m.message, `at ${ms(m.t_ms)}`].join('\n'))}"/>`;
      const nearEdge = mx > W - R - 70;
      if (m.type !== 'route_decision') svg += `<text x="${nearEdge ? mx - 4 : mx + 4}" y="${T - 12}" text-anchor="${nearEdge ? 'end' : 'start'}" style="fill:${color};font-size:10.5px;font-weight:600">${esc(label)}</text>`;
    }
  }
  const totalCalls = tl.llm_calls.filter((c) => c.operation !== 'embedding').length;
  const busiest = Object.entries(tl.spans.reduce((a, s) => { a[s.node] = (a[s.node] || 0) + s.duration_ms; return a; }, {})).sort((a, b) => b[1] - a[1])[0];
  el.innerHTML = `<div class="fade-in space-y-4">
    <div class="grid grid-cols-2 gap-3 lg:grid-cols-4">
      <div class="viz-hero"><span>Wall time</span><b>${ms(tl.duration_ms)}</b><i>from question to publication</i></div>
      <div class="viz-hero"><span>Research passes</span><b>${Math.max(1, ...tl.spans.filter((s) => s.node === 'researcher').map((s) => s.iteration))}</b><i>${tl.markers.filter((m) => m.type === 'loop_back').length} loop-back(s)</i></div>
      <div class="viz-hero"><span>Mesh model calls</span><b>${totalCalls}</b><i>${tl.llm_calls.length - totalCalls} embedding batch(es)</i></div>
      <div class="viz-hero"><span>Slowest step</span><b>${esc(LANE_LABEL[busiest?.[0]] || '—')}</b><i>${busiest ? `${ms(busiest[1])} in total` : ''}</i></div>
    </div>
    <section class="viz-card viz"><div class="viz-title">Run timeline</div>
      <div class="viz-sub">Each lane is a graph node. Wide bars are node executions; thin bars are the Mesh API calls made inside them (parallel calls overlap). Dashed lines mark the router's decisions and loop-backs.</div>
      <div class="viz-legend"><span><i class="viz-swatch" style="background:var(--series-1)"></i>Node execution</span><span><i class="viz-swatch" style="background:var(--series-2)"></i>Mesh API call</span><span><i class="viz-swatch" style="background:rgb(var(--orange))"></i>Loop-back</span><span><i class="viz-swatch" style="background:rgb(var(--purple))"></i>Router decision</span></div>
      <div class="mt-3 overflow-x-auto"><svg viewBox="0 0 ${W} ${H}" style="min-width:760px" role="img" aria-label="Run timeline">${svg}</svg></div>
      <details class="viz-table"><summary>View as table</summary><table><thead><tr><th>Node</th><th>Iteration</th><th>Start</th><th>Duration</th></tr></thead><tbody>${tl.spans.map((s) => `<tr><td>${esc(LANE_LABEL[s.node] || s.node)}</td><td>${s.iteration}</td><td>${ms(s.start_ms)}</td><td>${ms(s.duration_ms)}</td></tr>`).join('')}</tbody></table></details>
    </section></div>`;
}

// ------------------------------------------------------------------ routing between sub-views
export async function showArchView(view) {
  A.view = view;
  document.body.dataset.xview = view;
  $$('[data-xview]').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.xview === view)));
  $$('[data-xpanel]').forEach((p) => p.classList.toggle('hidden', p.dataset.xpanel !== view));
  if (view === 'graph') return;
  const el = $(`[data-xpanel="${view}"]`);
  if (view === 'timeline') { await renderTimeline(el); return; }
  try {
    if (!A.arch || !A.stats) el.innerHTML = '<div class="dwait">Loading architecture…</div>';
    await load();
    ({ workflow: renderWorkflow, agents: renderAgents, state: renderState })[view](el);
  } catch (e) { el.innerHTML = `<div class="dwait">${esc(e.message)}</div>`; }
}

export function initArchitecture({ getRun, focusNode }) {
  A.getRun = getRun;
  document.body.dataset.xview = 'graph';
  $('#xSubnav').addEventListener('click', (e) => { const b = e.target.closest('[data-xview]'); if (b) showArchView(b.dataset.xview); });
  document.addEventListener('click', (e) => {
    const c = e.target.closest('[data-contract]');
    if (c && A.arch) { $('#archSheetBody').innerHTML = `<div class="flex justify-end"><button class="icon-btn" data-close-arch aria-label="Close">✕</button></div>${contractHtml(c.dataset.contract)}`; $('#archSheet').classList.remove('hidden'); return; }
    if (e.target.closest('[data-close-arch]') || e.target.id === 'archSheet') $('#archSheet').classList.add('hidden');
    const f = e.target.closest('[data-arch-focus]');
    if (f) { showArchView('graph'); focusNode(f.dataset.archFocus); }
  });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') $('#archSheet').classList.add('hidden'); });
}

export function refreshTimelineIfVisible() { if (A.view === 'timeline') renderTimeline($('[data-xpanel="timeline"]')).catch((e) => toast(e.message)); }
