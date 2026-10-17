// Agent group chat: renders the hand-offs DeepDive's agents post to each other (AGENT_MESSAGE audit events).
'use strict';
import { esc, pct } from './core.js';
import { ICONS } from './explorer.js';

export const AGENTS = [
  { id: 'input_guard', label: 'Shield', role: 'input guardrails', icon: 'shield', color: 'rgb(var(--orange))' },
  { id: 'planner', label: 'Planner', role: 'splits · revises', icon: 'plan', color: 'rgb(var(--purple))' },
  { id: 'master', label: 'Master Approver', role: 'approves plan & rounds', icon: 'master', color: 'rgb(var(--indigo))' },
  { id: 'researcher', label: 'Researcher', role: 'search · score · index', icon: 'search', color: 'rgb(var(--cyan))' },
  { id: 'drafter', label: 'Drafter', role: 'cited drafts', icon: 'pen', color: 'rgb(var(--pink))' },
  { id: 'verifier', label: 'Verifier', role: 'claim-by-claim proof', icon: 'verify', color: 'rgb(var(--green))' },
  { id: 'synthesizer', label: 'Synthesizer', role: 'verified-only report', icon: 'spark', color: 'rgb(var(--accent))' },
  { id: 'report_approver', label: 'Report Approver', role: 'verifies every sentence', icon: 'reportcheck', color: 'rgb(var(--teal))' },
  { id: 'output_guard', label: 'Publish', role: 'output shield · seal', icon: 'shieldOut', color: 'rgb(var(--orange))' },
];
export const AGENT = Object.fromEntries(AGENTS.map((a) => [a.id, a]));
export const CHAT_AGENT = {
  ...AGENT, output_guard: { ...AGENT.output_guard, label: 'Output shield' },
  // senders from runs recorded before the Master Approver existed
  approver: { id: 'approver', label: 'Approver', icon: 'approve', color: 'rgb(var(--indigo))' },
  lead: { ...AGENT.master },
  quality_gate: { id: 'quality_gate', label: 'Quality gate', icon: 'route', color: 'rgb(var(--gray))' },
};
const MENTION = { 'Report Approver': 'report_approver', 'Master Approver': 'master', Planner: 'planner', Approver: 'approver', Researcher: 'researcher', Drafter: 'drafter', Verifier: 'verifier', Synthesizer: 'synthesizer', 'Output shield': 'output_guard' };
export const svgIcon = (name) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">${ICONS[name]}</svg>`;

const KIND_LABEL = { proposal: 'Proposal', critique: 'Changes requested', revision: 'Revision', approval: 'Approved', evidence: 'Evidence', draft: 'Drafts', verdict: 'Verdicts', decision: 'Decision', report: 'Report' };
function kindPill(m) {
  const again = (m.to || []).includes('researcher');
  if (m.sender === 'report_approver') return m.kind === 'approval' ? ['Report approved', 'good'] : m.kind === 'fix_request' ? ['Fix request', 'warn'] : ['Removed failing sentences', 'warn'];
  if (m.kind === 'approval') return m.passed ? ['Plan approved', 'good'] : ['Plan approved with open issues', 'warn'];
  if (m.kind === 'critique') return ['Changes requested', 'warn'];
  if (m.kind === 'fix_request') return ['Fix request', 'warn'];
  if (m.kind === 'fix') return ['Quick fix', ''];
  if (m.kind === 'decision' && m.sender === 'master') return again ? (m.passed ? ['Extra pass', 'warn'] : ['Another round', 'warn']) : m.passed ? ['Approved for synthesis', 'good'] : ['Approved with gaps', 'warn'];
  if (m.kind === 'decision' && m.sender === 'quality_gate') return m.passed ? ['Quality bar met', 'good'] : again ? ['Loop back', 'warn'] : ['Publish with gaps', 'warn'];
  if (m.kind === 'decision') return ['Published', 'good'];
  return [KIND_LABEL[m.kind] || m.kind, ''];
}
export const roomOf = (m) => (m.channel === 'plan_review' ? `Plan review · version ${m.round}` : m.channel === 'evidence_loop' ? `Group chat · round ${m.round}`
  : m.sender === 'report_approver' || (m.sender === 'synthesizer' && m.kind === 'fix') ? `Report check · round ${m.round}` : 'Publish');

function rich(text) {
  return esc(text)
    .replace(/\bSRC-\d{3,4}\b/g, (id) => `<span class="chip" data-src="${id}">${id}</span>`)
    .replace(/\bSQ-\d+\b/g, (id) => `<span class="sqref">${id}</span>`)
    .replace(/\b(Report Approver|Master Approver|Planner|Approver|Researcher|Drafter|Verifier|Synthesizer|Output shield)\b/g, (n) => `<span class="mention" style="--c:${CHAT_AGENT[MENTION[n]].color}">${n}</span>`);
}

export function chatMessage(m) {
  const a = CHAT_AGENT[m.sender] || { label: m.sender, icon: 'chat', color: 'rgb(var(--gray))' };
  const [kind, tone] = kindPill(m);
  const items = m.items || [];
  const li = (x) => `<li class="${x.startsWith('   ') ? 'sub' : ''}">${rich(x.trim())}</li>`;
  const isPct = m.sender === 'quality_gate' || (m.sender === 'master' && m.channel === 'evidence_loop');
  const f = (x) => (isPct ? pct(x) : Number(x).toFixed(2));
  const score = m.score !== null && m.score !== undefined && m.threshold !== null && m.threshold !== undefined
    ? `<div class="cmsg-score" data-state="${m.passed ? 'pass' : 'fail'}"><span>${isPct ? 'grounding' : 'score'} <b class="text-slate-100">${f(m.score)}</b></span><div class="pgauge" role="img" aria-label="${isPct ? 'grounding' : 'score'} ${f(m.score)}, needs ${f(m.threshold)}"><i style="width:${Math.round(Math.min(1, m.score) * 100)}%"></i><b style="left:calc(${Math.round(m.threshold * 100)}% - 1px)"></b></div><span>needs ${f(m.threshold)} ${m.passed ? '✓' : '✗'}</span></div>` : '';
  return `<div class="cmsg fade-in" style="--c:${a.color}"><div class="cmsg-av" title="${esc(a.label)}">${svgIcon(a.icon)}</div><div class="cmsg-main">
    <div class="cmsg-head"><b>${esc(a.label)}</b>${(m.to || []).length ? `<span class="cmsg-to">→ ${(m.to || []).map((n) => esc(CHAT_AGENT[n]?.label || n)).join(', ')}</span>` : ''}<span class="cmsg-kind ${tone}">${esc(kind)}</span><span class="cmsg-t">${(m.t / 1000).toFixed(1)}s</span></div>
    <div class="cmsg-bubble">${rich(m.text)}</div>
    ${items.length ? `<ul class="cmsg-items">${items.slice(0, 4).map(li).join('')}</ul>${items.length > 4 ? `<details class="cmsg-more"><summary>${items.length - 4} more</summary><ul class="cmsg-items">${items.slice(4).map(li).join('')}</ul></details>` : ''}` : ''}
    ${score}</div></div>`;
}

export function chatTranscript(messages) {
  return messages.map((m, j) => `${j === 0 || roomOf(m) !== roomOf(messages[j - 1]) ? `<div class="cdivider">${esc(roomOf(m))}</div>` : ''}${chatMessage(m)}`).join('');
}
