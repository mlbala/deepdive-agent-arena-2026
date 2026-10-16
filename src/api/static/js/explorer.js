// "How it works" — interactive architecture explorer that replays (or follows live) a real DeepDive run.
'use strict';
import { $, $$, api, esc, fmt, on, pct, store, toast, ts } from './core.js';

const SVGNS = 'http://www.w3.org/2000/svg';
const NODE_H = 62;

// ------------------------------------------------------------------ icons (24x24 stroke glyphs)
const ICONS = {
  user: '<circle cx="12" cy="8" r="3.6"/><path d="M5 20c.6-3.8 3.5-6.4 7-6.4s6.4 2.6 7 6.4"/>',
  shield: '<path d="M12 3l7 3v5.2c0 4.4-3 8.1-7 9.8-4-1.7-7-5.4-7-9.8V6z"/><path d="M9.2 12.2l1.9 1.9 3.7-3.9"/>',
  shieldOut: '<path d="M12 3l7 3v5.2c0 4.4-3 8.1-7 9.8-4-1.7-7-5.4-7-9.8V6z"/><path d="M9 12.5h6"/>',
  plan: '<circle cx="6" cy="6" r="2"/><circle cx="6" cy="18" r="2"/><circle cx="18" cy="12" r="2"/><path d="M8 6h2.5a3 3 0 0 1 3 3v.5M8 18h2.5a3 3 0 0 0 3-3v-.5M13.5 12H16"/>',
  search: '<circle cx="10.5" cy="10.5" r="5.5"/><path d="M20 20l-5.2-5.2"/>',
  db: '<ellipse cx="12" cy="6" rx="7" ry="2.8"/><path d="M5 6v12c0 1.5 3.1 2.8 7 2.8s7-1.3 7-2.8V6M5 12c0 1.5 3.1 2.8 7 2.8s7-1.3 7-2.8"/>',
  pen: '<path d="M4 20l4.2-1 10.6-10.6-3.2-3.2L5 15.8z"/><path d="M13.8 6.8l3.2 3.2"/>',
  verify: '<circle cx="10.5" cy="10.5" r="6"/><path d="M20 20l-5-5M8 10.6l1.8 1.8 3.4-3.6"/>',
  route: '<path d="M12 21v-7M12 14L6 8M12 14l6-6M6 8V4M18 8V4M4 6l2-2 2 2M16 6l2-2 2 2"/>',
  spark: '<path d="M12 3.5l1.9 4.6 4.6 1.9-4.6 1.9L12 16.5l-1.9-4.6-4.6-1.9 4.6-1.9z"/><path d="M18.5 15.5l.7 1.8 1.8.7-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7z"/>',
  doc: '<path d="M7 3h7l5 5v13H7z"/><path d="M14 3v5h5M10 13h6M10 17h6"/>',
  chat: '<path d="M4 5h16v11H9.5L5 20v-4H4z"/><path d="M8 9.5h8M8 12.5h5"/>',
  globe: '<circle cx="12" cy="12" r="8.5"/><path d="M3.5 12h17M12 3.5c2.4 2.5 3.6 5.3 3.6 8.5s-1.2 6-3.6 8.5c-2.4-2.5-3.6-5.3-3.6-8.5S9.6 6 12 3.5z"/>',
  upload: '<path d="M7 3h7l5 5v13H7z"/><path d="M14 3v5h5M12 18v-6M9.5 14.5L12 12l2.5 2.5"/>',
  chain: '<rect x="2.5" y="9" width="9" height="6" rx="3"/><rect x="12.5" y="9" width="9" height="6" rx="3"/><path d="M9 12h6"/>',
  stop: '<circle cx="12" cy="12" r="8.5"/><path d="M6 6l12 12"/>',
  mesh: '<circle cx="5" cy="12" r="2"/><circle cx="19" cy="6" r="2"/><circle cx="19" cy="18" r="2"/><path d="M7 12h4l6-5.2M11 12l6 5.2"/>',
};

// ------------------------------------------------------------------ architecture (our graph, not a generic pipeline)
const NODES = [
  { id: 'user', n: 1, label: 'You', sub: 'your question', kind: 'io', icon: 'user', x: 30, y: 190, w: 150 },
  { id: 'input_guard', n: 2, label: 'Input shield', sub: 'inject·harm·PII', kind: 'guard', icon: 'shield', x: 205, y: 190, w: 150, agent: 'guardian' },
  { id: 'planner', n: 3, label: 'Planner', sub: 'MECE split', kind: 'agent', icon: 'plan', x: 380, y: 190, w: 150, agent: 'planner' },
  { id: 'researcher', n: 4, label: 'Researcher', sub: 'retrieve·score·index', kind: 'agent', icon: 'search', x: 560, y: 190, w: 170, agent: 'researcher' },
  { id: 'index', n: 5, label: 'Evidence index', sub: 'BM25+Chroma spans', kind: 'store', icon: 'db', x: 760, y: 190, w: 160, agent: 'embedder' },
  { id: 'drafter', n: 6, label: 'Drafter', sub: 'cited drafts', kind: 'agent', icon: 'pen', x: 950, y: 190, w: 150, agent: 'drafter' },
  { id: 'verifier', n: 7, label: 'Verifier', sub: 'entail·guards·radar', kind: 'agent', icon: 'verify', x: 1050, y: 330, w: 170, agent: 'verifier' },
  { id: 'router', n: 8, label: 'Router', sub: 'route_verification', kind: 'decision', icon: 'route', cx: 960, cy: 505 },
  { id: 'synthesizer', n: 9, label: 'Synthesizer', sub: 'verified-only', kind: 'agent', icon: 'spark', x: 690, y: 474, w: 160, agent: 'synthesizer' },
  { id: 'output_guard', n: 10, label: 'Output shield', sub: 'redact · govern', kind: 'guard', icon: 'shieldOut', x: 500, y: 474, w: 160 },
  { id: 'report', n: 11, label: 'Final report', sub: 'cited·audited·PDF', kind: 'io', icon: 'doc', x: 310, y: 474, w: 160 },
  { id: 'followup', n: 12, label: 'Follow-up chat', sub: 'grounded Q&A', kind: 'agent', icon: 'chat', x: 110, y: 474, w: 170, agent: 'followup' },
  { id: 'web', label: 'Web retrieval', sub: 'Tavily→DDG→corpus', kind: 'tool', icon: 'globe', x: 470, y: 40, w: 170 },
  { id: 'docs', label: 'Your documents', sub: 'PDF·TXT·MD', kind: 'tool', icon: 'upload', x: 660, y: 40, w: 170 },
  { id: 'audit', label: 'Audit trail', sub: 'SHA-256 hash chain', kind: 'store', icon: 'chain', x: 530, y: 612, w: 180 },
  { id: 'blocked', label: 'Blocked', sub: 'ends safely', kind: 'end', icon: 'stop', x: 215, y: 296, w: 130, h: 46 },
];
const NODE = Object.fromEntries(NODES.map((n) => [n.id, n]));

const EDGES = [
  { id: 'user_guard', d: 'M180,221 L205,221', label: 'question', lx: 192, ly: 175 },
  { id: 'guard_planner', d: 'M355,221 L380,221', label: 'screened', lx: 367, ly: 175 },
  { id: 'guard_block', d: 'M280,252 L280,296', label: 'blocked', lx: 318, ly: 272, kind: 'danger', anchor: 'start' },
  { id: 'planner_res', d: 'M530,221 L560,221', label: 'sub-questions', lx: 545, ly: 175 },
  { id: 'web_res', d: 'M555,102 C555,150 612,146 618,190', label: 'searches', lx: 520, ly: 150, kind: 'tool' },
  { id: 'docs_res', d: 'M745,102 C745,150 680,146 674,190', label: 'uploads', lx: 778, ly: 150, kind: 'tool' },
  { id: 'res_index', d: 'M730,221 L760,221', label: 'sources', lx: 745, ly: 175 },
  { id: 'index_drafter', d: 'M920,221 L950,221', label: 'evidence', lx: 935, ly: 175 },
  { id: 'drafter_ver', d: 'M1100,221 C1150,221 1135,282 1135,330', label: 'drafts', lx: 1180, ly: 268 },
  { id: 'ver_router', d: 'M1135,392 C1135,468 1068,505 1014,505', label: 'verdicts', lx: 1128, ly: 462 },
  { id: 'loop', d: 'M960,463 C960,404 652,338 645,252', label: 'loop-back', lx: 820, ly: 318, kind: 'loop' },
  { id: 'router_synth', d: 'M906,505 L850,505', label: 'publish', lx: 878, ly: 556 },
  { id: 'synth_output', d: 'M690,505 L660,505', label: 'report', lx: 675, ly: 556 },
  { id: 'output_report', d: 'M500,505 L470,505', label: 'redactions', lx: 485, ly: 556 },
  { id: 'report_followup', d: 'M310,505 L280,505', label: 'follow-ups', lx: 295, ly: 556 },
  { id: 'output_audit', d: 'M580,536 C580,578 620,574 620,612', label: 'hash-chained', lx: 658, ly: 586, kind: 'audit', anchor: 'start' },
];
const GATEWAY = { x: 200, y: 360, w: 830, h: 30 };
const CONNECTORS = { // agent node -> point on the Mesh gateway bar
  input_guard: 'M250,252 L250,360', planner: 'M455,252 L455,360', researcher: 'M600,252 L600,360',
  index: 'M840,252 L840,360', drafter: 'M1015,252 L1015,360', verifier: 'M1050,368 L1030,374',
  synthesizer: 'M770,474 L770,390', followup: 'M195,474 C195,430 215,410 215,390',
};
const AGENT_NODE = { guardian: 'input_guard', planner: 'planner', researcher: 'researcher', embedder: 'index', drafter: 'drafter', verifier: 'verifier', synthesizer: 'synthesizer', followup: 'followup' };

const KIND = {
  agent: { color: '#818cf8', label: 'LLM agent · via Mesh' },
  guard: { color: '#fbbf24', label: 'Guardrail' },
  tool: { color: '#38bdf8', label: 'Deterministic tool' },
  store: { color: '#2dd4bf', label: 'Data store' },
  decision: { color: '#c084fc', label: 'Conditional edge' },
  io: { color: '#94a3b8', label: 'Input / output' },
  end: { color: '#fb7185', label: 'Terminal state' },
};

const INFO = {
  user: { what: 'Your research question enters the LangGraph. Nothing touches it before the input shield has screened it.', io: 'question → input shield', checks: ['1,000-character limit', 'PII redacted before the run row is written'], code: 'src/api/routes.py' },
  input_guard: { what: 'A deterministic shield followed by the Guardian classifier (a Mesh LLM call, as defence in depth). It blocks prompt injection, jailbreaks and credential probing, plus requests for operational harm: weapons, explosives, malware, self-harm, CSAM and violent extremism. It also strips hidden characters and redacts PII and secrets.', io: 'question → screened question | END (blocked)', checks: ['Pattern library aimed at operational intent, so research on the same topics still passes', 'Guardian can only block at ≥ 0.80 confidence and fails open', 'Blocked runs never reach any research agent'], code: 'src/guardrails/shields.py · src/agents/guardian.py' },
  planner: { what: 'Splits the question into 3–5 non-overlapping sub-questions, each with targeted search queries. If the question rests on a doubtful premise, one sub-question checks that premise.', io: 'question → PlannerOutput (SubQuestion × 3–5)', checks: ['Near-duplicate sub-questions removed (Jaccard > 0.8)', 'Pydantic contract with self-healing JSON repair'], code: 'src/agents/planner.py' },
  researcher: { what: 'Runs queries in parallel through the 3-tier retriever and deduplicates by URL and content hash. Each page is scored for authority, sanitised by the content shield and chunked into exact spans. Evidence is then ranked by authority for each sub-question. On a loop-back it first asks Mesh for refinement queries aimed at the gaps.', io: 'sub-questions (+ gaps) → SourceDocument[], EvidenceChunk[]', checks: ['Content shield: injected sentences neutralised, hidden text stripped', 'URL shield: no private IPs, metadata hosts or unsafe redirects', 'Tier 1/2/3 authority scoring; poisoned sources score lower'], code: 'src/agents/researcher.py · src/tools/web_search.py' },
  index: { what: 'Hybrid retrieval: BM25 plus ChromaDB with Mesh embeddings, merged by reciprocal-rank fusion. Every chunk keeps its exact [char_start, char_end) offsets in the source, so any quote can be traced back.', io: 'chunks → ranked evidence spans', checks: ['Drops to sparse-only retrieval if embeddings fail', 'Spans rehydrate from SQLite for follow-ups'], code: 'src/storage/vector_store.py' },
  drafter: { what: 'Writes 80–220 words per sub-question using only the evidence provided. Every factual sentence must end with [SRC-XXX] citations.', io: 'sub-question + evidence → DraftSection', checks: ['Citations outside the evidence pool are flagged', 'Evidence is treated as untrusted data, never as instructions'], code: 'src/agents/drafter.py' },
  verifier: { what: 'Splits drafts into atomic claims and checks each one against the spans of its cited sources only. Deterministic guards follow: the quote must be found in the source, the source must be one the claim cites, figures must appear in the source, and the source must clear the admissibility floor. A contradiction radar compares sources against each other.', io: 'drafts → AtomicClaim[], VerificationResult[], ContradictionItem[]', checks: ['Quote matching: exact → normalised → elliptical → fuzzy, with numbers locked', 'Claims citing sources below the authority floor are downgraded', 'Both sides of a contradiction must be quoted verbatim'], code: 'src/agents/verifier.py' },
  router: { what: 'The conditional edge. While coverage is below 70% or grounding below 75%, iterations remain and the budget soft limit hasn\'t been reached, it loops back to the Researcher for the failing sub-questions only. Otherwise it publishes.', io: 'metrics → researcher | synthesizer', checks: ['Only targeted sub-questions are re-researched', 'Budget guard suppresses loop-backs at 75% of the run budget'], code: 'src/agents/orchestrator.py · route_verification()' },
  synthesizer: { what: 'Sees only SUPPORTED claims. A claim whose source is outranked by a contradicting source is withheld, and a citation guard drops any sentence that cites sources outside the verified set.', io: 'verified claims → FinalReport', checks: ['Authority-weighted contradiction resolution', 'Removed claims are kept in the audit record'], code: 'src/agents/synthesizer.py' },
  output_guard: { what: 'Redacts PII and secrets from everything that gets published. It also attaches the governance summary (findings, budget use, models, audit-chain head) and the disclosure.', io: 'FinalReport → published FinalReport', checks: ['Secrets always redacted', 'PII redacted by policy'], code: 'src/guardrails/shields.py · OutputGuard' },
  report: { what: 'The published executive report: claim matrix, contradiction radar, sources, governance section and a PDF with the full audit appendix.', io: 'report → UI · PDF · API', checks: ['Every sentence has verified citations or states that evidence is missing'], code: 'src/tools/pdf_exporter.py' },
  followup: { what: 'Answers follow-up questions using only this run\'s evidence pool. Questions pass the input shield first; citations outside the pool are stripped.', io: 'question → FollowUpAnswer', checks: ['Answers it can\'t ground are marked not answerable', 'Input and output shields applied'], code: 'src/agents/followup.py' },
  web: { what: 'Tier 1: Tavily, called with include_answer=False so it only scrapes. Tier 2: DuckDuckGo plus BeautifulSoup for HTML and PDF. Tier 3: an offline corpus. Each tier falls back to the next.', io: 'query → RawSearchResult[]', checks: ['Every fetch and redirect is validated by the URL shield', 'Hidden HTML is removed; responses are capped in size'], code: 'src/tools/web_search.py · src/guardrails/url_safety.py' },
  docs: { what: 'Your PDF, TXT and MD files. Each is magic-byte checked, injection-neutralised and PII-redacted, then indexed with exact spans as user-supplied evidence (authority 0.75).', io: 'upload → private evidence', checks: ['Binary and spoofed files rejected', 'Duplicates removed by content hash'], code: 'src/tools/document_ingestor.py' },
  audit: { what: 'Every decision is recorded as an event, SHA-256-chained to the previous event in the run. An edit made outside the app breaks the chain; you can verify it from the Governance tab.', io: 'events → append-only SQLite', checks: ['ORM guard blocks UPDATE and DELETE', 'Hash chain exposes raw-SQL tampering'], code: 'src/storage/database.py' },
  gateway: { what: 'The only way out to an LLM. All chat and embedding calls go through Mesh API, with retries and backoff, JSON self-repair, a model allowlist, per-run budgets, and token and latency tracing.', io: 'agent → Mesh API → model', checks: ['Disallowed models are refused before any request is sent', 'Hard budget refuses calls past the per-run cap'], code: 'src/core/mesh_gateway.py' },
  blocked: { what: 'A blocked question ends here, and the reason is recorded in the hash-chained audit trail. No planner, retrieval or drafting runs.', io: 'END', checks: ['Run status: blocked'], code: 'src/agents/orchestrator.py · route_after_guard()' },
};

// ------------------------------------------------------------------ replay engine
const X = { steps: [], i: -1, playing: false, speed: 1, timer: null, follow: true, focus: null, pinned: null, runId: null, policy: null, built: false };
const DUR = { node_started: 650, node_completed: 320, claim_verified: 140, source_indexed: 220, search_executed: 260, trace: 110, guardrail_finding: 420, loop_back: 1100, route_decision: 750 };

function stepOf(item) {
  if (item.kind === 'trace') {
    const node = AGENT_NODE[item.agent_name] || 'planner';
    return { item, node, gateway: node, color: '#a78bfa', dur: DUR.trace, text: `Mesh call · ${item.agent_name} · ${item.model} · ${fmt(item.total_tokens)} tok · ${item.latency_ms} ms`, tone: 'mesh' };
  }
  const t = item.event_type; const p = item.payload || {}; const node = item.node;
  const s = { item, node: node === 'orchestrator' ? 'user' : node, dur: DUR[t] || 380, text: item.message, tone: 'base' };
  switch (t) {
    case 'run_started': s.node = 'user'; s.edge = 'user_guard'; s.text = `You asked: “${(p.query || '').slice(0, 140)}”`; break;
    case 'guardrail_check': if (p.stage === 'input' && p.status !== 'blocked') s.edge = 'guard_planner'; if (p.stage === 'output') s.edge = 'output_report'; s.tone = 'guard'; break;
    case 'guardrail_finding': s.tone = 'guard'; if (item.node === 'researcher') s.edge = p.guard === 'url_shield' ? 'web_res' : undefined; s.color = '#fbbf24'; break;
    case 'guardrail_blocked': s.node = 'input_guard'; s.edge = 'guard_block'; s.color = '#fb7185'; s.tone = 'danger'; break;
    case 'run_blocked': s.node = 'blocked'; s.tone = 'danger'; break;
    case 'plan_created': s.edge = 'planner_res'; break;
    case 'search_executed': s.edge = 'web_res'; s.color = '#38bdf8'; break;
    case 'source_indexed': s.edge = p.retrieval_tier === 'uploaded_document' ? 'docs_res' : 'res_index'; if (p.retrieval_tier === 'uploaded_document') s.extraEdge = 'res_index'; break;
    case 'node_started':
      s.edge = { planner: 'guard_planner', researcher: item.iteration > 1 ? null : 'planner_res', drafter: 'index_drafter', verifier: 'drafter_ver', synthesizer: 'router_synth', output_guard: 'synth_output' }[node] || undefined;
      s.text = `${NODE[node]?.label || node} started${item.iteration > 1 ? ` (iteration ${item.iteration})` : ''} — ${item.message}`; break;
    case 'draft_created': s.edge = 'drafter_ver'; break;
    case 'claim_verified': s.color = p.verdict === 'SUPPORTED' ? '#34d399' : p.verdict === 'CONTRADICTED' ? '#fb7185' : '#fbbf24'; s.edge = 'ver_router'; s.tone = p.verdict === 'SUPPORTED' ? 'ok' : 'warn'; break;
    case 'claim_downgraded': s.tone = 'warn'; break;
    case 'contradiction_detected': s.tone = 'danger'; break;
    case 'metrics_computed': s.edge = 'ver_router'; break;
    case 'route_decision': s.node = 'router'; s.edge = p.route === 'researcher' ? 'loop' : 'router_synth'; s.color = p.route === 'researcher' ? '#fbbf24' : '#818cf8'; break;
    case 'loop_back': s.node = 'router'; s.edge = 'loop'; s.color = '#fbbf24'; s.tone = 'loop'; break;
    case 'budget_guard': s.node = 'router'; s.tone = 'guard'; break;
    case 'report_compiled': s.edge = 'synth_output'; break;
    case 'run_completed': s.node = 'report'; s.edge = 'output_report'; s.tone = 'ok'; break;
    case 'run_failed': s.node = 'report'; s.tone = 'danger'; break;
    case 'followup_answered': s.node = 'followup'; s.edge = 'report_followup'; break;
    default: break;
  }
  return s;
}

function buildSteps(events, traces) {
  const items = [
    ...events.map((e) => ({ ...e, kind: 'audit' })),
    ...traces.map((t) => ({ ...t, kind: 'trace', timestamp: t.created_at || t.timestamp })),
  ].sort((a, b) => ts(a) - ts(b) || (a.kind === 'audit' ? -1 : 1));
  return items.map(stepOf);
}

function reduce(k) {
  const S = {
    nodes: Object.fromEntries(NODES.map((n) => [n.id, { status: 'idle', visits: 0, findings: 0 }])),
    edges: Object.fromEntries(EDGES.map((e) => [e.id, { visited: false, label: null }])),
    c: { subq: 0, searches: 0, tiers: {}, sources: 0, uploads: 0, spans: 0, drafts: 0, invalid: 0, claims: 0, supported: 0, contradicted: 0, unsupported: 0, downgraded: 0, ctr: 0, removed: 0, loops: 0, kept: null, redactions: 0, findings: 0, injections: 0, urlsBlocked: 0, calls: 0, tokens: 0, byAgent: {}, events: 0, followups: 0 },
    routes: [], subQuestions: [], status: 'idle', blocked: null, query: '', inputStatus: null, llmScreen: null, budget: null, lastMetrics: null,
  };
  const c = S.c;
  for (let j = 0; j <= k && j < X.steps.length; j++) {
    const st = X.steps[j]; const it = st.item; const p = it.payload || {};
    if (it.kind === 'trace') {
      c.calls += it.operation === 'embedding' ? 0 : 1; c.tokens += it.total_tokens || 0;
      const a = (c.byAgent[it.agent_name] ||= { calls: 0, tokens: 0, ms: 0 }); a.calls++; a.tokens += it.total_tokens || 0; a.ms += it.latency_ms || 0;
      continue;
    }
    c.events++;
    if (st.edge) S.edges[st.edge].visited = true;
    if (st.extraEdge) S.edges[st.extraEdge].visited = true;
    const n = S.nodes[st.node];
    switch (it.event_type) {
      case 'run_started': S.status = 'running'; S.query = p.query || ''; S.nodes.user.status = 'done'; S.edges.user_guard.label = 'question'; break;
      case 'node_started': if (n) { n.status = 'active'; n.visits++; } if (it.node === 'planner') S.nodes.blocked.status = 'skipped'; break;
      case 'node_completed': if (n && n.status !== 'blocked') n.status = 'done'; break;
      case 'guardrail_check':
        if (p.stage === 'input') { S.inputStatus = p.status; S.llmScreen = p.llm_screen; S.edges.guard_planner.label = p.status === 'passed_with_redactions' ? 'redacted' : p.status; }
        if (p.stage === 'output') S.edges.output_report.label = `${c.redactions} redaction${c.redactions === 1 ? '' : 's'}`;
        break;
      case 'guardrail_finding':
        c.findings++; if (n) n.findings++;
        if (p.kind === 'indirect_prompt_injection') c.injections++;
        if (p.guard === 'url_shield') c.urlsBlocked++;
        if (p.action === 'redact') { const m = /^(\d+)/.exec(p.detail || ''); c.redactions += m ? Number(m[1]) : 1; }
        break;
      case 'guardrail_blocked': S.blocked = it.message; S.nodes.input_guard.status = 'blocked'; S.nodes.blocked.status = 'blocked'; S.edges.guard_block.label = 'blocked'; break;
      case 'run_blocked': S.status = 'blocked'; for (const id of ['planner', 'researcher', 'index', 'drafter', 'verifier', 'router', 'synthesizer', 'output_guard', 'report']) S.nodes[id].status = 'skipped'; break;
      case 'plan_created': S.subQuestions = p.sub_questions || []; c.subq = S.subQuestions.length; S.edges.planner_res.label = `${c.subq} sub-questions`; break;
      case 'search_executed': c.searches++; if (p.tier) c.tiers[p.tier] = (c.tiers[p.tier] || 0) + 1; S.edges.web_res.label = `${c.searches} searches`; S.nodes.web.status = 'done'; break;
      case 'source_indexed':
        c.sources++; S.nodes.index.status = 'done';
        if (p.retrieval_tier === 'uploaded_document') { c.uploads++; S.nodes.docs.status = 'done'; S.edges.docs_res.label = `${c.uploads} upload${c.uploads === 1 ? '' : 's'}`; }
        S.edges.res_index.label = `${c.sources} sources`; break;
      case 'refinement_planned': S.edges.loop.label = `refine: ${(p.queries || []).length} queries`; break;
      case 'draft_created': c.drafts++; S.edges.drafter_ver.label = `${c.drafts} drafts`; break;
      case 'citation_invalid': c.invalid += (p.invalid_citations || []).length; break;
      case 'claims_extracted': S.edges.index_drafter.label = 'evidence spans'; break;
      case 'claim_verified':
        c.claims++; if (p.verdict === 'SUPPORTED') c.supported++; else if (p.verdict === 'CONTRADICTED') c.contradicted++; else c.unsupported++;
        S.edges.ver_router.label = `${c.supported}✓ ${c.claims - c.supported}✗`; break;
      case 'claim_downgraded': c.downgraded++; break;
      case 'contradiction_detected': c.ctr++; break;
      case 'metrics_computed': S.lastMetrics = p; S.edges.ver_router.label = `ground ${pct(p.grounding_rate)} · cover ${pct(p.coverage_score)}`; break;
      case 'route_decision':
        S.routes.push(p); S.nodes.router.status = 'done'; S.nodes.router.visits++;
        if (p.route === 'synthesizer') S.edges.router_synth.label = S.budget ? 'budget spent' : (p.coverage_score >= p.coverage_threshold && p.grounding_rate >= p.grounding_threshold ? 'thresholds met' : 'max iterations');
        break;
      case 'loop_back': c.loops++; S.edges.loop.label = `loop ${c.loops}: re-research ${((p.gaps || []).map((g) => g.sub_question_id)).join(', ')}`; break;
      case 'budget_guard': S.budget = p; S.edges.router_synth.label = 'budget spent'; break;
      case 'claim_removed': c.removed++; break;
      case 'report_compiled': c.kept = p.metrics?.claims_in_report ?? null; S.edges.synth_output.label = `${c.kept ?? '?'} kept · ${p.metrics?.claims_removed ?? c.removed} cut`; break;
      case 'run_completed': S.status = 'completed'; S.nodes.report.status = 'done'; S.nodes.audit.status = 'done'; S.nodes.blocked.status = 'skipped'; S.edges.output_audit.visited = true; break;
      case 'run_failed': S.status = 'failed'; Object.values(S.nodes).forEach((x) => { if (x.status === 'active') x.status = 'failed'; }); break;
      case 'followup_answered': c.followups++; S.nodes.followup.status = 'done'; S.edges.report_followup.label = `${c.followups} answered`; break;
      default: break;
    }
  }
  // the node of the current step pulses even if its node_started was earlier
  const cur = X.steps[k];
  if (cur && S.status === 'running' && S.nodes[cur.node] && S.nodes[cur.node].status !== 'blocked') S.nodes[cur.node].status = 'active';
  return S;
}

// ------------------------------------------------------------------ rendering
function fitText(textEl, maxWidth) {
  const full = textEl.dataset.full ?? textEl.textContent;
  textEl.dataset.full = full;
  textEl.textContent = full;
  if (!textEl.getComputedTextLength || textEl.getComputedTextLength() <= maxWidth) return;
  let lo = 0; let hi = full.length;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    textEl.textContent = `${full.slice(0, mid)}…`;
    if (textEl.getComputedTextLength() <= maxWidth) lo = mid; else hi = mid - 1;
  }
  textEl.textContent = `${full.slice(0, lo)}…`;
}

function setFitted(textEl, value, maxWidth) {
  delete textEl.dataset.full;
  textEl.textContent = value;
  fitText(textEl, maxWidth);
}

function el(tag, attrs = {}, parent) {
  const node = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function buildSvg() {
  const svg = $('#archSvg');
  svg.innerHTML = `<defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#475569"/></marker>
    <marker id="arrowOn" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#a5b4fc"/></marker>
    <marker id="arrowLoop" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#fbbf24"/></marker>
    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    <pattern id="dots" width="22" height="22" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r="1" fill="#1b2230"/></pattern>
  </defs>
  <rect x="0" y="0" width="1240" height="700" fill="url(#dots)"/>`;

  // gateway bar + connectors
  const gw = el('g', { class: 'gateway', 'data-node': 'gateway', tabindex: '0' }, svg);
  el('rect', { x: GATEWAY.x, y: GATEWAY.y, width: GATEWAY.w, height: GATEWAY.h, rx: 9, class: 'gw-bar' }, gw);
  el('rect', { x: GATEWAY.x + 2, y: GATEWAY.y + GATEWAY.h - 5, width: 0, height: 3, rx: 1.5, class: 'gw-budget', id: 'gwBudget' }, gw);
  const gwText = el('text', { x: GATEWAY.x + GATEWAY.w / 2, y: GATEWAY.y + 19, 'text-anchor': 'middle', class: 'gw-label', id: 'gwLabel' }, gw);
  gwText.textContent = 'Mesh API gateway · the only LLM egress · retries · JSON repair · model allowlist · budgets';
  for (const [id, d] of Object.entries(CONNECTORS)) el('path', { d, class: 'connector', id: `conn-${id}` }, svg);

  // edges
  for (const e of EDGES) {
    el('path', { d: e.d, id: `edge-${e.id}`, class: `edge edge-${e.kind || 'main'}`, 'marker-end': e.kind === 'loop' ? 'url(#arrowLoop)' : 'url(#arrow)' }, svg);
  }
  // nodes
  for (const n of NODES) {
    const g = el('g', { class: `node kind-${n.kind}`, 'data-node': n.id, tabindex: '0', role: 'button', 'aria-label': `${n.label}: ${n.sub}` }, svg);
    const color = KIND[n.kind].color;
    if (n.kind === 'decision') {
      const { cx, cy } = n;
      el('polygon', { points: `${cx},${cy - 44} ${cx + 54},${cy} ${cx},${cy + 44} ${cx - 54},${cy}`, class: 'node-box' }, g);
      const icon = el('g', { transform: `translate(${cx - 10},${cy - 28}) scale(0.85)`, class: 'node-icon', stroke: color }, g);
      icon.innerHTML = ICONS[n.icon];
      el('text', { x: cx, y: cy + 8, 'text-anchor': 'middle', class: 'node-label' }, g).textContent = n.label;
      el('text', { x: cx, y: cy + 22, 'text-anchor': 'middle', class: 'node-stat', id: `stat-${n.id}` }, g);
      el('text', { x: cx + 26, y: cy + 54, 'text-anchor': 'start', class: 'node-sub' }, g).textContent = n.sub;
      el('circle', { cx: cx + 40, cy: cy - 22, r: 4, class: 'node-dot' }, g);
      el('text', { x: cx - 58, y: cy - 34, class: 'node-num' }, g).textContent = n.n;
      continue;
    }
    const h = n.h || NODE_H;
    el('rect', { x: n.x, y: n.y, width: n.w, height: h, rx: 14, class: 'node-box' }, g);
    el('rect', { x: n.x + 10, y: n.y + (h - 34) / 2, width: 34, height: 34, rx: 9, class: 'node-iconbox', fill: `${color}1f`, stroke: `${color}55` }, g);
    const icon = el('g', { transform: `translate(${n.x + 15},${n.y + (h - 24) / 2}) scale(1)`, class: 'node-icon', stroke: color }, g);
    icon.innerHTML = ICONS[n.icon];
    el('text', { x: n.x + 54, y: n.y + (h === NODE_H ? 22 : 20), class: 'node-label' }, g).textContent = n.label;
    const subEl = el('text', { x: n.x + 54, y: n.y + (h === NODE_H ? 37 : 34), class: 'node-sub', 'data-max': n.w - 62 }, g);
    subEl.textContent = n.sub;
    if (h === NODE_H) el('text', { x: n.x + 54, y: n.y + 52, class: 'node-stat', id: `stat-${n.id}`, 'data-max': n.w - 62 }, g);
    el('circle', { cx: n.x + n.w - 12, cy: n.y + 12, r: 4, class: 'node-dot' }, g);
    if (n.n) el('text', { x: n.x + 2, y: n.y - 7, class: 'node-num' }, g).textContent = n.n;
    el('text', { x: n.x + n.w - 26, y: n.y + 15, class: 'node-mult', id: `mult-${n.id}`, 'text-anchor': 'end' }, g);
    const badge = el('g', { class: 'node-badge', id: `badge-${n.id}`, transform: `translate(${n.x - 8},${n.y - 8})` }, g);
    el('circle', { cx: 9, cy: 9, r: 9 }, badge);
    el('text', { x: 9, y: 12.5, 'text-anchor': 'middle' }, badge);
  }
  // edge labels on top
  for (const e of EDGES) {
    const g = el('g', { class: `edge-label edge-label-${e.kind || 'main'}`, id: `label-${e.id}` }, svg);
    el('rect', { rx: 7, height: 18, y: e.ly - 12 }, g);
    el('text', { x: e.lx, y: e.ly + 1, 'text-anchor': e.anchor === 'start' ? 'start' : 'middle' }, g).textContent = e.label;
  }
  el('g', { id: 'packets' }, svg);
  layoutLabels();
  $$('#archSvg .node-sub[data-max]').forEach((t) => fitText(t, Number(t.dataset.max)));

  svg.addEventListener('mouseover', (ev) => { const g = ev.target.closest('[data-node]'); if (g) inspect(g.dataset.node, false); });
  svg.addEventListener('mouseleave', () => { if (!X.pinned) inspect(null, false); else inspect(X.pinned, false); });
  svg.addEventListener('click', (ev) => { const g = ev.target.closest('[data-node]'); X.pinned = g ? (X.pinned === g.dataset.node ? null : g.dataset.node) : null; inspect(X.pinned || (g && g.dataset.node), true); });
  svg.addEventListener('keydown', (ev) => { const g = ev.target.closest('[data-node]'); if (g && (ev.key === 'Enter' || ev.key === ' ')) { ev.preventDefault(); X.pinned = g.dataset.node; inspect(X.pinned, true); } });
  X.built = true;
}

function layoutLabels() {
  for (const e of EDGES) {
    const g = $(`#label-${e.id}`); if (!g) continue;
    const text = g.querySelector('text'); const rect = g.querySelector('rect');
    const w = Math.max(28, text.getComputedTextLength ? text.getComputedTextLength() + 14 : text.textContent.length * 6.4 + 14);
    rect.setAttribute('width', w);
    rect.setAttribute('x', e.anchor === 'start' ? e.lx - 7 : e.lx - w / 2);
  }
}

function render() {
  if (!X.built) return;
  const S = reduce(X.i);
  X.state = S;
  const c = S.c;
  for (const n of NODES) {
    const g = $(`[data-node="${n.id}"]`); const ns = S.nodes[n.id];
    g.dataset.status = ns.status;
    g.classList.toggle('focused', X.pinned === n.id);
    const badge = $(`#badge-${n.id}`);
    if (badge) { badge.style.display = ns.findings ? '' : 'none'; badge.querySelector('text').textContent = ns.findings; }
    const mult = $(`#mult-${n.id}`);
    if (mult) mult.textContent = ns.visits > 1 && ['researcher', 'drafter', 'verifier'].includes(n.id) ? `×${ns.visits}` : '';
  }
  const stats = {
    user: S.query ? `“${S.query.slice(0, 18)}…”` : '',
    input_guard: S.inputStatus ? `${S.inputStatus.replace('_with_redactions', ' + redact')} · LLM ${S.llmScreen || '-'}` : '',
    planner: c.subq ? `${c.subq} sub-questions` : '',
    researcher: c.searches ? `${c.searches} searches · ${c.sources} src` : '',
    index: c.sources ? `${c.sources} sources indexed` : '',
    drafter: c.drafts ? `${c.drafts} drafts${c.invalid ? ` · ${c.invalid} bad cites` : ''}` : '',
    verifier: c.claims ? `${c.supported}✓ ${c.unsupported}⚠ ${c.contradicted}✗ · ${c.ctr} conflicts` : '',
    router: S.routes.length ? `${S.routes.length} decision${S.routes.length > 1 ? 's' : ''}` : '',
    synthesizer: c.kept !== null ? `${c.kept} kept · ${c.removed} cut` : '',
    output_guard: S.status === 'completed' ? `${c.redactions} redactions` : '',
    report: S.status === 'completed' ? 'published' : S.status === 'blocked' ? 'not produced' : '',
    followup: c.followups ? `${c.followups} answered` : '',
    audit: c.events ? `${c.events} events chained` : '',
  };
  for (const [id, text] of Object.entries(stats)) {
    const t = $(`#stat-${id}`);
    if (t) { if (t.dataset.max) setFitted(t, text, Number(t.dataset.max)); else t.textContent = text; }
  }
  for (const e of EDGES) {
    const es = S.edges[e.id];
    $(`#edge-${e.id}`).classList.toggle('visited', es.visited);
    const label = $(`#label-${e.id}`);
    label.querySelector('text').textContent = es.label || e.label;
    label.classList.toggle('live', !!es.label);
    label.classList.toggle('visited', es.visited);
  }
  layoutLabels();
  // gateway budget bar
  const cap = X.policy?.budgets?.max_llm_calls_per_run || 150;
  $('#gwBudget').setAttribute('width', Math.min(GATEWAY.w - 4, (GATEWAY.w - 4) * (c.calls / cap)));
  $('#gwLabel').textContent = c.calls
    ? `Mesh API gateway · ${c.calls} LLM calls · ${fmt(c.tokens)} tokens · budget ${Math.round((c.calls / cap) * 100)}% of ${cap} calls`
    : 'Mesh API gateway · the only LLM egress · retries · JSON repair · model allowlist · budgets';
  // header/status
  $('#xStatus').textContent = X.steps.length ? `${S.status}${store.live && X.runId === store.runId ? ' · live' : ''}` : 'no run loaded';
  $('#xStatus').dataset.status = S.status;
  $('#xStep').textContent = X.steps.length ? `${Math.max(0, X.i + 1)} / ${X.steps.length}` : '0 / 0';
  $('#xScrub').max = Math.max(0, X.steps.length - 1);
  $('#xScrub').value = Math.max(0, X.i);
  $('#xPlay').textContent = X.playing ? 'Pause' : 'Play';
  const cur = X.steps[X.i];
  $('#xNarration').innerHTML = cur ? `<span class="tl-node" style="--c:${KIND[NODE[cur.node]?.kind || 'io'].color}">${esc(NODE[cur.node]?.label || cur.node)}</span> ${esc(cur.text)}` : 'Press <kbd>Space</kbd> or Play to trace the run step by step.';
  renderTimelineCursor();
  inspect(X.pinned || X.focus, false);
}

// ------------------------------------------------------------------ packets
function firePacket(edgeId, color = '#a5b4fc', reverse = false, isConnector = false) {
  const path = isConnector ? $(`#conn-${edgeId}`) : $(`#edge-${edgeId}`);
  if (!path) return;
  const len = path.getTotalLength();
  const dot = el('circle', { r: isConnector ? 3 : 4.5, fill: color, filter: 'url(#glow)', class: 'packet' }, $('#packets'));
  const duration = Math.max(180, 700 / X.speed);
  const start = performance.now();
  path.classList.add('firing');
  const tick = (now) => {
    const t = Math.min(1, (now - start) / duration);
    const pos = path.getPointAtLength((reverse ? 1 - t : t) * len);
    dot.setAttribute('cx', pos.x); dot.setAttribute('cy', pos.y);
    if (t < 1) requestAnimationFrame(tick); else { dot.remove(); path.classList.remove('firing'); }
  };
  requestAnimationFrame(tick);
}

function animateStep(st) {
  if (!st) return;
  if (st.gateway && CONNECTORS[st.gateway]) { firePacket(st.gateway, st.color, st.gateway === 'synthesizer' || st.gateway === 'followup', true); $('.gw-bar').classList.add('pulse'); setTimeout(() => $('.gw-bar')?.classList.remove('pulse'), 260); return; }
  if (st.edge) firePacket(st.edge, st.color || (st.edge === 'loop' ? '#fbbf24' : '#a5b4fc'));
  if (st.extraEdge) setTimeout(() => firePacket(st.extraEdge, st.color), 200 / X.speed);
  const g = $(`[data-node="${st.node}"]`);
  if (g) { g.classList.remove('flash'); void g.getBBox(); g.classList.add('flash'); }
}

// ------------------------------------------------------------------ transport
function go(i, animate = true) {
  X.i = Math.max(-1, Math.min(X.steps.length - 1, i));
  render();
  if (animate) animateStep(X.steps[X.i]);
}
function play() {
  if (!X.steps.length) { toast('Pick a run first (or start one in the console).', 'info'); return; }
  if (X.i >= X.steps.length - 1) X.i = -1;
  X.playing = true; X.follow = false; tickPlay(); render();
}
function pause() { X.playing = false; clearTimeout(X.timer); render(); }
function tickPlay() {
  clearTimeout(X.timer);
  if (!X.playing) return;
  if (X.i >= X.steps.length - 1) { X.playing = false; render(); return; }
  go(X.i + 1);
  const st = X.steps[X.i];
  X.timer = setTimeout(tickPlay, (st?.dur || 380) / X.speed);
}
function reset() { pause(); go(-1, false); }

// ------------------------------------------------------------------ inspector
function inspect(id, explicit) {
  const panel = $('#xInspector');
  if (!id) { panel.innerHTML = legendHtml(); return; }
  const n = NODE[id]; const info = INFO[id];
  if (!info) return;
  const kind = id === 'gateway' ? { color: '#a78bfa', label: 'LLM gateway' } : KIND[n.kind];
  const S = X.state || reduce(X.i);
  panel.innerHTML = `<div class="fade-in">
    <div class="flex items-center gap-2"><span class="kind-pill" style="--c:${kind.color}">${esc(kind.label)}</span>${n?.n ? `<span class="font-mono text-[10px] text-slate-500">step ${n.n}</span>` : ''}
      ${X.pinned === id ? '<span class="ml-auto text-[10px] text-slate-500">pinned · click again to unpin</span>' : ''}</div>
    <h3 class="mt-2 text-base font-semibold">${esc(id === 'gateway' ? 'Mesh API gateway' : n.label)}</h3>
    <p class="mt-1.5 text-[13px] leading-relaxed text-slate-300">${esc(info.what)}</p>
    <div class="mt-3 text-[11px] uppercase tracking-wider text-slate-500">Contract</div>
    <div class="mt-1 font-mono text-[11.5px] text-indigo-200">${esc(info.io)}</div>
    <div class="mt-3 text-[11px] uppercase tracking-wider text-slate-500">Guardrails & checks here</div>
    <ul class="mt-1 space-y-1 text-[12.5px] text-slate-300">${info.checks.map((c) => `<li class="flex gap-2"><span class="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full" style="background:${kind.color}"></span>${esc(c)}</li>`).join('')}</ul>
    ${liveStatsHtml(id, S)}
    <div class="mt-3 font-mono text-[10.5px] text-slate-500">${esc(info.code)}</div>
  </div>`;
  if (explicit) panel.scrollTop = 0;
}

function liveStatsHtml(id, S) {
  if (!X.steps.length) return '<div class="mt-3 rounded-lg border border-ink-700 p-2.5 text-[12px] text-slate-500">Load a run to see this component\'s live numbers.</div>';
  const c = S.c; const rows = [];
  const row = (k, v) => rows.push(`<div class="flex justify-between gap-3"><span class="text-slate-500">${esc(k)}</span><span class="text-right tabular-nums text-slate-200">${v}</span></div>`);
  const agent = NODE[id]?.agent; const usage = agent && c.byAgent[agent];
  switch (id) {
    case 'user': row('question', esc(S.query.slice(0, 90) || '—')); break;
    case 'input_guard': row('screen', esc(S.inputStatus || '—')); row('Guardian', esc(S.llmScreen || '—')); row('findings', S.nodes.input_guard.findings); if (S.blocked) row('blocked', `<span class="text-rose-300">${esc(S.blocked.slice(0, 120))}</span>`); break;
    case 'planner': S.subQuestions.forEach((q) => row(q.id, esc(q.question.slice(0, 70)))); break;
    case 'researcher': row('passes', S.nodes.researcher.visits); row('searches', c.searches); row('tiers', esc(Object.entries(c.tiers).map(([k, v]) => `${k} ${v}`).join(' · ') || '—')); row('sources', c.sources); row('injections neutralised', c.injections); row('URLs refused', c.urlsBlocked); break;
    case 'index': row('sources indexed', c.sources); row('uploads', c.uploads); break;
    case 'web': row('searches', c.searches); row('URLs refused', c.urlsBlocked); break;
    case 'docs': row('uploads used', c.uploads); break;
    case 'drafter': row('drafts', c.drafts); row('invalid citations', c.invalid); break;
    case 'verifier': row('claims checked', c.claims); row('supported', `<span class="text-emerald-300">${c.supported}</span>`); row('unsupported', `<span class="text-amber-300">${c.unsupported}</span>`); row('contradicted', `<span class="text-rose-300">${c.contradicted}</span>`); row('downgraded by guards', c.downgraded); row('contradictions', c.ctr); break;
    case 'router': S.routes.forEach((r) => row(`iteration ${r.iteration}`, `${pct(r.grounding_rate)} / ${pct(r.coverage_score)} → <b>${esc(r.route)}</b>`)); if (S.budget) row('budget guard', `<span class="text-amber-300">${pct(S.budget.fraction_used)} used</span>`); break;
    case 'synthesizer': row('claims kept', c.kept ?? '—'); row('claims removed', c.removed); break;
    case 'output_guard': row('redactions', c.redactions); row('guardrail findings (run)', c.findings); break;
    case 'report': row('status', esc(S.status)); break;
    case 'followup': row('answered', c.followups); break;
    case 'audit': row('events', c.events); row('integrity', '<button class="link" data-verify>verify chain</button>'); break;
    case 'gateway': row('LLM calls', c.calls); row('tokens', fmt(c.tokens)); Object.entries(c.byAgent).forEach(([a, u]) => row(a, `${u.calls} · ${fmt(u.tokens)} tok`)); break;
    case 'blocked': row('reason', esc(S.blocked || '—')); break;
    default: break;
  }
  if (usage) row('Mesh calls / tokens', `${usage.calls} · ${fmt(usage.tokens)}`);
  return `<div class="mt-3 text-[11px] uppercase tracking-wider text-slate-500">This run · at step ${Math.max(0, X.i + 1)}</div><div class="mt-1 space-y-1 rounded-lg border border-ink-700 bg-ink-900/60 p-2.5 text-[12px]">${rows.join('') || '<span class="text-slate-500">Not reached yet.</span>'}</div>`;
}

function legendHtml() {
  return `<div class="fade-in"><h3 class="text-sm font-semibold">How to read this</h3>
    <p class="mt-1.5 text-[13px] leading-relaxed text-slate-300">This is DeepDive's LangGraph. Hover or click any box to see what it does, the checks applied there, and its numbers for the loaded run. Press <kbd>Space</kbd> to replay the run event by event: every step comes from the hash-chained audit trail. While a run is in progress, the graph shows where it is executing.</p>
    <div class="mt-3 grid grid-cols-2 gap-1.5 text-[11.5px]">${Object.values(KIND).map((k) => `<span class="kind-pill" style="--c:${k.color}">${esc(k.label)}</span>`).join('')}<span class="kind-pill" style="--c:#a78bfa">Mesh gateway</span></div>
    <ul class="mt-3 space-y-1 text-[12px] text-slate-400"><li><span class="legend-dot" style="background:#818cf8"></span>active node · <span class="legend-dot" style="background:#34d399"></span>done · <span class="legend-dot" style="background:#fb7185"></span>blocked/failed</li>
    <li><span class="legend-badge">3</span> guardrail findings at that node</li><li><span class="text-amber-300">amber dashed</span> = the self-correction loop-back</li><li><span class="text-violet-300">purple pulses</span> = Mesh API calls (tokens & latency in the bar)</li></ul></div>`;
}

// ------------------------------------------------------------------ timeline
function renderTimeline() {
  const list = $('#xTimeline');
  if (!X.steps.length) { list.innerHTML = '<div class="p-6 text-center text-sm text-slate-500">No run loaded. Start a run in the Live Console or pick one above.</div>'; return; }
  const t0 = ts(X.steps[0].item);
  list.innerHTML = X.steps.map((s, i) => {
    const node = NODE[s.node]; const color = KIND[node?.kind || 'io'].color;
    return `<button class="tl-row tone-${s.tone}" data-step="${i}"><span class="tl-time">+${((ts(s.item) - t0) / 1000).toFixed(1)}s</span><span class="tl-node" style="--c:${s.item.kind === 'trace' ? '#a78bfa' : color}">${esc(s.item.kind === 'trace' ? 'Mesh' : node?.label || s.node)}</span><span class="tl-text">${esc(s.text)}</span></button>`;
  }).join('');
}
function renderTimelineCursor() {
  $$('.tl-row.current').forEach((r) => r.classList.remove('current'));
  const row = $(`.tl-row[data-step="${X.i}"]`);
  if (row) {
    row.classList.add('current');
    const box = $('#xTimeline');
    if ((X.playing || X.follow) && box) {
      const top = row.offsetTop - box.offsetTop;
      if (top < box.scrollTop || top + row.offsetHeight > box.scrollTop + box.clientHeight) box.scrollTop = top - box.clientHeight / 2;
    }
  }
}

// ------------------------------------------------------------------ loading runs
async function loadRunList() {
  try {
    const runs = await api('/api/runs?limit=40');
    const sel = $('#xRun');
    const current = X.runId || store.runId;
    sel.innerHTML = '<option value="">Pick a run to trace…</option>' + runs.map((r) => `<option value="${esc(r.id)}" ${r.id === current ? 'selected' : ''}>[${esc(r.status)}] ${esc(r.query.slice(0, 80))}</option>`).join('');
  } catch (_) { /* auth modal handles 401 */ }
}

export async function loadRun(runId, { jumpToEnd = false } = {}) {
  if (!runId) return;
  pause();
  X.runId = runId;
  let events; let traces;
  if (store.runId === runId && store.events.length) { events = store.events; traces = store.traces; }
  else {
    const [audit, tr] = await Promise.all([api(`/api/runs/${runId}/audit`), api(`/api/runs/${runId}/traces`)]);
    events = audit; traces = tr.traces;
  }
  X.steps = buildSteps(events, traces);
  renderTimeline();
  go(jumpToEnd ? X.steps.length - 1 : -1, false);
  $('#xRun').value = runId;
}

function onLiveItem() {
  if (X.runId !== store.runId) return;
  X.steps = buildSteps(store.events, store.traces);
  renderTimeline();
  if (X.follow && !X.playing) go(X.steps.length - 1);
  else render();
}

// ------------------------------------------------------------------ init
export function initExplorer() {
  buildSvg();
  inspect(null);
  api('/api/governance/policy').then((p) => { X.policy = p; render(); }).catch(() => {});
  $('#xPlay').addEventListener('click', () => (X.playing ? pause() : play()));
  $('#xPrev').addEventListener('click', () => { pause(); go(X.i - 1); });
  $('#xNext').addEventListener('click', () => { pause(); go(X.i + 1); });
  $('#xReset').addEventListener('click', reset);
  $('#xSpeed').addEventListener('change', (e) => { X.speed = Number(e.target.value); });
  $('#xScrub').addEventListener('input', (e) => { pause(); X.follow = false; go(Number(e.target.value)); });
  $('#xFollow').addEventListener('change', (e) => { X.follow = e.target.checked; if (X.follow) go(X.steps.length - 1); });
  $('#xRun').addEventListener('change', (e) => loadRun(e.target.value));
  $('#xRefresh').addEventListener('click', loadRunList);
  $('#xTimeline').addEventListener('click', (e) => { const r = e.target.closest('[data-step]'); if (r) { pause(); X.follow = false; go(Number(r.dataset.step)); } });
  $('#xInspector').addEventListener('click', async (e) => {
    if (!e.target.closest('[data-verify]') || !X.runId) return;
    try { const v = await api(`/api/runs/${X.runId}/audit/verify`); toast(v.valid ? `Audit chain intact: ${v.verified} events, head ${v.head.slice(0, 12)}…` : `Audit chain BROKEN at ${v.broken_at}`, v.valid ? 'ok' : 'error'); }
    catch (err) { toast(err.message); }
  });
  document.addEventListener('keydown', (e) => {
    if ($('#view-explorer').classList.contains('hidden') || ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return;
    if (e.code === 'Space') { e.preventDefault(); X.playing ? pause() : play(); }
    if (e.key === 'ArrowRight') { pause(); go(X.i + 1); }
    if (e.key === 'ArrowLeft') { pause(); go(X.i - 1); }
    if (e.key === 'r' || e.key === 'R') reset();
    if (e.key === 'Escape') { X.pinned = null; inspect(null); render(); }
  });
  on('run-opened', ({ runId }) => { X.runId = runId; X.follow = true; $('#xFollow').checked = true; X.steps = []; renderTimeline(); render(); loadRunList(); });
  on('run-events', () => onLiveItem());
  on('run-loaded', ({ runId }) => { if (X.runId === runId || !X.runId) loadRun(runId, { jumpToEnd: true }); loadRunList(); });
  loadRunList();
  render();
}

export function explorerShown(params) {
  loadRunList();
  // SVG text can only be measured once the view is visible (display:none measures 0px)
  requestAnimationFrame(() => {
    layoutLabels();
    $$('#archSvg .node-sub[data-max]').forEach((t) => fitText(t, Number(t.dataset.max)));
    render();
  });
  if (params?.run && params.run !== X.runId) {
    loadRun(params.run).then(() => {
      if (params.step) go(Number(params.step) - 1);
      if (params.focus) { X.pinned = params.focus; inspect(params.focus, true); render(); }
    }).catch((e) => toast(e.message));
  } else if (params?.focus) { X.pinned = params.focus; render(); }
  else if (!X.runId && store.runId) loadRun(store.runId, { jumpToEnd: true });
}
