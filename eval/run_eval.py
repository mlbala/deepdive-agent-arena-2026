"""DeepDive self-evaluation suite.

    python3 -m eval.run_eval                       # Mesh LLM if MESH_API_KEY is set, offline corpus retrieval
    python3 -m eval.run_eval --retrieval live      # Tavily/DDG web retrieval instead of the offline corpus
    python3 -m eval.run_eval --llm simulated       # pipeline dry run with the deterministic simulator
    python3 -m eval.run_eval --update-readme       # also refresh the README results table
    python3 -m eval.run_eval --render-only --update-readme   # re-render README from latest_results.json
    python3 -m eval.run_eval --ablation --update-readme      # what each layer adds (see ABLATIONS)

Metrics
-------
claim_groundedness_pre   share of first-draft atomic claims SUPPORTED by their cited source (iteration 1)
claim_groundedness_post  share of claims in the FINAL REPORT that an independent re-audit pass
                         (fresh decomposition + entailment against cited sources) finds SUPPORTED
citation_precision       share of cited report claims whose cited source entails them (re-audit)
quote_verification_rate  share of verifier evidence quotes located verbatim in the cited source
contradiction_recall     share of contradiction-bearing cases where the radar surfaced a conflict
                         involving the expected low-authority domain (offline) / any conflict (live)
contradiction_fpr        share of contradiction-free control cases where a conflict was surfaced
loop_back_efficacy       share of looped runs whose grounding or coverage improved after loop-back
trap_pass_rate           false-premise and thin-evidence traps handled correctly
key_fact_recall          share of expected key facts present in the report narrative

report_sentence_support  share of cited sentences in the whole published narrative (summary, findings,
                         sections) that an independent sentence-level re-check finds SUPPORTED

Ablation (--ablation) runs the same cases with one layer switched off at a time (the Report
Approver, the Verifier -> Drafter quick fixes, the loop-backs) plus a plain-AI row: one plain
Mesh answer per question, checked by the Verifier against the full run's evidence. Results go
to eval/ablation_results.json.

Results are written to eval/latest_results.json. Runs produced with the simulator are
labelled `"mode": "simulated"` and must not be read as Mesh-model quality metrics.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from src.agents.base import EventCollector
from src.agents.orchestrator import ResearchOrchestrator
from src.agents.report_approver import ReportApproverAgent
from src.core.config import Settings
from src.core.mesh_gateway import MeshGateway
from src.schemas.domain import DraftSection, FinalReport, SourceDocument, Verdict, extract_citations
from src.storage.database import AuditStore
from src.storage.vector_store import HybridVectorStore
from src.tools.web_search import WebRetriever

EVAL_DIR = Path(__file__).resolve().parent
BENCHMARK_PATH = EVAL_DIR / "benchmark_questions.json"
RESULTS_PATH = EVAL_DIR / "latest_results.json"
ABLATION_PATH = EVAL_DIR / "ablation_results.json"
README_PATH = EVAL_DIR.parent / "README.md"
README_START, README_END = "<!-- EVAL:START -->", "<!-- EVAL:END -->"
ABLATION_START, ABLATION_END = "<!-- ABLATION:START -->", "<!-- ABLATION:END -->"

# configuration id -> (label, settings overrides): each switches exactly one layer off
ABLATIONS: dict[str, tuple[str, dict[str, Any]]] = {
    "full": ("Full DeepDive", {}),
    "no_report_check": ("Without the Report Approver", {"report_audit": False}),
    "no_quick_fixes": ("Without Verifier -> Drafter quick fixes", {"group_chat_exchanges": 0}),
    "single_pass": ("Without loop-backs (one research round)", {"max_iterations": 1, "master_extra_rounds": 0}),
}


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def build_orchestrator(settings: Settings, llm_mode: str) -> ResearchOrchestrator:
    client: Any | None = None
    if llm_mode == "simulated":
        from eval.simulated_llm import HeuristicMeshSimulator

        client = HeuristicMeshSimulator()
    store = AuditStore("sqlite:///:memory:")
    gateway = MeshGateway(settings, client=client, audit_store=store)
    return ResearchOrchestrator(
        settings, gateway, WebRetriever(settings), HybridVectorStore(settings, gateway), store=store
    )


def resolve_llm_mode(requested: str, settings: Settings) -> str:
    if requested == "auto":
        if settings.mesh_configured:
            return "mesh"
        print(
            "WARNING: MESH_API_KEY not set -> falling back to the deterministic simulator. "
            "Results will be labelled 'simulated' and are NOT Mesh-model metrics.",
            file=sys.stderr,
        )
        return "simulated"
    if requested == "mesh" and not settings.mesh_configured:
        raise SystemExit("--llm mesh requires MESH_API_KEY")
    return requested


# ---------------------------------------------------------------------------
# Per-question scoring
# ---------------------------------------------------------------------------
def _narrative(report: FinalReport) -> str:
    return " ".join([report.executive_summary, *report.key_findings, *(s.body for s in report.sections)])


def _contains_any(text: str, alternatives: list[str]) -> bool:
    lowered = text.lower()
    return any(alt.lower() in lowered for alt in alternatives)


def independent_reaudit(
    orchestrator: ResearchOrchestrator, run_id: str, report: FinalReport, sources: dict[str, SourceDocument]
) -> dict[str, Any]:
    """Fresh decomposition + entailment of the FINAL report text (not the pipeline's own verdicts)."""
    verifier = orchestrator.verifier
    collector = EventCollector(run_id, "eval-reaudit", 0)
    by_id = {sq.id: sq for sq in report.sub_questions}
    supported = total = cited = cited_supported = 0
    for section in report.sections:
        if not section.claim_ids or section.sub_question_id not in by_id:
            continue
        draft = DraftSection(
            sub_question_id=section.sub_question_id,
            iteration=99,
            text=section.body,
            cited_source_ids=extract_citations(section.body),
        )
        sq = by_id[section.sub_question_id]
        claims = verifier.extract_claims(run_id, draft, sq, collector)
        results = {r.claim_id: r for r in verifier.judge_claims(run_id, sq, claims, sources, collector)}
        for claim in claims:
            ok = results[claim.claim_id].verdict is Verdict.SUPPORTED
            total += 1
            supported += ok
            if claim.cited_source_ids:
                cited += 1
                cited_supported += ok
    return {
        "reaudit_claims": total,
        "claim_groundedness_post": round(supported / total, 4) if total else None,
        "citation_precision": round(cited_supported / cited, 4) if cited else None,
    }


def sentence_reaudit(
    orchestrator: ResearchOrchestrator, run_id: str, report: FinalReport, sources: dict[str, SourceDocument]
) -> dict[str, Any]:
    """Every cited sentence of the published narrative, re-judged from scratch (no reused verdicts)."""
    checker = ReportApproverAgent(orchestrator.gateway, orchestrator.settings, orchestrator.vector_store)
    audit = checker.audit(run_id, report, sources, audit_round=1)
    return {
        "report_sentences_judged": audit.judged,
        "report_sentences_unsupported": audit.failed,
        "report_sentence_support": audit.groundedness if audit.judged else None,
    }


def score_question(
    case: dict[str, Any], state: dict[str, Any], reaudit: dict[str, Any], *, offline: bool, wall_ms: int
) -> dict[str, Any]:
    report: FinalReport = state["final_report"]
    m = report.metrics
    history = state["iteration_metrics"]
    sources: dict[str, SourceDocument] = state["sources"]
    narrative = _narrative(report)
    cross = [c for c in report.contradictions if c.origin == "cross_source"]

    def domains_of(c: Any) -> set[str]:
        return {sources[s].domain for s in (c.source_a, c.source_b) if s in sources}

    if offline and case["conflict_domains"]:
        surfaced = any(domains_of(c) & set(case["conflict_domains"]) for c in cross)
    else:
        surfaced = bool(cross)

    facts = case.get("key_facts", [])
    fact_hits = sum(_contains_any(narrative, alts) for alts in facts)

    trap_pass: bool | None = None
    trap_detail = ""
    if case.get("trap") == "false_premise":
        premise_domains = set(case.get("premise_source_domains", []))
        premise_ids = {sid for sid, s in sources.items() if s.domain in premise_domains}
        cited_in_findings = set(extract_citations(" ".join([report.executive_summary, *report.key_findings])))
        premise_asserted = bool(premise_ids & cited_in_findings)
        trap_pass = surfaced and not premise_asserted
        trap_detail = f"myth contradicted={surfaced}, myth cited in summary/findings={premise_asserted}"
    elif case.get("trap") == "thin_evidence":
        rumor_as_fact = any(
            marker.lower() in sentence.lower() and "conflict" not in sentence.lower()
            for marker in case.get("rumor_markers", [])
            for sentence in re.split(r"(?<=[.!?])\s+", narrative)
        )
        acknowledged = (
            m.coverage_score < 1.0
            or bool(report.limitations)
            or "insufficient" in narrative.lower()
        )
        trap_pass = (not rumor_as_fact) and acknowledged
        trap_detail = f"rumor stated as fact={rumor_as_fact}, insufficiency acknowledged={acknowledged}"

    looped = len(history) > 1
    return {
        "id": case["id"],
        "domain": case["domain"],
        "trap": case.get("trap"),
        "question": case["question"],
        "run_id": state["run_id"],
        "iterations": len(history),
        "loop_backs": max(0, len(history) - 1),
        "claim_groundedness_pre": (history[0].first_check_grounding if history[0].first_check_grounding is not None
                                   else history[0].grounding_rate) if history else None,
        "quick_fixes": sum(m.exchanges for m in history),
        "grounding_final_iteration": history[-1].grounding_rate if history else None,
        "coverage_pre": history[0].coverage_score if history else None,
        "coverage_final": m.coverage_score,
        "loop_improved": (
            looped
            and (history[-1].grounding_rate > history[0].grounding_rate or history[-1].coverage_score > history[0].coverage_score)
        ),
        "quote_verification_rate": m.quote_verification_rate,
        **reaudit,
        "claims_evaluated": m.total_claims_evaluated,
        "claims_in_report": m.claims_in_report,
        "claims_removed": m.claims_removed,
        "expected_contradiction": case["expected_contradiction"],
        "contradiction_surfaced": surfaced,
        "contradictions_total": len(report.contradictions),
        "report_check_fixed": m.report_sentences_fixed,
        "report_check_removed": m.report_sentences_removed,
        "key_fact_recall": round(fact_hits / len(facts), 4) if facts else None,
        "trap_pass": trap_pass,
        "trap_detail": trap_detail,
        "sources_indexed": m.sources_indexed,
        "llm_calls": m.llm_calls,
        "total_tokens": m.total_tokens,
        "wall_clock_ms": wall_ms,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(key: str, subset: list[dict[str, Any]] | None = None) -> float | None:
        values = [r[key] for r in (subset if subset is not None else rows) if r.get(key) is not None]
        return round(mean(values), 4) if values else None

    positives = [r for r in rows if r["expected_contradiction"]]
    controls = [r for r in rows if not r["expected_contradiction"] and r["trap"] is None]
    looped = [r for r in rows if r["loop_backs"] > 0]
    traps = [r for r in rows if r["trap"]]
    return {
        "questions": len(rows),
        "claim_groundedness_pre": avg("claim_groundedness_pre"),
        "claim_groundedness_post": avg("claim_groundedness_post"),
        "citation_precision": avg("citation_precision"),
        "quote_verification_rate": avg("quote_verification_rate"),
        "coverage_final": avg("coverage_final"),
        "contradiction_recall": round(sum(r["contradiction_surfaced"] for r in positives) / len(positives), 4) if positives else None,
        "contradiction_fpr": round(sum(r["contradiction_surfaced"] for r in controls) / len(controls), 4) if controls else None,
        "loop_back_rate": round(len(looped) / len(rows), 4) if rows else None,
        "loop_back_efficacy": round(sum(r["loop_improved"] for r in looped) / len(looped), 4) if looped else None,
        "mean_grounding_gain_when_looped": (
            round(mean(r["grounding_final_iteration"] - r["claim_groundedness_pre"] for r in looped), 4) if looped else None
        ),
        "trap_pass_rate": round(sum(bool(r["trap_pass"]) for r in traps) / len(traps), 4) if traps else None,
        "key_fact_recall": avg("key_fact_recall"),
        "claims_removed_total": sum(r["claims_removed"] for r in rows),
        "mean_llm_calls": avg("llm_calls"),
        "total_tokens": sum(r["total_tokens"] for r in rows),
        "mean_wall_clock_s": round(mean(r["wall_clock_ms"] for r in rows) / 1000, 2) if rows else None,
    }


def plain_row(case: dict[str, Any], orchestrator: ResearchOrchestrator, run_id: str) -> dict[str, Any]:
    """The plain-AI baseline on the same question, checked against the full run's evidence."""
    c = orchestrator.compare_baseline(run_id)
    facts = case.get("key_facts", [])
    rumor_as_fact = any(marker.lower() in c.answer.lower() for marker in case.get("rumor_markers", []))
    return {
        "id": case["id"],
        "trap": case.get("trap"),
        "claims": len(c.claims),
        "traceable_rate": c.traceable_rate,
        "contradicted": c.contradicted,
        "unsupported": c.unsupported + c.not_found,
        "with_figures": c.with_figures,
        "key_fact_recall": round(sum(_contains_any(c.answer, alts) for alts in facts) / len(facts), 4) if facts else None,
        "trap_pass": (not rumor_as_fact) if case.get("trap") == "thin_evidence" else None,
        "llm_calls": c.llm_calls,
        "total_tokens": c.total_tokens,
        "answer": c.answer,
    }


def run_case(settings: Settings, llm_mode: str, case: dict[str, Any], *, offline: bool, plain: bool) -> dict[str, Any]:
    orchestrator = build_orchestrator(settings, llm_mode)
    started = time.perf_counter()
    state: dict[str, Any] = dict(orchestrator.run(case["question"]))
    wall_ms = round((time.perf_counter() - started) * 1000)
    report: FinalReport = state["final_report"]
    sources: dict[str, SourceDocument] = state["sources"]
    reaudit = independent_reaudit(orchestrator, state["run_id"], report, sources)
    row = score_question(case, state, {**reaudit, **sentence_reaudit(orchestrator, state["run_id"], report, sources)},
                         offline=offline, wall_ms=wall_ms)
    if plain:
        row["plain"] = plain_row(case, orchestrator, state["run_id"])
    return row


def aggregate_ablation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(key: str) -> float | None:
        values = [r[key] for r in rows if r.get(key) is not None]
        return round(mean(values), 4) if values else None

    traps = [r for r in rows if r["trap"]]
    positives = [r for r in rows if r["expected_contradiction"]]
    return {
        "questions": len(rows),
        "report_sentence_support": avg("report_sentence_support"),
        "unsupported_sentences_published": sum(r.get("report_sentences_unsupported") or 0 for r in rows),
        "claim_groundedness_post": avg("claim_groundedness_post"),
        "coverage_final": avg("coverage_final"),
        "key_fact_recall": avg("key_fact_recall"),
        "trap_pass_rate": round(sum(bool(r["trap_pass"]) for r in traps) / len(traps), 4) if traps else None,
        "contradiction_recall": round(sum(r["contradiction_surfaced"] for r in positives) / len(positives), 4) if positives else None,
        "mean_iterations": avg("iterations"),
        "quick_fixes": sum(r["quick_fixes"] for r in rows),
        "report_check_fixes": sum(r.get("report_check_fixed", 0) + r.get("report_check_removed", 0) for r in rows),
        "mean_llm_calls": avg("llm_calls"),
        "mean_tokens": avg("total_tokens"),
        "mean_wall_clock_s": round(mean(r["wall_clock_ms"] for r in rows) / 1000, 1) if rows else None,
    }


def aggregate_plain(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(key: str) -> float | None:
        values = [r[key] for r in rows if r.get(key) is not None]
        return round(mean(values), 4) if values else None

    traps = [r for r in rows if r["trap_pass"] is not None]
    return {
        "questions": len(rows),
        "traceable_rate": avg("traceable_rate"),
        "claims": sum(r["claims"] for r in rows),
        "contradicted": sum(r["contradicted"] for r in rows),
        "unsupported": sum(r["unsupported"] for r in rows),
        "key_fact_recall": avg("key_fact_recall"),
        "trap_pass_rate": round(sum(bool(r["trap_pass"]) for r in traps) / len(traps), 4) if traps else None,
        "mean_llm_calls": avg("llm_calls"),
        "mean_tokens": avg("total_tokens"),
    }


def ablation_table(results: dict[str, Any]) -> str:
    backend = results["llm_backend"]
    lines = [
        f"_Generated {results['generated_at']} · LLM backend: **{backend['mode']}** (`{backend['model']}`) · "
        f"retrieval: **{results['retrieval']}** · {results['questions']} benchmark questions per configuration_",
        "",
    ]
    if backend["mode"] != "mesh":
        lines += [f"> **Warning:** {backend['note']}", ""]
    lines += [
        "| Configuration | Report sentences supported (independent re-check) | Unsupported sentences published | "
        "Coverage | Key facts | Traps passed | Rounds | Fixes made | Mesh calls / run | Tokens / run |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cid, (label, _overrides) in ABLATIONS.items():
        a = results["configs"].get(cid)
        if not a:
            continue
        fixes = f"{a['quick_fixes']} quick · {a['report_check_fixes']} report"
        lines.append(
            f"| **{label}** | {_fmt(a['report_sentence_support'])} | {a['unsupported_sentences_published']} | "
            f"{_fmt(a['coverage_final'])} | {_fmt(a['key_fact_recall'])} | {_fmt(a['trap_pass_rate'])} | "
            f"{_fmt(a['mean_iterations'])} | {fixes} | {_fmt(a['mean_llm_calls'])} | {_fmt_int(a['mean_tokens'])} |"
        )
    p = results.get("plain")
    if p:
        lines.append(
            f"| **Plain AI answer** (one Mesh call, no search) | {_fmt(p['traceable_rate'])} of claims traceable¹ | "
            f"{p['contradicted'] + p['unsupported']} of {p['claims']} claims¹ | n/a | {_fmt(p['key_fact_recall'])} | "
            f"{_fmt(p['trap_pass_rate'])}² | - | - | 1 (+ check) | {_fmt_int(p['mean_tokens'])} incl. check |"
        )
        lines += ["", "¹ Plain answers carry no citations, so their claims are checked against the full run's evidence "
                  "pool (the closest sources per claim); unit: atomic claims, not report sentences. "
                  "² Only the thin-evidence trap can be scored without sources (the rumour stated as fact or not)."]
    return "\n".join(lines)


def _fmt_int(value: float | None) -> str:
    return "n/a" if value is None else f"{round(value):,}"


def run_ablation(args: argparse.Namespace, settings_base: dict[str, Any]) -> dict[str, Any]:
    cases = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))["questions"]
    if args.questions:
        cases = [c for c in cases if c["id"] in set(args.questions)]
    llm_mode = resolve_llm_mode(args.llm, Settings(**settings_base))
    configs = [c for c in ABLATIONS if not args.configs or c in args.configs]
    jobs = [(cid, case) for cid in configs for case in cases]

    def job(item: tuple[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        cid, case = item
        settings = Settings(**{**settings_base, **ABLATIONS[cid][1]})
        row = run_case(settings, llm_mode, case, offline=args.retrieval == "offline", plain=cid == "full" and not args.no_plain)
        print(f"  [{cid}] {case['id']}: rounds={row['iterations']} sentences={_fmt(row['report_sentence_support'])} "
              f"coverage={_fmt(row['coverage_final'])} ({row['wall_clock_ms'] / 1000:.0f}s)", flush=True)
        return cid, row

    print(f"Ablation: {len(configs)} configuration(s) x {len(cases)} question(s), {args.workers} worker(s)", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        done = list(pool.map(job, jobs))
    by_config: dict[str, list[dict[str, Any]]] = {cid: [] for cid in configs}
    for cid, row in done:
        by_config[cid].append(row)
    plain_rows = [r["plain"] for r in by_config.get("full", []) if "plain" in r]
    sample = Settings(**settings_base)
    model = "offline-heuristic-simulator" if llm_mode == "simulated" else sample.mesh_default_model
    results = {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "llm_backend": {
            "mode": llm_mode, "model": model, "gateway": sample.mesh_base_url,
            "note": ("Real Mesh API completions; every LLM and embedding call routed through Mesh." if llm_mode == "mesh"
                     else "Dry run with the deterministic offline simulator: validates the harness, NOT Mesh-model quality."),
        },
        "retrieval": "offline curated corpus" if args.retrieval == "offline" else "live web (Tavily/DuckDuckGo)",
        "questions": len(cases),
        "configs": {cid: aggregate_ablation(rows) for cid, rows in by_config.items()},
        "plain": aggregate_plain(plain_rows) if plain_rows else None,
        "runs": by_config,
    }
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    table = ablation_table(results)
    print("\n" + table + "\n")
    print(f"Results written to {args.output}")
    if args.update_readme:
        print("README updated." if update_readme(table, markers=(ABLATION_START, ABLATION_END)) else "README markers not found; skipped.")
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value * 100:.0f}%" if 0 <= value <= 1 else f"{value:.2f}"
    return str(value)


def markdown_tables(results: dict[str, Any]) -> str:
    agg = results["aggregate"]
    backend = results["llm_backend"]
    lines = [
        f"_Generated {results['generated_at']} · LLM backend: **{backend['mode']}** (`{backend['model']}`) · "
        f"retrieval: **{results['retrieval']}** · max iterations {results['settings']['max_iterations']}_",
        "",
    ]
    if backend["mode"] != "mesh":
        lines += [f"> **Warning:** {backend['note']}", ""]
    lines += [
        "| Metric | Value |",
        "| --- | --- |",
        f"| Claim groundedness, first draft (pre-verification) | {_fmt(agg['claim_groundedness_pre'])} |",
        f"| Claim groundedness, final report (independent re-audit) | {_fmt(agg['claim_groundedness_post'])} |",
        f"| Citation precision (re-audit) | {_fmt(agg['citation_precision'])} |",
        f"| Evidence quotes located verbatim | {_fmt(agg['quote_verification_rate'])} |",
        f"| Contradiction recall | {_fmt(agg['contradiction_recall'])} |",
        f"| Contradiction false-positive rate (control) | {_fmt(agg['contradiction_fpr'])} |",
        f"| Loop-back rate / efficacy | {_fmt(agg['loop_back_rate'])} / {_fmt(agg['loop_back_efficacy'])} |",
        f"| Trap pass rate (false premise + thin evidence) | {_fmt(agg['trap_pass_rate'])} |",
        f"| Key-fact recall | {_fmt(agg['key_fact_recall'])} |",
        f"| Claims stripped before publication | {agg['claims_removed_total']} |",
        f"| Mean Mesh calls / run · total tokens | {_fmt(agg['mean_llm_calls'])} · {agg['total_tokens']:,} |",
        "",
        "| Case | Trap | Iter. | Grounding pre → re-audit | Coverage | Contradiction (expected → surfaced) | Trap | Key facts |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results["questions"]:
        lines.append(
            f"| `{r['id']}` | {r['trap'] or '-'} | {r['iterations']} | {_fmt(r['claim_groundedness_pre'])} → "
            f"{_fmt(r['claim_groundedness_post'])} | {_fmt(r['coverage_final'])} | "
            f"{_fmt(r['expected_contradiction'])} → {_fmt(r['contradiction_surfaced'])} | "
            f"{_fmt(r['trap_pass']) if r['trap'] else '-'} | {_fmt(r['key_fact_recall'])} |"
        )
    return "\n".join(lines)


def update_readme(table: str, *, markers: tuple[str, str] = (README_START, README_END)) -> bool:
    start, end = markers
    if not README_PATH.exists():
        return False
    text = README_PATH.read_text(encoding="utf-8")
    if start not in text or end not in text:
        return False
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    README_PATH.write_text(f"{head}{start}\n{table}\n{end}{tail}", encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="DeepDive self-evaluation suite")
    parser.add_argument("--llm", choices=["auto", "mesh", "simulated"], default="auto")
    parser.add_argument("--retrieval", choices=["offline", "live"], default="offline")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--questions", nargs="*", help="subset of benchmark ids")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--ablation", action="store_true", help="run every ABLATIONS configuration plus the plain-AI baseline")
    parser.add_argument("--configs", nargs="*", choices=list(ABLATIONS), help="ablation subset")
    parser.add_argument("--no-plain", action="store_true", help="skip the plain-AI baseline in an ablation")
    parser.add_argument("--workers", type=int, default=1, help="parallel runs in an ablation")
    parser.add_argument("--update-readme", action="store_true")
    parser.add_argument(
        "--render-only", action="store_true", help="re-render the README table from existing results (no runs)"
    )
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = ABLATION_PATH if args.ablation else RESULTS_PATH

    if args.render_only:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        table = ablation_table(existing) if args.ablation else markdown_tables(existing)
        markers = (ABLATION_START, ABLATION_END) if args.ablation else (README_START, README_END)
        print(table)
        if args.update_readme:
            print("README updated." if update_readme(table, markers=markers) else "README markers not found; skipped.")
        return existing

    overrides: dict[str, Any] = {"database_url": "sqlite:///:memory:"}
    if args.retrieval == "offline":
        overrides["search_mode"] = "offline"
    if args.max_iterations:
        overrides["max_iterations"] = args.max_iterations
    if args.ablation:
        return run_ablation(args, overrides)
    settings = Settings(**overrides)
    llm_mode = resolve_llm_mode(args.llm, settings)

    cases = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))["questions"]
    if args.questions:
        cases = [c for c in cases if c["id"] in set(args.questions)]

    rows: list[dict[str, Any]] = []
    for i, case in enumerate(cases, start=1):
        orchestrator = build_orchestrator(settings, llm_mode)
        print(f"[{i}/{len(cases)}] {case['id']}: {case['question']}", flush=True)
        started = time.perf_counter()
        state = orchestrator.run(case["question"])
        wall_ms = round((time.perf_counter() - started) * 1000)
        reaudit = independent_reaudit(orchestrator, state["run_id"], state["final_report"], state["sources"])
        row = score_question(case, state, reaudit, offline=args.retrieval == "offline", wall_ms=wall_ms)
        rows.append(row)
        print(
            f"      iterations={row['iterations']} grounding {_fmt(row['claim_groundedness_pre'])} -> "
            f"{_fmt(row['claim_groundedness_post'])} coverage={_fmt(row['coverage_final'])} "
            f"contradiction={_fmt(row['contradiction_surfaced'])} trap={_fmt(row['trap_pass'])} "
            f"({wall_ms / 1000:.1f}s)",
            flush=True,
        )

    model = (
        "offline-heuristic-simulator"
        if llm_mode == "simulated"
        else f"{settings.mesh_default_model} (verifier: {settings.model_for('verifier')})"
    )
    results = {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "llm_backend": {
            "mode": llm_mode,
            "model": model,
            "gateway": settings.mesh_base_url,
            "note": (
                "Real Mesh API completions; every LLM and embedding call routed through Mesh."
                if llm_mode == "mesh"
                else "Pipeline dry run with the deterministic offline simulator (eval/simulated_llm.py). "
                "These numbers validate the harness and the deterministic guards, NOT Mesh-model quality. "
                "Re-run `python3 -m eval.run_eval` with MESH_API_KEY set to record real metrics."
            ),
        },
        "retrieval": "offline curated corpus" if args.retrieval == "offline" else "live web (Tavily/DuckDuckGo)",
        "settings": {
            "max_iterations": settings.max_iterations,
            "coverage_threshold": settings.coverage_threshold,
            "grounding_threshold": settings.grounding_threshold,
            "vector_backend": settings.vector_backend,
        },
        "aggregate": aggregate(rows),
        "questions": rows,
    }
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    table = markdown_tables(results)
    print("\n" + table + "\n")
    print(f"Results written to {args.output}")
    if args.update_readme:
        print("README updated." if update_readme(table) else "README markers not found; skipped.")
    return results


if __name__ == "__main__":
    main()
