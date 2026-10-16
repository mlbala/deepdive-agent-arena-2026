"""Deterministic, prompt-driven stand-in for a Mesh chat model.

`HeuristicMeshSimulator` exposes the same surface the gateway uses on an
`openai.OpenAI` client (`chat.completions.create`, `embeddings.create`). It reads the
`ROLE:` line of each agent's system prompt and the agent's JSON input block, and computes
a response *from that input* with transparent lexical heuristics (extractive drafting,
token-overlap entailment, polarity-based contradiction spotting).

Purpose: hermetic tests of the full LangGraph pipeline and an offline *dry run* of the
evaluation harness. It is NOT a language model and its scores are NOT Mesh-model metrics —
`eval/run_eval.py` labels any results produced with it accordingly.

`hallucinate_first_draft=True` makes the drafter append one fabricated, cited sentence on
first drafts (no verifier feedback present), which exercises the verifier's rejection path
and the loop-back edge end-to-end.

`reject_first_plan=True` makes the Team Lead raise a blocking issue on the first plan, which
exercises the Planner <-> Team Lead revision loop.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from types import SimpleNamespace
from typing import Any

INPUT_MARKER = "INPUT (JSON):"
_STOP = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in into is
    it its may might more most of on or our over so such than that the their them then there these they
    this those through to was we were what when where which while who why will with would you your about
    after also any between both during each other own same should some very""".split()
)
_NEGATIONS = frozenset(
    "not no never fell fall falling declined decline decreased stalled cannot insufficient disappointing".split()
)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_CITE = re.compile(r"\s*\[[^\]]*SRC[^\]]*\]")


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()) if t not in _STOP and len(t) > 1]


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text or "") if len(s.strip().split()) >= 5]


def _overlap(claim: str, sentence: str) -> float:
    c, s = set(_tokens(claim)), set(_tokens(sentence))
    return len(c & s) / len(c) if c else 0.0


_UP = frozenset("rose rising rise increased increase surged grew growth record higher expanded".split())
_DOWN = frozenset("fell falling fall declined decline decreased slump lower stalled cut cancelled".split())


def _direction(text: str) -> int:
    words = set(re.findall(r"[a-z]+", text.lower()))
    return (1 if words & _UP else 0) - (1 if words & _DOWN else 0)


def _negated(text: str) -> bool:
    return bool(set(re.findall(r"[a-z']+", text.lower())) & _NEGATIONS) or "n't" in text.lower()


def _core(question: str) -> str:
    q = question.strip().rstrip("?")
    q = re.sub(r"^(how (much|many)|what|how|why|when|where|which|who|does|do|is|are|did)\s+", "", q, flags=re.I)
    return q[:120]


def hashed_embedding(text: str, dims: int = 96) -> list[float]:
    vec = [0.0] * dims
    for tok in _tokens(text):
        vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % dims] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class HeuristicMeshSimulator:
    """Offline chat/embedding client. Thread-safe (stateless apart from counters)."""

    model_label = "offline-heuristic-simulator"

    def __init__(self, *, hallucinate_first_draft: bool = False, reject_first_plan: bool = False) -> None:
        self.hallucinate_first_draft = hallucinate_first_draft
        self.reject_first_plan = reject_first_plan
        self.calls_by_role: dict[str, int] = {}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.embeddings = SimpleNamespace(create=self._embed)

    # -- client surface ------------------------------------------------------
    def _create(self, **kwargs: Any) -> Any:
        messages = kwargs["messages"]
        system = messages[0]["content"]
        role_match = re.search(r"ROLE:\s*([a-z.]+)", system)
        role = role_match.group(1) if role_match else "unknown"
        self.calls_by_role[role] = self.calls_by_role.get(role, 0) + 1
        user = next(m["content"] for m in messages if m["role"] == "user")
        payload = json.loads(user.split(INPUT_MARKER, 1)[1]) if INPUT_MARKER in user else {}
        handler = getattr(self, "_" + role.replace(".", "_"), None)
        body = handler(payload) if handler else {}
        content = json.dumps(body, ensure_ascii=False)
        prompt_tokens = sum(len(m["content"]) for m in messages) // 4
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=len(content) // 4,
                total_tokens=prompt_tokens + len(content) // 4,
            ),
        )

    def _embed(self, *, model: str, input: list[str]) -> Any:
        data = [SimpleNamespace(index=i, embedding=hashed_embedding(t)) for i, t in enumerate(input)]
        tokens = sum(len(t) for t in input) // 4
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=tokens, total_tokens=tokens))

    # -- agents ---------------------------------------------------------------
    _PARAPHRASED_JAILBREAKS = (
        "no longer bound", "pretend the rules", "without your usual limits", "rules do not apply to you",
        "act unrestricted", "set aside your guidelines",
    )

    def _guardian(self, p: dict[str, Any]) -> dict[str, Any]:
        question = p.get("question", "").lower()
        if any(marker in question for marker in self._PARAPHRASED_JAILBREAKS):
            return {"verdict": "block", "categories": ["prompt_injection"], "prompt_injection": True,
                    "confidence": 0.93, "rationale": "paraphrased attempt to disable safety rules"}
        return {"verdict": "allow", "categories": [], "prompt_injection": False, "confidence": 0.97, "rationale": "ordinary research question"}

    def _planner(self, p: dict[str, Any]) -> dict[str, Any]:
        core = _core(p.get("research_question", "the topic"))
        return {
            "research_strategy": "Facts, then mechanisms, then points of disagreement.",
            "sub_questions": [
                {
                    "question": f"What do authoritative sources report about {core}?",
                    "rationale": "Establish the factual baseline.",
                    "search_queries": [core, f"{core} official data"],
                    "priority": 1,
                },
                {
                    "question": f"What factors or mechanisms explain {core}?",
                    "rationale": "Explain drivers and causal mechanisms.",
                    "search_queries": [f"{core} drivers mechanism"],
                    "priority": 2,
                },
                {
                    "question": f"Which claims about {core} are disputed or contradicted by evidence?",
                    "rationale": "Test the premise and surface conflicting evidence.",
                    "search_queries": [f"{core} evidence myth contradicted"],
                    "priority": 3,
                },
            ],
        }

    def _planner_revise(self, p: dict[str, Any]) -> dict[str, Any]:
        core = _core(p.get("research_question", "the topic"))
        current = p.get("current_plan", {}).get("sub_questions", [])
        subs = [{"question": sq["question"], "search_queries": sq["search_queries"][:3], "priority": min(i + 1, 5)}
                for i, sq in enumerate(current)]
        notes = []
        for issue in p.get("lead_review", {}).get("issues", []):
            if issue.get("severity") in ("high", "medium") and len(subs) < 5:
                subs.append({
                    "question": f"What official statistics or primary data quantify {core}?",
                    "rationale": "Requested by the Team Lead.",
                    "search_queries": [f"{core} official statistics"],
                    "priority": min(len(subs) + 1, 5),
                })
                notes.append(f"Added a primary-data sub-question for: {issue.get('problem', 'review issue')}")
        return {"research_strategy": p.get("current_plan", {}).get("research_strategy", ""),
                "sub_questions": subs, "revision_notes": notes}

    _PREMISE_WORDS = frozenset("disputed contradicted myth premise whether verify true evidence".split())

    def _lead_plan(self, p: dict[str, Any]) -> dict[str, Any]:
        question = p.get("research_question", "")
        subs = p.get("plan", {}).get("sub_questions", [])
        plan_text = " ".join(f"{sq['question']} {' '.join(sq.get('search_queries', []))}" for sq in subs)
        q_tokens = set(_tokens(question))
        coverage = 0.6 + 0.4 * (len(q_tokens & set(_tokens(plan_text))) / len(q_tokens) if q_tokens else 1.0)
        pairs = [(a, b) for i, a in enumerate(subs) for b in subs[i + 1:]]
        def focus(text: str) -> set[str]:
            return set(_tokens(text)) - q_tokens

        overlap = max(
            (len(focus(a["question"]) & focus(b["question"])) / len(focus(a["question"]) | focus(b["question"]))
             for a, b in pairs if focus(a["question"]) and focus(b["question"])),
            default=0.0,
        )
        queries = [q for sq in subs for q in sq.get("search_queries", [])]
        # lenient on length (the simulator's own planner echoes the question); the deterministic check still flags it
        searchable = sum(2 <= len(q.split()) <= 18 for q in queries) / len(queries) if queries else 0.0
        tests_premise = any(set(_tokens(sq["question"])) & self._PREMISE_WORDS for sq in subs)
        premise = 1.0 if tests_premise or not question.lower().startswith("why") else 0.5
        issues: list[dict[str, Any]] = []
        specificity = 0.85
        if self.reject_first_plan and p.get("round") == 1:
            specificity = 0.55
            issues.append({"target": "plan", "severity": "high",
                           "problem": "no sub-question asks for primary, quantified data",
                           "fix": "Add a sub-question asking which official statistics quantify the topic."})
        if premise < 1.0:
            issues.append({"target": "plan", "severity": "high", "problem": "the question's premise is never tested",
                           "fix": "Add a sub-question that checks whether the premise is true."})
        rubric = {"coverage": round(coverage, 2), "distinctness": round(1.0 - overlap, 2),
                  "searchability": round(searchable, 2), "premise_check": premise, "specificity": specificity}
        verdict = "needs revision" if issues else "ready for retrieval"
        return {"rubric": rubric, "issues": issues, "summary": f"Plan reviewed against the rubric: {verdict}."}

    def _lead_review(self, p: dict[str, Any]) -> dict[str, Any]:
        """Conservative lead: loops only when the quality gate requires it, approves otherwise."""
        gate = p.get("quality_gate", {})
        gaps = p.get("gaps", [])
        if not gate.get("another_round_required"):
            return {"decision": "approve", "rationale": "The supported claims cover the question; ready for synthesis.",
                    "focus_sub_question_ids": [], "instructions": []}
        instructions = []
        for g in gaps:
            core = _core(g.get("sub_question") or p.get("research_question", ""))
            instructions.append({"agent": "researcher", "sub_question_id": g["sub_question_id"],
                                 "instruction": f"Look for official statistics or primary reports on {core}."})
            instructions.append({"agent": "drafter", "sub_question_id": g["sub_question_id"],
                                 "instruction": "Remove claims the Verifier rejected; state only what the evidence says verbatim."})
        return {"decision": "another_round", "rationale": "Grounding is below the bar; the weak sub-questions need primary evidence.",
                "focus_sub_question_ids": [g["sub_question_id"] for g in gaps], "instructions": instructions}

    def _researcher(self, p: dict[str, Any]) -> dict[str, Any]:
        queries = []
        for gap in p.get("gaps", []):
            base = _core(gap.get("sub_question") or p.get("research_question", ""))
            queries.append({"sub_question_id": gap["sub_question_id"], "query": f"{base} official statistics"})
            queries.append({"sub_question_id": gap["sub_question_id"], "query": f"{base} peer reviewed study"})
        return {"queries": queries or [{"sub_question_id": "SQ-1", "query": p.get("research_question", "")}]}

    def _drafter(self, p: dict[str, Any]) -> dict[str, Any]:
        target = f"{p.get('sub_question', '')} {p.get('research_question', '')}"
        scored: list[tuple[float, str, str]] = []
        for ev in p.get("evidence", []):
            for sentence in _sentences(ev["text"]):
                score = _overlap(target, sentence) + 0.3 * float(ev.get("authority") or 0.5)
                scored.append((score, ev["source_id"], sentence))
        scored.sort(key=lambda t: t[0], reverse=True)
        picked: list[str] = []
        used_sources: dict[str, int] = {}
        for _score, sid, sentence in scored:
            if used_sources.get(sid, 0) >= 2:
                continue
            used_sources[sid] = used_sources.get(sid, 0) + 1
            picked.append(f"{sentence.rstrip('.')} [{sid}].")
            if len(picked) >= 4:
                break
        if self.hallucinate_first_draft and "previous_attempt_feedback" not in p and scored:
            picked.append(f"Independent analysts expect this figure to double again by 2031 [{scored[0][1]}].")
        if not picked:
            return {"answer": "The available evidence does not establish an answer.", "evidence_sufficient": False}
        return {"answer": " ".join(picked), "evidence_sufficient": True}

    def _verifier_claims(self, p: dict[str, Any]) -> dict[str, Any]:
        claims = []
        for sentence in _SENT.split(p.get("draft", "")):
            cites = re.findall(r"SRC-\d+", sentence)
            text = _CITE.sub("", sentence).strip()
            if cites and len(text.split()) >= 4:
                claims.append({"text": text, "cited_source_ids": cites})
        return {"claims": claims[:12]}

    def _verifier_entailment(self, p: dict[str, Any]) -> dict[str, Any]:
        judgements = []
        for claim in p.get("claims", []):
            best: tuple[float, str | None, str | None] = (0.0, None, None)
            for ev in claim.get("evidence", []):
                for sentence in _sentences(ev["text"]) or [ev["text"]]:
                    score = _overlap(claim["claim"], sentence)
                    if score > best[0]:
                        best = (score, ev["source_id"], sentence)
            score, sid, sentence = best
            if sentence and score >= 0.8 and _numbers(claim["claim"]) <= _numbers(sentence):
                verdict, conf = "SUPPORTED", min(0.99, 0.5 + score / 2)
            elif sentence and score >= 0.5 and (
                _negated(claim["claim"]) != _negated(sentence) or not _numbers(claim["claim"]) <= _numbers(sentence)
            ):
                verdict, conf = "CONTRADICTED", 0.7
            else:
                verdict, conf, sentence = "UNSUPPORTED", 0.6, None
            judgements.append(
                {
                    "claim_id": claim["claim_id"],
                    "verdict": verdict,
                    "confidence": round(conf, 3),
                    "evidence_source_id": sid if sentence else None,
                    "evidence_quote": sentence,
                    "rationale": f"token overlap {score:.2f} with best evidence sentence",
                }
            )
        return {"judgements": judgements}

    def _verifier_contradictions(self, p: dict[str, Any]) -> dict[str, Any]:
        sents = [(ev["source_id"], s) for ev in p.get("evidence", []) for s in _sentences(ev["text"])]
        found: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for i, (sa, a) in enumerate(sents):
            for sb, b in sents[i + 1 :]:
                if sa == sb or (sa, sb) in seen:
                    continue
                ta, tb = set(_tokens(a)), set(_tokens(b))
                shared = ta & tb
                da, db = _direction(a), _direction(b)
                conflict = _negated(a) != _negated(b) or (da * db == -1)
                if len(shared) >= 3 and len(shared) / len(ta | tb) >= 0.15 and conflict:
                    seen.add((sa, sb))
                    found.append(
                        {
                            "topic": " ".join(sorted(shared)[:5]),
                            "source_a": sa,
                            "quote_a": a,
                            "source_b": sb,
                            "quote_b": b,
                            "severity": "medium",
                            "explanation": "Opposite polarity statements about the same subject.",
                        }
                    )
        return {"contradictions": found[:5]}

    def _synthesizer(self, p: dict[str, Any]) -> dict[str, Any]:
        sections, findings, summary = [], [], []
        removed = 0
        for sq in p.get("sub_questions", []):
            claims = sq.get("verified_claims", [])
            removed += int(sq.get("claims_removed_by_verifier", 0))
            sentences = [f"{c['claim'].rstrip('.')} [{', '.join(c['citations'])}]." for c in claims]
            if claims:
                summary.append(sentences[0])
                findings.extend(sentences[:2])
            body = " ".join(sentences) or "The available evidence was insufficient to answer this sub-question."
            sections.append({"sub_question_id": sq["sub_question_id"], "heading": sq["question"], "body": body})
        for c in p.get("contradictions", []):
            target = sections[0] if sections else None
            if target is not None:
                target["body"] += (
                    f" Sources conflict: {c['position_a'].rstrip('.')} [{c['source_a']}], whereas "
                    f"{c['position_b'].rstrip('.')} [{c['source_b']}]."
                )
        limitations = [f"{removed} drafted claim(s) were removed because verification failed."] if removed else []
        if p.get("low_authority_sources"):
            limitations.append("Some evidence comes from low-authority sources.")
        return {
            "title": f"Research brief: {_core(p.get('research_question', ''))}",
            "executive_summary": " ".join(summary[:4])
            or "The available evidence was insufficient to answer the research question.",
            "key_findings": findings[:6],
            "sections": sections,
            "limitations": limitations,
        }

    def _followup(self, p: dict[str, Any]) -> dict[str, Any]:
        best: tuple[float, str | None, str | None] = (0.0, None, None)
        for ev in p.get("evidence", []):
            for sentence in _sentences(ev["text"]):
                score = _overlap(p.get("question", ""), sentence)
                if score > best[0]:
                    best = (score, ev["source_id"], sentence)
        score, sid, sentence = best
        if sentence and score >= 0.34:
            return {
                "answer": f"{sentence.rstrip('.')} [{sid}].",
                "answerable": True,
                "cited_source_ids": [sid],
                "confidence": round(min(0.95, score), 3),
            }
        return {
            "answer": "The run's evidence does not answer this question.",
            "answerable": False,
            "cited_source_ids": [],
            "confidence": 0.1,
        }
