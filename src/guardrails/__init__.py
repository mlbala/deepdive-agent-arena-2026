"""DeepDive guardrails: deterministic shields around every boundary of the agent graph.

    input_shield    user questions + follow-ups: prompt injection, harmful requests, PII, secrets
    llm_screen      Mesh-backed Guardian classifier (defence in depth after the deterministic shield)
    content_shield  retrieved web pages + uploads: indirect prompt injection, hidden text, PII, secrets
    url_shield      SSRF-safe fetching: scheme/port/host/IP/redirect/size policy
    output_shield   final report + follow-up answers: PII and secret redaction, disclosure
    budget          per-run Mesh call/token caps (soft: stop loop-backs, hard: refuse calls)
    model_policy    Mesh model allowlist

Every decision is a typed `GuardrailFinding` recorded in the hash-chained audit trail.
"""
