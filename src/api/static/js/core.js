// DeepDive dashboard core: HTML escaping, API client (with optional API key), shared run store + event bus.
'use strict';

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
export const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
export const pct = (v) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : `${Math.round(v * 100)}%`;
export const fmt = (n) => (n ?? 0).toLocaleString();

// ---------------------------------------------------------------- API key (sessionStorage: per tab, never persisted)
const KEY_STORAGE = 'deepdive_api_key';
export function apiKey() { try { return sessionStorage.getItem(KEY_STORAGE) || ''; } catch (_) { return ''; } }
export function setApiKey(key) { try { key ? sessionStorage.setItem(KEY_STORAGE, key) : sessionStorage.removeItem(KEY_STORAGE); } catch (_) {} }
export function withKey(url) { const k = apiKey(); return k ? `${url}${url.includes('?') ? '&' : '?'}api_key=${encodeURIComponent(k)}` : url; }

export class ApiError extends Error {
  constructor(status, message, body) { super(message); this.status = status; this.body = body; }
}

export async function api(path, opts = {}) {
  const headers = new Headers(opts.headers || {});
  const key = apiKey();
  if (key) headers.set('X-API-Key', key);
  const res = await fetch(path, { ...opts, headers });
  if (!res.ok) {
    let body = null; let detail = res.statusText;
    try { body = await res.json(); detail = body.error?.message || (typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)) || detail; } catch (_) {}
    if (res.status === 401) bus.dispatchEvent(new CustomEvent('auth-required'));
    throw new ApiError(res.status, detail, body);
  }
  const ctype = res.headers.get('content-type') || '';
  return ctype.includes('application/json') ? res.json() : res;
}

// ---------------------------------------------------------------- shared state
export const bus = new EventTarget();
export const emit = (type, detail) => bus.dispatchEvent(new CustomEvent(type, { detail }));
export const on = (type, fn) => bus.addEventListener(type, (e) => fn(e.detail));

export const store = {
  runId: null,
  run: null,
  events: [],      // audit events (live or replayed)
  traces: [],      // Mesh call traces
  live: false,     // true while an SSE stream is attached
  health: null,
};

// ---------------------------------------------------------------- UI helpers
export function toast(msg, tone = 'error') {
  const t = $('#toast');
  t.textContent = msg;
  t.dataset.tone = tone;
  t.classList.remove('hidden');
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.add('hidden'), 5200);
}

export function cite(text) {
  return esc(text).replace(/\[((?:SRC-\d{3,4}(?:\s*[,;]\s*)?)+)\]/g, (_m, ids) =>
    ids.split(/\s*[,;]\s*/).filter(Boolean).map((id) => `<span class="chip" data-src="${id}">${id}</span>`).join(' '));
}

export const ts = (ev) => Date.parse(ev.timestamp || ev.created_at || 0) || 0;
