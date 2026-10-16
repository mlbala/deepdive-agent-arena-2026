// Bootstrap: views (Live Demo / How it works / Console), appearance, command palette, shortcuts, deep links
// (?view=demo|explorer|console&run=…&step=…&focus=…&tab=…&scenario=…&autorun=1).
'use strict';
import { $, $$, emit, on, store, toast } from './core.js';
import { initChartTips } from './charts.js';
import { initConsole, openRun, selectTab } from './console.js';
import { SCENARIOS, demoReplay, demoShown, initDemo } from './demo.js';
import { explorerShown, initExplorer } from './explorer.js';
import { closeLens, initLens, openLens } from './lens.js';
import { closePalette, initPalette, openPalette } from './palette.js';

const VIEWS = ['demo', 'explorer', 'console'];
let currentView = 'demo';

function showView(view, params = {}) {
  currentView = view;
  VIEWS.forEach((v) => document.getElementById(`view-${v}`).classList.toggle('hidden', v !== view));
  $$('[data-view]').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.view === view)));
  if (view === 'explorer') explorerShown(params);
  if (view === 'demo') demoShown(params);
  if (view === 'console' && params.tab) selectTab(params.tab);
  const url = new URL(location.href);
  url.searchParams.set('view', view);
  ['run', 'step', 'focus', 'tab', 'scenario', 'autorun', 'lens', 'cmdk'].forEach((k) => { if (!params[k]) url.searchParams.delete(k); });
  history.replaceState(null, '', url);
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ---------------------------------------------------------------- appearance (auto follows the OS)
const THEME_KEY = 'deepdive_theme';
const THEME_ICON = {
  auto: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="8"/><path d="M12 4a8 8 0 0 1 0 16z" fill="currentColor"/></svg>',
  light: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2.5v2M12 19.5v2M4.6 4.6l1.4 1.4M18 18l1.4 1.4M2.5 12h2M19.5 12h2M4.6 19.4L6 18M18 6l1.4-1.4"/></svg>',
  dark: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/></svg>',
};
function currentTheme() { try { return localStorage.getItem(THEME_KEY) || 'auto'; } catch (_) { return 'auto'; } }
function applyTheme(theme) {
  if (theme === 'auto') delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = theme;
  try { theme === 'auto' ? localStorage.removeItem(THEME_KEY) : localStorage.setItem(THEME_KEY, theme); } catch (_) { /* private mode */ }
  const btn = $('#themeBtn');
  btn.innerHTML = THEME_ICON[theme];
  btn.setAttribute('aria-label', `Appearance: ${theme}`);
  btn.title = `Appearance: ${theme} (T)`;
}
function cycleTheme() {
  const order = ['auto', 'light', 'dark'];
  const next = order[(order.indexOf(currentTheme()) + 1) % order.length];
  applyTheme(next);
  toast(`Appearance: ${next === 'auto' ? 'automatic (follows your system)' : next}`, 'info');
}

// ---------------------------------------------------------------- shortcuts sheet
const SHORTCUTS = [
  ['⌘K / Ctrl K', 'Command palette: run scenarios, open runs, find claims and sources'],
  ['1 · 2 · 3', 'Live Demo · How it works · Console'],
  ['Space', 'Play / pause (Live Demo and How it works)'],
  ['→ / ←', 'Next / previous step · next / previous claim in the Evidence Lens'],
  ['R', 'Reset the replay'],
  ['T', 'Cycle appearance: auto, light, dark'],
  ['?', 'This sheet'],
  ['Esc', 'Close any sheet'],
];
function openShortcuts() {
  $('#shortcutsBody').innerHTML = `<div class="flex items-center"><div class="text-[17px] font-semibold tracking-tight">Keyboard shortcuts</div><button class="icon-btn ml-auto" data-close-shortcuts aria-label="Close">✕</button></div>
    <div class="mt-4 divide-y divide-ink-700">${SHORTCUTS.map(([k, d]) => `<div class="flex items-center gap-4 py-2.5 text-[13.5px]"><span class="w-32 shrink-0">${k.split(' ').map((x) => (['/', '·'].includes(x) ? `<span class="text-slate-500">${x}</span>` : `<kbd>${x}</kbd>`)).join(' ')}</span><span class="text-slate-300">${d}</span></div>`).join('')}</div>`;
  $('#shortcuts').classList.remove('hidden');
}
const closeShortcuts = () => $('#shortcuts').classList.add('hidden');

// ---------------------------------------------------------------- wiring
document.addEventListener('click', (e) => {
  const v = e.target.closest('[data-view]'); if (v) { showView(v.dataset.view); return; }
  const g = e.target.closest('[data-goto]'); if (g) { showView(g.dataset.goto, g.dataset.tab ? { tab: g.dataset.tab } : {}); return; }
  if (e.target.closest('[data-close-shortcuts]') || e.target.id === 'shortcuts') closeShortcuts();
  const pop = $('#statusPop');
  if (e.target.closest('#statusBtn')) pop.classList.toggle('hidden');
  else if (!e.target.closest('#statusPop')) pop.classList.add('hidden');
});

document.addEventListener('keydown', (e) => {
  const typing = ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName);
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); openPalette(SCENARIOS); return; }
  if (e.key === 'Escape') { closePalette(); closeShortcuts(); closeLens(); $('#statusPop').classList.add('hidden'); return; }
  if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
  if (!$('#lens').classList.contains('hidden') || !$('#cmdk').classList.contains('hidden')) return;
  if (e.key === '1') showView('demo');
  if (e.key === '2') showView('explorer');
  if (e.key === '3') showView('console');
  if (e.key === 't' || e.key === 'T') cycleTheme();
  if (e.key === '?') openShortcuts();
});

on('navigate', ({ view, params = {} }) => showView(view, params));
on('theme-cycle', cycleTheme);
on('shortcuts', openShortcuts);
on('demo-scenario', () => { if (currentView !== 'demo') showView('demo'); });
on('open-run', ({ runId }) => {
  if (currentView === 'demo') demoReplay(runId);
  else if (currentView === 'explorer') explorerShown({ run: runId });
  else openRun(runId).catch((err) => toast(err.message));
});

$('#cmdkBtn').addEventListener('click', () => openPalette(SCENARIOS));
$('#themeBtn').addEventListener('click', cycleTheme);
applyTheme(currentTheme());

initConsole();
initExplorer();
initDemo();
initLens();
initPalette();
initChartTips();

const params = Object.fromEntries(new URLSearchParams(location.search));
const view = VIEWS.includes(params.view) ? params.view : 'demo';
if (params.run && view === 'console') openRun(params.run).then(() => params.tab && selectTab(params.tab)).catch(() => {});
showView(view, params);
if (params.lens && params.run) setTimeout(() => openLens({ runId: params.run, claimId: params.lens }), 400); // shareable claim links
if (params.cmdk === '1') setTimeout(() => openPalette(SCENARIOS), 300);
emit('app-ready', { store });
