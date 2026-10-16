// Bootstrap: view switching (Live Demo / How it works / Console) and deep links
// (?view=demo|explorer|console&run=…&step=…&focus=…&tab=…).
'use strict';
import { $$ } from './core.js';
import { initConsole, openRun, selectTab } from './console.js';
import { demoShown, initDemo } from './demo.js';
import { explorerShown, initExplorer } from './explorer.js';

const VIEWS = ['demo', 'explorer', 'console'];

function showView(view, params = {}) {
  VIEWS.forEach((v) => document.getElementById(`view-${v}`).classList.toggle('hidden', v !== view));
  $$('[data-view]').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.view === view)));
  if (view === 'explorer') explorerShown(params);
  if (view === 'demo') demoShown(params);
  const url = new URL(location.href);
  url.searchParams.set('view', view);
  ['run', 'step', 'focus', 'tab'].forEach((k) => { if (!params[k]) url.searchParams.delete(k); });
  history.replaceState(null, '', url);
}

document.addEventListener('click', (e) => {
  const v = e.target.closest('[data-view]'); if (v) { showView(v.dataset.view); return; }
  const g = e.target.closest('[data-goto]'); if (g) showView(g.dataset.goto);
});

initConsole();
initExplorer();
initDemo();

const params = Object.fromEntries(new URLSearchParams(location.search));
const view = VIEWS.includes(params.view) ? params.view : 'demo';
if (params.run && view === 'console') openRun(params.run).then(() => params.tab && selectTab(params.tab)).catch(() => {});
showView(view, params);
