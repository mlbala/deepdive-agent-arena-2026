// Bootstrap: view switching (Live Console / How it works) and deep links (?view=explorer&run=…&step=…&focus=…).
'use strict';
import { $, $$ } from './core.js';
import { initConsole, openRun, selectTab } from './console.js';
import { explorerShown, initExplorer } from './explorer.js';

function showView(view, params = {}) {
  const explorer = view === 'explorer';
  $('#view-console').classList.toggle('hidden', explorer);
  $('#view-explorer').classList.toggle('hidden', !explorer);
  $$('[data-view]').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.view === view)));
  if (explorer) explorerShown(params);
  const url = new URL(location.href);
  url.searchParams.set('view', view);
  history.replaceState(null, '', url);
}

document.addEventListener('click', (e) => {
  const v = e.target.closest('[data-view]'); if (v) { showView(v.dataset.view); return; }
  const g = e.target.closest('[data-goto]'); if (g) showView(g.dataset.goto);
});

initConsole();
initExplorer();

const params = Object.fromEntries(new URLSearchParams(location.search));
if (params.run && params.view !== 'explorer') openRun(params.run).then(() => params.tab && selectTab(params.tab)).catch(() => {});
showView(params.view === 'explorer' ? 'explorer' : 'console', params);
