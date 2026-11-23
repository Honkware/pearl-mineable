import '@fontsource/atkinson-hyperlegible-next/400.css';
import '@fontsource/atkinson-hyperlegible-next/600.css';
import '@fontsource/atkinson-hyperlegible-next/700.css';
import '@fontsource/atkinson-hyperlegible-mono/400.css';
import './style.css';
import { analyze, autoTp, type Result } from './analyze.js';
import { readConfig, UnsupportedModel, type Model } from './arch.js';
import { fetchTip, type Tip } from './chain.js';
import { GPUS, gpu } from './gpus.js';
import { KNOWN, type Known } from './known.js';
import { analyzeNow } from './now.js';

const state = { gpu: 'h100', tp: 'auto' as 'auto' | number, tip: null as Tip | null, tipError: false, sort: 'fp8', desc: true };
const extra = new Map<string, { name: string; config: Record<string, unknown> }>();
const app = document.getElementById('app')!;

const esc = (s: string): string => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);
const int = (n: number): string => n.toLocaleString('en-US');
const pct = (x: number): string => `${x >= 0.995 || x === 0 ? Math.round(x * 100) : (x * 100).toFixed(x < 0.1 ? 1 : 0)}%`;
const sig = (x: number): string => (x === 0 ? '0' : x >= 1 ? x.toFixed(2) : x.toPrecision(2));
const prlPerMillion = (r: Result | null): string => (state.tip && r ? sig(r.mined * 1e6 * state.tip.prlPerMac) : '-');
const TIPS: Record<string, string> = {"Mineable now": "Share of the model's prompt math that mines on mainnet today", "After FP8": "Share of the model's prompt math that mines after the FP8 fork", "Mineable": "Share of the model's prompt math that mines", "PRL / 1M tokens": "PRL earned per million prompt tokens at the current difficulty", "TP": "GPUs the model is split across"};
const th = (label: string, cls = '', key = ''): string => {
  const arrow = key && state.sort === key ? (state.desc ? ' ↓' : ' ↑') : '';
  return `<th${cls ? ` class="${cls}"` : ''}${key ? ` data-sort="${key}"` : ''}${TIPS[label] ? ` title="${esc(TIPS[label]!)}"` : ''}>${label}${arrow}</th>`;
};
const share = (r: Result | null): string => (r ? bar(r.share) : '-');

interface Entry {
  id: string;
  name: string;
  org: string;
  config: Record<string, unknown>;
  int7?: Known['int7'];
}

// Before the fork only Pearl's INT7 checkpoints mine, so earnings follow the
// era the chain tip is in.
function views(e: Entry, model: Model, card = gpu(state.gpu)): { fp8: Result | null; now: Result | null; paid: Result | null } {
  const fp8 = state.tp === 'auto' ? autoTp(model, card) : analyze(model, card, state.tp);
  const now = fp8 && e.int7 ? analyzeNow(e.int7.quant, fp8) : null;
  return { fp8, now, paid: state.tip?.fp8 ? fp8 : now };
}

function seg<T>(options: [T, string][], current: T, attr: string): string {
  return `<span class="seg">${options.map(([v, label]) => `<button type="button" data-${attr}="${v}" aria-pressed="${v === current}">${esc(label)}</button>`).join('')}</span>`;
}

function controls(): string {
  return `${seg(GPUS.map((g) => [g.id, g.name] as [string, string]), state.gpu, 'gpu')}
    ${seg([['auto', 'Auto TP'], [1, '1'], [2, '2'], [4, '4'], [8, '8']] as [string | number, string][], state.tp, 'tp')}
    <input class="gpus" type="number" min="1" max="64" placeholder="GPUs" value="${typeof state.tp === 'number' && ![1, 2, 4, 8].includes(state.tp) ? state.tp : ''}">`;
}

function stats(items: [string, string][]): string {
  return `<div class="stats">${items.map(([k, v]) => `<div class="stat"><div class="k"${TIPS[k] ? ` title="${esc(TIPS[k]!)}"` : ''}>${k}</div><div class="v">${v}</div></div>`).join('')}</div>`;
}

function bar(share: number): string {
  return `<div class="share"><div class="mbar"><i style="width:${(share * 100).toFixed(1)}%"></i></div><span class="pctv">${pct(share)}</span></div>`;
}

function listPage(): string {
  const forked = state.tip?.fp8 === true;
  const rows = entries().map((e) => {
    const model = readConfig(e.config);
    return { e, model, ...views(e, model) };
  });
  type Row = (typeof rows)[number];
  const keys: Record<string, (x: Row) => number | string | null> = {
    name: (x) => x.e.name.toLowerCase(),
    kind: (x) => (x.model.experts ? 'MoE' : 'Dense'),
    tp: (x) => x.fp8?.tp ?? null,
    now: (x) => x.now?.share ?? null,
    fp8: (x) => x.fp8?.share ?? null,
    prl: (x) => (x.paid ? x.paid.mined : null),
  };
  const key = keys[state.sort] ?? keys.fp8!;
  // Models without a value stay at the bottom in either direction.
  const order = (a: Row, b: Row): number => {
    const [x, y] = [key(a), key(b)];
    if (x === null || y === null) return x === y ? 0 : x === null ? 1 : -1;
    const c = x < y ? -1 : x > y ? 1 : 0;
    return state.desc ? -c : c;
  };
  const body = rows
    .sort(order)
    .map(({ e, model, fp8, now, paid }, i) => `<tr><td class="rank">${i + 1}</td>
        <td><a class="who" href="#/m/${encodeURIComponent(e.id)}">${esc(e.name)}</a><span class="sub">${esc(e.org)}</span></td>
        <td>${model.experts ? 'MoE' : 'Dense'}</td><td>${fp8 ? `${fp8.tp}${fp8.fits ? '' : '+'}` : '-'}</td>
        ${forked ? '' : `<td>${share(now)}</td>`}<td>${share(fp8)}</td><td class="n">${prlPerMillion(paid)}</td></tr>`)
    .join('');
  return `<div class="title"><h1>Models</h1></div>
  <form class="lookup" id="search"><input name="id" placeholder="Hugging Face ID, such as Qwen/Qwen3-8B" autocomplete="off"><button type="submit">Check</button></form>
  <div id="search-error" class="err" hidden></div>
  <div class="card">
    <div class="filters">${controls()}</div>
    <div class="scroll"><table>
      <thead><tr><th></th>${th('Model', '', 'name')}${th('Kind', '', 'kind')}${th('TP', '', 'tp')}${forked ? th('Mineable', '', 'fp8') : th('Mineable now', '', 'now') + th('After FP8', '', 'fp8')}${th('PRL / 1M tokens', 'n', 'prl')}</tr></thead>
      <tbody>${body}</tbody>
    </table></div>
  </div>`;
}

function entries(): Entry[] {
  return [...KNOWN, ...[...extra].map(([id, e]) => ({ id, name: e.name, org: id.split('/')[0]!, config: e.config }))];
}

function modelPage(id: string): string {
  const entry = entries().find((e) => e.id === id);
  if (!entry) return `<div class="card empty">Not loaded. <a class="more inline" href="#/">Back</a></div>`;
  const forked = state.tip?.fp8 === true;
  const model = readConfig(entry.config);
  const { fp8: r, now, paid } = views(entry, model);
  if (!r) return `<div class="title"><h1>${esc(entry.name)}</h1></div><div class="filters">${controls()}</div><div class="card empty">Cannot split ${state.tp} ways</div>`;
  const byGpu = GPUS.map((g) => {
    const v = views(entry, model, g);
    return `<tr><td class="who">${esc(g.name)}</td><td>${g.memory} GB</td><td>${v.fp8 ? `${v.fp8.tp}${v.fp8.fits ? '' : '+'}` : '-'}</td>
      ${forked ? '' : `<td>${share(v.now)}</td>`}<td>${share(v.fp8)}</td><td class="n">${prlPerMillion(v.paid)}</td></tr>`;
  }).join('');
  const rows = r.rows
    .map((x, i) => `<tr><td class="mono">${esc(x.name)}</td><td class="n">${int(x.n)}</td><td class="n">${int(x.k)}</td>
      <td class="n">${int(x.localN)} × ${int(x.localK)}</td><td class="n">${x.layers}</td>
      ${forked ? '' : `<td class="n">${now ? now.rows[i]!.minedLayers || '-' : '-'}</td>`}<td>${x.tile ?? '-'}</td>
      <td class="n">${pct(x.work / r.work)}</td></tr>`)
    .join('');
  const gpuHead = forked ? th('Mineable') : th('Mineable now') + th('After FP8');
  return `<div class="title"><h1>${esc(entry.name)}</h1><span class="chip mono">${esc(id)}</span></div>
  <div class="filters">${controls()}</div>
  ${stats([
    ...(forked ? [['Mineable', pct(r.share)] as [string, string]] : [['Mineable now', now ? pct(now.share) : '-'], ['After FP8', pct(r.share)]] as [string, string][]),
    ['PRL / 1M tokens', prlPerMillion(paid)],
    ['TP', `${r.tp}${r.fits ? '' : '+'}`],
    ['GB per GPU', String(Math.round(r.bytesPerGpu / 1e9))],
  ])}
  <div class="card">
    <div class="head"><h2>By GPU</h2></div>
    <div class="scroll"><table>
      <thead><tr><th>GPU</th><th>Memory</th>${th('TP')}${gpuHead}${th('PRL / 1M tokens', 'n')}</tr></thead>
      <tbody>${byGpu}</tbody>
    </table></div>
  </div>
  <div class="card">
    <div class="head"><h2>Layers</h2></div>
    <div class="scroll"><table>
      <thead><tr><th>Linear</th><th class="n">n</th><th class="n">k</th><th class="n">Per GPU</th><th class="n">Layers</th>${forked ? '' : '<th class="n">Mined now</th>'}<th>FP8 tile</th><th class="n">Work</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
  </div>`;
}

async function lookUp(id: string): Promise<void> {
  const error = document.getElementById('search-error')!;
  const show = (text: string): void => {
    error.textContent = text;
    error.hidden = false;
  };
  id = id.trim().replace(/^https:\/\/huggingface\.co\//, '').replace(/\/+$/, '');
  if (!/^[\w.-]+\/[\w.-]+$/.test(id)) return show('Use owner/model');
  if (entries().some((e) => e.id === id)) {
    location.hash = `#/m/${encodeURIComponent(id)}`;
    return;
  }
  let response: Response;
  try {
    response = await fetch(`https://huggingface.co/${id}/resolve/main/config.json`);
  } catch {
    return show('Hugging Face unreachable');
  }
  if (response.status === 401 || response.status === 403) return show('Gated on Hugging Face');
  if (!response.ok) return show(`No config.json (HTTP ${response.status})`);
  const config = (await response.json()) as Record<string, unknown>;
  try {
    readConfig(config);
  } catch (err) {
    return show(err instanceof UnsupportedModel ? err.message : 'Config not readable');
  }
  extra.set(id, { name: id.split('/')[1]!, config });
  location.hash = `#/m/${encodeURIComponent(id)}`;
}

function render(): void {
  const match = /^#\/m\/(.+)$/.exec(location.hash);
  app.innerHTML = match ? modelPage(decodeURIComponent(match[1]!)) : listPage();
  const t = state.tip;
  document.getElementById('tip')!.innerHTML = t ? '<span class="live"></span>mainnet' : state.tipError ? 'chain unreachable' : 'mainnet';
  document.getElementById('strip')!.innerHTML = [
    ['Block height', t ? int(t.height) : '-'],
    ['Difficulty', t ? `${(t.difficulty / 1e6).toFixed(2)} M` : '-'],
    ['Block reward', t ? `${t.subsidy.toFixed(2)} PRL` : '-'],
  ].map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join('');
  document.getElementById('foot')!.innerHTML = 'pearljs';
}

app.addEventListener('click', (e) => {
  const head = (e.target as HTMLElement).closest<HTMLElement>('th[data-sort]');
  if (head) {
    const key = head.dataset.sort!;
    state.desc = state.sort === key ? !state.desc : key !== 'name' && key !== 'kind';
    state.sort = key;
    render();
    return;
  }
  const b = (e.target as HTMLElement).closest('button');
  if (!b) return;
  if (b.dataset.gpu) state.gpu = b.dataset.gpu;
  if (b.dataset.tp) state.tp = b.dataset.tp === 'auto' ? 'auto' : Number(b.dataset.tp);
  render();
});
app.addEventListener('change', (e) => {
  const input = e.target as HTMLInputElement;
  if (!input.classList.contains('gpus')) return;
  const n = Math.round(Number(input.value));
  state.tp = Number.isFinite(n) && n >= 1 && n <= 64 ? n : 'auto';
  render();
});
app.addEventListener('submit', (e) => {
  e.preventDefault();
  void lookUp(new FormData(e.target as HTMLFormElement).get('id') as string);
});
document.getElementById('theme')!.addEventListener('click', () => {
  const r = document.documentElement;
  const dark = r.dataset.theme ? r.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  r.dataset.theme = dark ? 'light' : 'dark';
  try {
    localStorage.setItem('theme', r.dataset.theme);
  } catch {}
});
addEventListener('hashchange', render);
render();
fetchTip()
  .then((tip) => (state.tip = tip))
  .catch(() => (state.tipError = true))
  .finally(render);
