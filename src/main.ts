import '@fontsource/atkinson-hyperlegible-next/400.css';
import '@fontsource/atkinson-hyperlegible-next/600.css';
import '@fontsource/atkinson-hyperlegible-next/700.css';
import '@fontsource/atkinson-hyperlegible-mono/400.css';
import './style.css';
import { analyze, autoTp, type Result, type Row } from './analyze.js';
import { readConfig, UnsupportedModel, type Model } from './arch.js';
import { fetchTip, type Tip } from './chain.js';
import { GPUS, gpu } from './gpus.js';
import { KNOWN } from './known.js';
import vectors from '../vectors/tiles.json' with { type: 'json' };

const RULES = vectors.source.split(' ')[1]!.slice(0, 7);
const state = { gpu: 'h100', tp: 'auto' as 'auto' | number, tip: null as Tip | null, tipError: false };
const extra = new Map<string, { name: string; config: Record<string, unknown> }>();
const app = document.getElementById('app')!;

const esc = (s: string): string => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);
const int = (n: number): string => n.toLocaleString('en-US');
const pct = (x: number): string => `${x >= 0.995 || x === 0 ? Math.round(x * 100) : (x * 100).toFixed(x < 0.1 ? 1 : 0)}%`;
const sig = (x: number): string => (x === 0 ? '0' : x >= 1 ? x.toFixed(2) : x.toPrecision(2));
const prlPerMillion = (r: Result): string => (state.tip ? sig(r.mined * 1e6 * state.tip.prlPerMac) : '-');
const gb = (bytes: number): string => `${Math.round(bytes / 1e9)} GB`;

function run(m: Model): Result | null {
  return state.tp === 'auto' ? autoTp(m, gpu(state.gpu)) : analyze(m, gpu(state.gpu), state.tp);
}

/** One short label for what keeps the most work off the miner. */
function blocker(r: Result): { text: string; cls: string } {
  if (r.share >= 0.95) return { text: 'mines', cls: 'ok' };
  const worst = r.rows.filter((x) => !x.tile && x.name !== 'lm_head').sort((a, b) => b.work - a.work)[0];
  const cls = r.share >= 0.5 ? 'mid' : 'off';
  if (!worst) return { text: 'mines', cls: 'ok' };
  return { text: reason(worst), cls };
}

function reason(x: Row): string {
  if (x.why === 'shape') {
    if (x.localK % 512 !== 0) return `k ${int(x.localK)}`;
    if (x.localK < 1024) return 'split too small';
    return `n ${int(x.localN)}`;
  }
  if (x.why === 'never mined') return 'expert down proj';
  if (x.why === 'read directly by MLA') return 'MLA kv_b';
  if (x.why === 'not a linear layer in vLLM') return 'head';
  if (x.why === 'swiglu gate parameters') return 'SwiGLU limit';
  return x.why ?? '';
}

function seg<T>(options: [T, string][], current: T, attr: string): string {
  return `<span class="seg">${options.map(([v, label]) => `<button type="button" data-${attr}="${v}" aria-pressed="${v === current}">${esc(label)}</button>`).join('')}</span>`;
}

function controls(): string {
  return `${seg(GPUS.map((g) => [g.id, g.name] as [string, string]), state.gpu, 'gpu')}
    ${seg([['auto', 'Auto TP'], [1, '1'], [2, '2'], [4, '4'], [8, '8']] as [string | number, string][], state.tp, 'tp')}`;
}

function stats(items: [string, string, string][]): string {
  return `<div class="stats">${items.map(([k, v, s]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join('')}</div>`;
}

function bar(share: number): string {
  return `<div class="share"><div class="mbar"><i style="width:${(share * 100).toFixed(1)}%"></i></div><span class="pctv">${pct(share)}</span></div>`;
}

function listPage(): string {
  const models = [...KNOWN.map((k) => ({ id: k.id, name: k.name, org: k.org, pearl: k.pearl, config: k.config })), ...[...extra].map(([id, e]) => ({ id, name: e.name, org: id.split('/')[0]!, pearl: false, config: e.config }))];
  const rows = models.map((m) => {
    const model = readConfig(m.config);
    const r = run(model);
    return { m, model, r };
  });
  const scored = rows.filter((x) => x.r !== null).sort((a, b) => b.r!.share - a.r!.share);
  const body = scored
    .map(({ m, model, r }, i) => {
      const b = blocker(r!);
      return `<tr><td class="rank">${i + 1}</td>
        <td><a class="who" href="#/m/${encodeURIComponent(m.id)}">${esc(m.name)}</a><span class="sub">${esc(m.org)}${m.pearl ? ' · Pearl' : ''}</span></td>
        <td>${model.experts ? 'MoE' : 'Dense'}</td><td>${model.format.toUpperCase()}</td><td>${r!.tp}${r!.fits ? '' : '+'}</td>
        <td>${bar(r!.share)}</td><td class="n">${prlPerMillion(r!)}</td><td><span class="tag ${b.cls}">${esc(b.text)}</span></td></tr>`;
    })
    .join('');
  const tip = state.tip;
  return `${stats([
    ['Models', String(models.length), `${KNOWN.filter((k) => k.pearl).length} from Pearl`],
    ['PRL per 10<sup>18</sup> mined MAC', tip ? sig(tip.prlPerMac * 1e18) : '-', tip ? `block ${int(tip.height)}` : state.tipError ? 'chain unreachable' : 'loading'],
    ['Block subsidy', tip ? `${sig(tip.subsidy)} PRL` : '-', 'mainnet'],
    ['Rules', 'fp8-scheme', RULES],
  ])}
  <div class="card">
    <div class="filters">${controls()}<form class="search" id="search"><input name="id" placeholder="Hugging Face ID" autocomplete="off"></form></div>
    <div id="search-error" class="err" hidden></div>
    <div class="scroll"><table>
      <thead><tr><th></th><th>Model</th><th>Kind</th><th>Weights</th><th>TP</th><th>Mined</th><th class="n">PRL / 1M tokens</th><th></th></tr></thead>
      <tbody>${body}</tbody>
    </table></div>
  </div>`;
}

function modelPage(id: string): string {
  const known = KNOWN.find((k) => k.id === id);
  const entry = known ?? (extra.has(id) ? { id, name: extra.get(id)!.name, org: id.split('/')[0]!, pearl: false, config: extra.get(id)!.config } : null);
  if (!entry) return `<div class="card empty">Not loaded. <a class="more inline" href="#/">Back</a></div>`;
  const model = readConfig(entry.config);
  const r = run(model);
  if (!r) return `<div class="title"><h1>${esc(entry.name)}</h1></div><div class="filters">${controls()}</div><div class="card empty">Cannot split ${state.tp} ways</div>`;
  const minedLayers = r.rows.filter((x) => x.tile).reduce((s, x) => s + x.layers, 0);
  const allLayers = r.rows.reduce((s, x) => s + x.layers, 0);
  const rows = r.rows
    .map((x) => `<tr><td class="mono">${esc(x.name)}</td><td class="n">${int(x.n)}</td><td class="n">${int(x.k)}</td>
      <td class="n">${int(x.localN)} × ${int(x.localK)}</td><td class="n">${x.layers}</td><td>${x.tile ?? '-'}</td>
      <td class="n">${pct(x.work / r.work)}</td><td><span class="tag ${x.tile ? 'ok' : x.why === 'not a linear layer in vLLM' ? 'plain' : 'off'}">${x.tile ? 'mines' : esc(reason(x))}</span></td></tr>`)
    .join('');
  return `<div class="title"><h1>${esc(entry.name)}</h1><span class="chip mono">${esc(id)}</span></div>
  <div class="filters">${controls()}</div>
  ${stats([
    ['Mined', pct(r.share), 'of linear-layer work'],
    ['PRL / 1M tokens', prlPerMillion(r), 'prompt tokens'],
    ['TP', `${r.tp}${r.fits ? '' : '+'}`, `${gb(r.bytesPerGpu)} weights per GPU`],
    ['Mined layers', `${int(minedLayers)} / ${int(allLayers)}`, model.format.toUpperCase()],
  ])}
  <div class="card">
    <div class="head"><h2>Work per prompt token</h2><span class="muted">${sig(r.work / 1e9)} G MAC</span></div>
    <div class="mbar wide"><i style="width:${(r.share * 100).toFixed(1)}%">${r.share > 0.12 ? `Mines ${pct(r.share)}` : ''}</i><i class="off" style="width:${((1 - r.share) * 100).toFixed(1)}%"></i></div>
    <div class="legend"><span><b></b>Mines</span><span><b class="off"></b>Does not mine</span>${model.omitted.map((o) => `<span>Not counted: ${esc(o)}</span>`).join('')}</div>
  </div>
  <div class="card">
    <div class="head"><h2>Layers</h2></div>
    <div class="scroll"><table>
      <thead><tr><th>Linear</th><th class="n">n</th><th class="n">k</th><th class="n">Per GPU</th><th class="n">Layers</th><th>Tile</th><th class="n">Work</th><th></th></tr></thead>
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
  if (KNOWN.some((k) => k.id === id) || extra.has(id)) {
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
  document.getElementById('tip')!.innerHTML = state.tip ? `<span class="live"></span>mainnet <b>${int(state.tip.height)}</b>` : state.tipError ? 'chain unreachable' : 'mainnet';
  document.getElementById('foot')!.innerHTML = `Rules from Pearl's fp8-scheme branch at ${RULES} · chain data from pearlchain.live · made with pearljs`;
}

app.addEventListener('click', (e) => {
  const b = (e.target as HTMLElement).closest('button');
  if (!b) return;
  if (b.dataset.gpu) state.gpu = b.dataset.gpu;
  if (b.dataset.tp) state.tp = b.dataset.tp === 'auto' ? 'auto' : Number(b.dataset.tp);
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
