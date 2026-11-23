// What share of a model's prompt work mines on a given GPU and tensor-parallel
// size. Work is counted in multiply-accumulates per prompt token over every
// rank, since each rank mines its own shard and is credited m*n*k for it
// (mining_config.py effective_work_per_matmul).
import type { Format, Linear, Model } from './arch.js';
import type { Gpu } from './gpus.js';
import { denseTile, expertTile, type Tile } from './rules.js';

export interface Row {
  name: string;
  n: number;
  k: number;
  /** The shard one GPU holds. */
  localN: number;
  localK: number;
  layers: number;
  tile: Tile | null;
  /** Why it does not mine, when it doesn't. */
  why: string | null;
  /** Multiply-accumulates per prompt token, all layers and ranks. */
  work: number;
}

export interface Result {
  tp: number;
  rows: Row[];
  work: number;
  mined: number;
  share: number;
  /** Weight bytes per GPU at this split. */
  bytesPerGpu: number;
  fits: boolean;
}

const BYTES: Record<Format, number> = { bf16: 2, fp8: 1.03, nvfp4: 0.56, mxfp4: 0.53, other: 2 };
// Room left for activations, the KV cache and the miner's own workspaces.
const HEADROOM = 0.75;

function shard(l: Linear, tp: number): { n: number; k: number } | null {
  if (l.split === 'replicated' || tp === 1) return { n: l.n, k: l.k };
  if (l.split === 'row') return l.k % tp === 0 ? { n: l.n, k: l.k / tp } : null;
  if (l.heads) {
    // vLLM replicates key/value heads when there are fewer of them than ranks.
    const { q, kv, dim } = l.heads;
    if (q % tp !== 0 || (kv >= tp ? kv % tp !== 0 : tp % kv !== 0)) return null;
    return { n: (q / tp + 2 * Math.max(1, kv / tp)) * dim, k: l.k };
  }
  return l.n % tp === 0 ? { n: l.n / tp, k: l.k } : null;
}

export function params(m: Model): number {
  let total = m.linears.reduce((s, l) => s + l.n * l.k * l.layers, 0);
  if (m.tiedEmbeddings) total -= m.vocab * m.hidden;
  total += m.vocab * m.hidden;
  if (m.experts) total += m.experts.count * m.experts.layers * (m.experts.nE * m.experts.hidden + m.experts.hidden * m.experts.intermediate);
  return total;
}

/** Null when the model cannot be split this many ways. */
export function analyze(m: Model, gpu: Gpu, tp: number): Result | null {
  const rows: Row[] = [];
  for (const l of m.linears) {
    const s = shard(l, tp);
    if (s === null) return null;
    const tile = l.never ? null : denseTile(s.n, s.k, gpu.arch);
    rows.push({ name: l.name, n: l.n, k: l.k, localN: s.n, localK: s.k, layers: l.layers, tile, why: l.never ?? (tile ? null : 'shape'), work: s.n * s.k * tp * l.layers });
  }
  const e = m.experts;
  if (e && e.layers > 0) {
    if (e.intermediate % tp !== 0) return null;
    const nE = e.nE / tp;
    const tile = e.blocked ? null : expertTile(nE, e.hidden, e.count, gpu.arch);
    rows.push({ name: 'experts.w13 (gate/up)', n: e.nE, k: e.hidden, localN: nE, localK: e.hidden, layers: e.layers, tile, why: e.blocked ?? (tile ? null : 'shape'), work: e.topK * e.nE * e.hidden * e.layers });
    rows.push({ name: 'experts.w2 (down)', n: e.hidden, k: e.intermediate, localN: e.hidden, localK: e.intermediate / tp, layers: e.layers, tile: null, why: 'never mined', work: e.topK * e.hidden * e.intermediate * e.layers });
  }
  const work = rows.reduce((s, r) => s + r.work, 0);
  const mined = rows.reduce((s, r) => s + (r.tile ? r.work : 0), 0);
  const bytesPerGpu = (params(m) * BYTES[m.format]) / tp;
  return { tp, rows, work, mined, share: work > 0 ? mined / work : 0, bytesPerGpu, fits: bytesPerGpu <= gpu.memory * 1e9 * HEADROOM };
}

/** The smallest split of 1, 2, 4 or 8 that holds the weights, or 8 when none does. */
export function autoTp(m: Model, gpu: Gpu): Result | null {
  let last: Result | null = null;
  for (const tp of [1, 2, 4, 8]) {
    const r = analyze(m, gpu, tp);
    if (r === null) continue;
    last = r;
    if (r.fits) return r;
  }
  return last;
}
