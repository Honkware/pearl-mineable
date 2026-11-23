// Work is counted over every rank: each rank mines its own shard and is
// credited m*n*k for it (mining_config.py effective_work_per_matmul).
import type { Cache, Format, Linear, Model } from './arch.js';
import type { Gpu } from './gpus.js';
import { denseTile, expertTile, type Tile } from './rules.js';

export interface Row {
  name: string;
  n: number;
  k: number;
  localN: number;
  localK: number;
  layers: number;
  tile: Tile | null;
  why: string | null;
  // MACs per prompt token, all layers and ranks.
  work: number;
  minedWork: number;
  minedLayers: number;
}

export interface Result {
  tp: number;
  // Pipeline stages: each holds a share of the layers, so shapes follow tp alone.
  pp: number;
  rows: Row[];
  work: number;
  mined: number;
  share: number;
  bytesPerGpu: number;
  // Its parts: weights and the miner's planes, the miner's runtime reservation, one sequence of cache.
  parts: { weights: number; reserve: number; cache: number };
  fits: boolean;
}

const BYTES: Record<Format, number> = { bf16: 2, fp8: 1.03, nvfp4: 0.56, mxfp4: 0.53, other: 2 };
// GPU bytes per weight of a layer Pearl's FP8 miner mines, which replace the
// source weight (state.py): int8 codes 1, BF16 scales per 8 weights 0.25,
// b_prime 1, commit stats and hash roots about 0.06, and for dense layers the
// FP8 fallback copy 1 (experts have none).
const MINED_EXPERT = 1 + 0.25 + 1 + 0.06;
const MINED_DENSE = MINED_EXPERT + 1;
// Pearl's example command: vllm serve ... --max-model-len 8192 --gpu-memory-utilization 0.9 --enforce-eager.
export const CONTEXT = 8192;
const UTILIZATION = 0.9;
// vLLM's activation peak while profiling, which this does not model; eager mode keeps no CUDA graphs.
const OVERHEAD = 2e9;

export function shard(l: Linear, tp: number): { n: number; k: number } | null {
  if (l.split === 'replicated' || tp === 1) return { n: l.n, k: l.k };
  if (l.split === 'row') return l.k % tp === 0 ? { n: l.n, k: l.k / tp } : null;
  if (l.heads) {
    // vLLM replicates key/value heads when there are fewer of them than ranks.
    const { q, kv, dim } = l.heads;
    if (q % tp !== 0 || (kv >= tp ? kv % tp !== 0 : tp % kv !== 0)) return null;
    return { n: (q / tp + 2 * Math.max(1, kv / tp)) * dim, k: l.k };
  }
  const split = l.n - (l.fixed ?? 0);
  return split % tp === 0 ? { n: split / tp + (l.fixed ?? 0), k: l.k } : null;
}

/** Null when the model cannot be split this many ways. */
export function analyze(m: Model, gpu: Gpu, tp: number, pp = 1): Result | null {
  const rows: Row[] = [];
  for (const l of m.linears) {
    const s = shard(l, tp);
    if (s === null) return null;
    const tile = l.never ? null : denseTile(s.n, s.k, gpu.arch);
    rows.push({ name: l.name, n: l.n, k: l.k, localN: s.n, localK: s.k, layers: l.layers, tile, why: l.never ?? (tile ? null : 'shape'), work: s.n * s.k * tp * l.layers, minedWork: tile ? s.n * s.k * tp * l.layers : 0, minedLayers: tile ? l.layers : 0 });
  }
  const e = m.experts;
  if (e && e.layers > 0) {
    if (e.intermediate % tp !== 0) return null;
    const nE = e.nE / tp;
    const tile = e.blocked ? null : expertTile(nE, e.hidden, e.count, gpu.arch);
    rows.push({ name: 'experts.w13 (gate/up)', n: e.nE, k: e.hidden, localN: nE, localK: e.hidden, layers: e.layers, tile, why: e.blocked ?? (tile ? null : 'shape'), work: e.topK * e.nE * e.hidden * e.layers, minedWork: tile ? e.topK * e.nE * e.hidden * e.layers : 0, minedLayers: tile ? e.layers : 0 });
    rows.push({ name: 'experts.w2 (down)', n: e.hidden, k: e.intermediate, localN: e.hidden, localK: e.intermediate / tp, layers: e.layers, tile: null, why: 'never mined', work: e.topK * e.hidden * e.intermediate * e.layers, minedWork: 0, minedLayers: 0 });
  }
  const work = rows.reduce((s, r) => s + r.work, 0);
  const mined = rows.reduce((s, r) => s + r.minedWork, 0);
  // Per GPU: every layer's shard, mined ones in the miner's planes, the rest in the
  // checkpoint's format (the LM head and embeddings in BF16), over the pipeline stages.
  let bytes = (m.tiedEmbeddings ? 0 : (m.vocab * m.hidden * 2) / tp);
  for (const r of rows) {
    const shardWeights = r.localN * r.localK * r.layers;
    if (r.name.startsWith('experts.')) {
      const all = e!.count * shardWeights;
      bytes += all * (r.tile ? MINED_EXPERT : BYTES[m.format]);
    } else {
      bytes += shardWeights * (r.tile ? MINED_DENSE : r.name === 'lm_head' ? 2 : BYTES[m.format]);
    }
  }
  const parts = { weights: bytes / pp, reserve: minerReserve(rows, e), cache: cacheBytes(m.cache, tp) / pp };
  const bytesPerGpu = parts.weights + parts.reserve + parts.cache + OVERHEAD;
  return { tp, pp, rows, work, mined, share: work > 0 ? mined / work : 0, bytesPerGpu, parts, fits: bytesPerGpu <= gpu.memory * 1e9 * UTILIZATION };
}

const MAX_TP = 8; // one node; vLLM splits further by pipeline stages

/** The fewest GPUs, in powers of two, that hold the weights: heads first, then pipeline stages. */
export function autoTp(m: Model, gpu: Gpu): Result | null {
  for (let gpus = 1; gpus <= 256; gpus *= 2) {
    const r = onGpus(m, gpu, gpus);
    if (r?.fits) return r;
  }
  return onGpus(m, gpu, 256);
}

/** A given number of GPUs: the widest head split of at most 8 that divides it and the model, the rest as pipeline stages. */
export function onGpus(m: Model, gpu: Gpu, gpus: number): Result | null {
  for (let tp = Math.min(MAX_TP, gpus); tp >= 1; tp--) {
    if (gpus % tp !== 0) continue;
    const r = analyze(m, gpu, tp, gpus / tp);
    if (r) return r;
  }
  return null;
}

/** One CONTEXT-token sequence in vLLM's cache on one GPU, which vLLM needs before it starts. */
export function cacheBytes(cache: Cache[], tp: number): number {
  let total = 0;
  for (const c of cache) {
    const windowed = c.window ? Math.min(CONTEXT, c.window) : CONTEXT;
    const tokens = c.ratio ? CONTEXT / c.ratio + (c.window ? windowed : 0) : windowed;
    if (c.perHead) total += c.layers * tokens * c.perHead.bytes * Math.max(1, c.perHead.heads / tp);
    if (c.latent) total += c.layers * tokens * c.latent;
    if (c.state) total += (c.layers * c.state) / tp;
  }
  return total;
}

// memory.py estimate_peak_mining_bytes with the default settings: m buckets up to
// 8192, 8 launches in flight, a hit signal for the largest legal k (58,368).
const MAX_M = 8192;
const SLOTS = 8;
const SIGNAL_K = 58368;
const RANK = 32;
const MIB256 = 256 * 1024 * 1024;

function minerReserve(rows: Row[], e: Model['experts']): number {
  const mined = rows.filter((r) => r.tile);
  if (mined.length === 0) return 0;
  const signal = 2 * (MAX_M * SIGNAL_K + 2 * MAX_M * (SIGNAL_K / 8));
  const launch = (m: number, n: number, k: number, outRows: number): number =>
    m * (2 * k + k + k / 4 + (1.25 * k) / 32 + k / 64 + 4 + RANK + k + 4 * RANK) + 2 * outRows * n;
  let maxLaunch = 0;
  let tail = 0;
  let reload = 0;
  for (const r of mined) {
    if (r.name.startsWith('experts.') && e) {
      const cum = MAX_M * e.topK;
      const gathered = cum * (r.localK + 4 * RANK + 26 + 4.2);
      maxLaunch = Math.max(maxLaunch, gathered + launch(MAX_M, r.localN, r.localK, cum));
      tail = Math.max(tail, 2 * cum * (r.localN / 2) + 6 * cum * r.localK + 6 * cum + 2 * MAX_M * r.localK);
      reload = Math.max(reload, 16 * e.count * r.localN * r.localK);
    } else {
      maxLaunch = Math.max(maxLaunch, launch(MAX_M, r.localN, r.localK, MAX_M));
      reload = Math.max(reload, 16 * r.localN * r.localK);
    }
  }
  const raw = signal + SLOTS * maxLaunch + tail + reload;
  return Math.ceil((Math.ceil(raw * 1.1) + MIB256) / MIB256) * MIB256;
}
