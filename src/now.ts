// Pre-fork rules from Pearl's master miner: the checkpoint stores the layer as
// INT7 and the shard passes should_use_noisy_gemm (config.py). That check also
// wants 1,024 tokens in the batch, so short prompts do not mine. No GPU check:
// the INT7 kernels need SM90, which every listed card has.
import type { Result } from './analyze.js';
import now from '../vectors/now.json' with { type: 'json' };

export const { min_m: MIN_M, min_n: MIN_N, min_k: MIN_K } = now.thresholds;
// miner_base settings.py noise_rank default, which is also the rank-penalty base, so no penalty applies.
const NOISE_RANK = 128;

interface Group {
  weights: { type: string; num_bits: number };
  targets: string[];
}

export interface PearlCheckpoint {
  config_groups: Record<string, Group>;
  ignore?: string[];
}

function modules(linear: string, i: number): string[] {
  const p = `model.layers.${i}`;
  if (linear.startsWith('qkv_proj')) return ['q', 'k', 'v'].map((x) => `${p}.self_attn.${x}_proj`);
  if (linear.startsWith('o_proj')) return [`${p}.self_attn.o_proj`];
  if (linear === 'gate_up_proj') return [`${p}.mlp.gate_proj`, `${p}.mlp.up_proj`];
  if (linear === 'down_proj') return [`${p}.mlp.down_proj`];
  if (linear.startsWith('experts.w13')) return [`${p}.mlp.experts.0.gate_proj`, `${p}.mlp.experts.0.up_proj`];
  if (linear.startsWith('experts.w2')) return [`${p}.mlp.experts.0.down_proj`];
  return [linear];
}

// A regex target outranks a group that targets every Linear.
export function isInt7(name: string, q: PearlCheckpoint): boolean {
  if (q.ignore?.includes(name)) return false;
  const groups = Object.values(q.config_groups);
  const int7 = (g: Group): boolean => g.weights.type === 'int' && g.weights.num_bits === 7;
  const byRegex = groups.find((g) => g.targets.some((t) => t.startsWith('re:') && new RegExp(`^${t.slice(3)}$`).test(name)));
  if (byRegex) return int7(byRegex);
  return groups.some((g) => int7(g) && g.targets.includes('Linear'));
}

// Pearl's checkpoints are all models whose linears appear in every decoder
// layer, so layer i of a row is decoder layer i.
export function analyzeNow(q: PearlCheckpoint, fp8: Result): Result {
  const rows = fp8.rows.map((r) => {
    let layers = 0;
    for (let i = 0; i < r.layers; i++) if (modules(r.name, i).every((name) => isInt7(name, q))) layers += 1;
    const fits = r.localN >= MIN_N && r.localK >= MIN_K;
    const minedLayers = fits ? layers : 0;
    // Credited work uses k rounded down to the noise rank (inner_hash_per_matmul in noisy_gemm.py, rank 128 by default).
    const credited = Math.floor(r.localK / NOISE_RANK) * NOISE_RANK / r.localK;
    return { ...r, minedLayers, minedWork: (r.work * minedLayers * credited) / r.layers };
  });
  const mined = rows.reduce((s, r) => s + r.minedWork, 0);
  return { ...fp8, rows, mined, share: fp8.work > 0 ? mined / fp8.work : 0 };
}
