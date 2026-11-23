// Layers are fused as vLLM fuses them (q/k/v, gate/up, MLA's q_a with kv_a),
// since Pearl's rules apply to the fused shard.

export type Split = 'column' | 'row' | 'replicated';
export type Format = 'bf16' | 'fp8' | 'nvfp4' | 'mxfp4' | 'other';

export interface Linear {
  name: string;
  n: number;
  k: number;
  split: Split;
  layers: number;
  never?: string;
  // vLLM splits qkv by heads, not by rows.
  heads?: { q: number; kv: number; dim: number };
}

export interface Experts {
  count: number;
  topK: number;
  // Rows of one expert's fused gate/up weight, before splitting.
  nE: number;
  hidden: number;
  intermediate: number;
  layers: number;
  blocked?: string;
}

export interface Model {
  type: string;
  format: Format;
  hidden: number;
  vocab: number;
  tiedEmbeddings: boolean;
  linears: Linear[];
  experts: Experts | null;
  omitted: string[];
}

type Config = Record<string, unknown>;

export class UnsupportedModel extends Error {}

const num = (c: Config, ...keys: string[]): number | undefined => {
  for (const key of keys) if (typeof c[key] === 'number') return c[key] as number;
  return undefined;
};
const need = (c: Config, ...keys: string[]): number => {
  const v = num(c, ...keys);
  if (v === undefined) throw new UnsupportedModel(`config has no ${keys.join(' or ')}`);
  return v;
};

function format(root: Config, text: Config): Format {
  const q = (root.quantization_config ?? text.quantization_config) as Config | undefined;
  const method = typeof q?.quant_method === 'string' ? q.quant_method.toLowerCase() : '';
  if (method === '') return 'bf16';
  if (method === 'pearl') throw new UnsupportedModel("a pre-fork Pearl checkpoint; the FP8 miner loads the original BF16 model");
  if (method === 'fp8') return 'fp8';
  if (method === 'mxfp4') return 'mxfp4';
  if (method === 'modelopt_fp4' || (method === 'modelopt' && String(q?.quant_algo ?? '').includes('FP4'))) return 'nvfp4';
  if (method === 'compressed-tensors') {
    const groups = JSON.stringify(q?.config_groups ?? '');
    if (/"num_bits":4/.test(groups) && /"type":"float"/.test(groups)) return 'nvfp4';
    if (/"num_bits":8/.test(groups) && /"type":"float"/.test(groups)) return 'fp8';
  }
  return 'other';
}

function moeLayers(c: Config, total: number): number {
  const firstDense = num(c, 'first_k_dense_replace') ?? 0;
  const step = num(c, 'decoder_sparse_step', 'moe_layer_freq') ?? 1;
  const mlpOnly = Array.isArray(c.mlp_only_layers) ? (c.mlp_only_layers as number[]) : [];
  let sparse = 0;
  for (let i = 0; i < total; i++) if (i >= firstDense && (i + 1) % step === 0 && !mlpOnly.includes(i)) sparse += 1;
  return sparse;
}

export function readConfig(root: Config): Model {
  const c = (root.text_config ?? root) as Config;
  const type = String(c.model_type ?? root.model_type ?? '');
  const fmt = format(root, c);
  const hidden = need(c, 'hidden_size');
  const layers = need(c, 'num_hidden_layers');
  const vocab = need(c, 'vocab_size');
  const heads = need(c, 'num_attention_heads');
  const kvHeads = num(c, 'num_key_value_heads') ?? heads;
  const linears: Linear[] = [];
  const omitted: string[] = [];

  if (fmt === 'other') throw new UnsupportedModel('quantized in a format the FP8 miner does not load');
  if (num(c, 'kv_lora_rank') !== undefined) {
    // MLA (DeepSeek V2/V3, Kimi K2, GLM-5): vLLM fuses q_a and kv_a into one replicated projection.
    const qLora = num(c, 'q_lora_rank');
    const nope = need(c, 'qk_nope_head_dim');
    const rope = need(c, 'qk_rope_head_dim');
    const vDim = need(c, 'v_head_dim');
    const kvLora = need(c, 'kv_lora_rank');
    if (qLora) {
      linears.push({ name: 'fused_qkv_a_proj', n: qLora + kvLora + rope, k: hidden, split: 'replicated', layers });
      linears.push({ name: 'q_b_proj', n: heads * (nope + rope), k: qLora, split: 'column', layers });
    } else {
      linears.push({ name: 'q_proj', n: heads * (nope + rope), k: hidden, split: 'column', layers });
      linears.push({ name: 'kv_a_proj_with_mqa', n: kvLora + rope, k: hidden, split: 'replicated', layers });
    }
    linears.push({ name: 'kv_b_proj', n: heads * (nope + vDim), k: kvLora, split: 'column', layers, never: 'read directly by MLA' });
    linears.push({ name: 'o_proj', n: hidden, k: heads * vDim, split: 'row', layers });
    if (num(c, 'index_n_heads') !== undefined) omitted.push('sparse-attention indexer projections');
  } else {
    const headDim = num(c, 'head_dim') ?? hidden / heads;
    const types = Array.isArray(c.layer_types) ? (c.layer_types as string[]) : [];
    const globalDim = num(c, 'global_head_dim');
    const globalKv = num(c, 'num_global_key_value_heads');
    // Gemma 4's full-attention layers use wider heads and fewer key/value heads than its sliding ones.
    const full = globalDim !== undefined ? types.filter((t) => t === 'full_attention').length : 0;
    const groups = full > 0 ? [{ layers: layers - full, dim: headDim, kv: kvHeads, tag: ' (sliding)' }, { layers: full, dim: globalDim!, kv: globalKv ?? kvHeads, tag: ' (full)' }] : [{ layers, dim: headDim, kv: kvHeads, tag: '' }];
    for (const g of groups) {
      linears.push({ name: `qkv_proj${g.tag}`, n: (heads + 2 * g.kv) * g.dim, k: hidden, split: 'column', layers: g.layers, heads: { q: heads, kv: g.kv, dim: g.dim } });
      linears.push({ name: `o_proj${g.tag}`, n: hidden, k: heads * g.dim, split: 'row', layers: g.layers });
    }
  }

  const expertCount = num(c, 'n_routed_experts', 'num_experts', 'num_local_experts') ?? 0;
  let experts: Experts | null = null;
  let denseLayers = layers;
  if (expertCount > 0) {
    const sparse = moeLayers(c, layers);
    denseLayers = layers - sparse;
    const inter = need(c, 'moe_intermediate_size', 'intermediate_size');
    let blocked: string | undefined;
    if (num(c, 'swiglu_limit') !== undefined) blocked = 'swiglu gate parameters';
    else if (type === 'gpt_oss') blocked = 'expert biases';
    else if (fmt !== 'bf16') blocked = `${fmt.toUpperCase()} experts`;
    else if (c.hidden_act !== undefined && c.hidden_act !== 'silu') blocked = `activation ${String(c.hidden_act)}`;
    experts = { count: expertCount, topK: need(c, 'num_experts_per_tok', 'experts_per_token'), nE: 2 * inter, hidden, intermediate: inter, layers: sparse, ...(blocked ? { blocked } : {}) };
    const shared = (num(c, 'n_shared_experts') ?? 0) * inter || (num(c, 'shared_expert_intermediate_size') ?? 0);
    if (shared > 0) {
      linears.push({ name: 'shared_experts.gate_up_proj', n: 2 * shared, k: hidden, split: 'column', layers: sparse });
      linears.push({ name: 'shared_experts.down_proj', n: hidden, k: shared, split: 'row', layers: sparse });
    }
  }
  if (denseLayers > 0) {
    const inter = need(c, 'intermediate_size');
    linears.push({ name: 'gate_up_proj', n: 2 * inter, k: hidden, split: 'column', layers: denseLayers });
    linears.push({ name: 'down_proj', n: hidden, k: inter, split: 'row', layers: denseLayers });
  }
  linears.push({ name: 'lm_head', n: vocab, k: hidden, split: 'column', layers: 1, never: 'not a linear layer in vLLM' });
  if ((num(c, 'num_nextn_predict_layers') ?? 0) > 0) omitted.push('multi-token prediction layers');

  return { type, format: fmt, hidden, vocab, tiedEmbeddings: c.tie_word_embeddings === true, linears, experts, omitted };
}
