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
  // Rows tensor parallelism does not split, such as replicated shards of a merged projection.
  fixed?: number;
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

// Architectures whose layers this reader builds as vLLM does. Newer ones with
// linear or compressed attention have other linears and must not be guessed.
const SUPPORTED = new Set(['llama', 'mistral', 'qwen2', 'qwen3', 'qwen2_moe', 'qwen3_moe', 'mixtral', 'gemma', 'gemma2', 'gemma3_text', 'gemma4_text', 'deepseek_v2', 'deepseek_v3', 'kimi_k2', 'glm_moe_dsa', 'gpt_oss', 'glm4_moe', 'llama4_text', 'qwen3_next', 'deepseek_v4', 'deepseek_v41_text', 'glm5_next_text', 'qwen3_5_text']);

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
  const step = num(c, 'decoder_sparse_step', 'moe_layer_freq', 'interleave_moe_layer_step') ?? 1;
  const mlpOnly = Array.isArray(c.mlp_only_layers) ? (c.mlp_only_layers as number[]) : [];
  let sparse = 0;
  for (let i = 0; i < total; i++) if (i >= firstDense && (i + 1) % step === 0 && !mlpOnly.includes(i)) sparse += 1;
  return sparse;
}

export function readConfig(root: Config): Model {
  const c = (root.text_config ?? root) as Config;
  const type = String(c.model_type ?? root.model_type ?? '');
  if (!SUPPORTED.has(type)) throw new UnsupportedModel(`${type || 'this'} architecture is not supported yet`);
  const fmt = format(root, c);
  const hidden = need(c, 'hidden_size');
  const layers = need(c, 'num_hidden_layers');
  const vocab = need(c, 'vocab_size');
  const heads = need(c, 'num_attention_heads');
  const kvHeads = num(c, 'num_key_value_heads') ?? heads;
  const linears: Linear[] = [];
  const omitted: string[] = [];

  if (fmt === 'other') throw new UnsupportedModel('quantized in a format the FP8 miner does not load');
  const types = Array.isArray(c.layer_types) ? (c.layer_types as string[]) : [];
  const count = (t: string): number => types.slice(0, layers).filter((x) => x === t).length;
  const add = (l: Linear): void => {
    if (l.layers > 0) linears.push(l);
  };
  if (type === 'deepseek_v4' || type === 'deepseek_v41_text') {
    // deepseek_v4/attention.py: wo_a runs as DeepGEMM on its raw weight
    // (nvidia/ops/o_proj.py), so like MLA's kv_b_proj it never reaches Pearl.
    const headDim = need(c, 'head_dim');
    const qLora = need(c, 'q_lora_rank');
    const groups = need(c, 'o_groups');
    const oLora = need(c, 'o_lora_rank');
    add({ name: 'fused_wqa_wkv', n: qLora + headDim, k: hidden, split: 'replicated', layers });
    add({ name: 'wq_b', n: heads * headDim, k: qLora, split: 'column', layers });
    add({ name: 'wo_a', n: groups * oLora, k: (heads * headDim) / groups, split: 'column', layers, never: 'read directly by DeepGEMM' });
    add({ name: 'wo_b', n: hidden, k: groups * oLora, split: 'row', layers });
    const ratios = Array.isArray(c.compress_ratios) ? (c.compress_ratios as number[]).slice(0, layers) : [];
    const overlapped = ratios.filter((r) => r === 4).length;
    // compressor.py: fused wkv/wgate, twice as wide where the ratio-4 windows overlap, replicated and never quantized.
    const raw = 'built without a quant config';
    add({ name: 'compressor (ratio 4)', n: 4 * headDim, k: hidden, split: 'replicated', layers: overlapped, never: raw });
    add({ name: 'compressor', n: 2 * headDim, k: hidden, split: 'replicated', layers: ratios.filter((r) => r > 1 && r !== 4).length, never: raw });
    if (type === 'deepseek_v4') {
      const indexDim = need(c, 'index_head_dim');
      const indexHeads = need(c, 'index_n_heads');
      add({ name: 'indexer.wq_b', n: indexHeads * indexDim, k: qLora, split: 'replicated', layers: overlapped });
      add({ name: 'indexer.compressor', n: 4 * indexDim, k: hidden, split: 'replicated', layers: overlapped, never: raw });
      add({ name: 'indexer.weights_proj', n: indexHeads, k: hidden, split: 'replicated', layers: overlapped, never: raw });
    }
    omitted.push('hyper-connection mixing');
    if (type === 'deepseek_v41_text') omitted.push('engram and candidate indexer');
  } else if (type === 'qwen3_5_text' || type === 'qwen3_next') {
    // Every full_attention_interval-th layer is full attention; the rest are Gated DeltaNet (qwen_gdn_linear_attn.py).
    const full = Math.floor(layers / need(c, 'full_attention_interval'));
    const keyDim = need(c, 'linear_key_head_dim') * need(c, 'linear_num_key_heads');
    const valueDim = need(c, 'linear_value_head_dim') * need(c, 'linear_num_value_heads');
    add({ name: 'in_proj_qkvz', n: 2 * keyDim + 2 * valueDim, k: hidden, split: 'column', layers: layers - full });
    add({ name: 'in_proj_ba', n: 2 * need(c, 'linear_num_value_heads'), k: hidden, split: 'column', layers: layers - full });
    add({ name: 'out_proj', n: hidden, k: valueDim, split: 'row', layers: layers - full });
    const headDim = need(c, 'head_dim');
    // attn_output_gate doubles the query heads in qkv_proj; Qwen3-Next has it unless the config says otherwise.
    const q = c.attn_output_gate === true || (type === 'qwen3_next' && c.attn_output_gate !== false) ? 2 * heads : heads;
    add({ name: 'qkv_proj', n: (q + 2 * kvHeads) * headDim, k: hidden, split: 'column', layers: full, heads: { q, kv: kvHeads, dim: headDim } });
    add({ name: 'o_proj', n: hidden, k: heads * headDim, split: 'row', layers: full });
    omitted.push('the DeltaNet convolution, built without a quant config');
  } else if (type === 'glm5_next_text') {
    // glm5next/common/kda.py for linear-attention layers, attention.py's MLA for the rest.
    const lin = (c.linear_attn_config ?? {}) as Config;
    const kdaHeads = need(lin, 'num_heads');
    const kdaDim = need(lin, 'head_dim');
    const kda = count('linear_attention');
    add({ name: 'in_proj_qkvbfg_a', n: 3 * kdaHeads * kdaDim + kdaHeads + 2 * kdaDim, k: hidden, split: 'column', layers: kda, fixed: 2 * kdaDim });
    add({ name: 'f_b_proj', n: kdaHeads * kdaDim, k: kdaDim, split: 'column', layers: kda });
    add({ name: 'g_b_proj', n: kdaHeads * kdaDim, k: kdaDim, split: 'column', layers: kda });
    add({ name: 'kda.o_proj', n: hidden, k: kdaHeads * kdaDim, split: 'row', layers: kda });
    mla(count('deepseek_sparse_attention'));
    omitted.push('hyper-connection mixing');
  } else if (num(c, 'kv_lora_rank') !== undefined) {
    mla(layers);
  } else {
    const headDim = num(c, 'head_dim') ?? hidden / heads;
    const globalDim = num(c, 'global_head_dim');
    const globalKv = num(c, 'num_global_key_value_heads');
    // Gemma 4's full-attention layers use wider heads and fewer key/value heads than its sliding ones.
    const full = globalDim !== undefined ? count('full_attention') : 0;
    const groups = full > 0 ? [{ layers: layers - full, dim: headDim, kv: kvHeads, tag: ' (sliding)' }, { layers: full, dim: globalDim!, kv: globalKv ?? kvHeads, tag: ' (full)' }] : [{ layers, dim: headDim, kv: kvHeads, tag: '' }];
    for (const g of groups) {
      linears.push({ name: `qkv_proj${g.tag}`, n: (heads + 2 * g.kv) * g.dim, k: hidden, split: 'column', layers: g.layers, heads: { q: heads, kv: g.kv, dim: g.dim } });
      linears.push({ name: `o_proj${g.tag}`, n: hidden, k: heads * g.dim, split: 'row', layers: g.layers });
    }
  }

  // MLA (DeepSeek V2/V3, Kimi K2, GLM-5): vLLM fuses q_a and kv_a into one replicated projection.
  function mla(n: number): void {
    const qLora = num(c, 'q_lora_rank');
    const nope = need(c, 'qk_nope_head_dim');
    const rope = num(c, 'qk_rope_head_dim') ?? 0;
    const vDim = need(c, 'v_head_dim');
    const kvLora = need(c, 'kv_lora_rank');
    if (qLora) {
      add({ name: 'fused_qkv_a_proj', n: qLora + kvLora + rope, k: hidden, split: 'replicated', layers: n });
      add({ name: 'q_b_proj', n: heads * (nope + rope), k: qLora, split: 'column', layers: n });
    } else {
      add({ name: 'q_proj', n: heads * (nope + rope), k: hidden, split: 'column', layers: n });
      add({ name: 'kv_a_proj_with_mqa', n: kvLora + rope, k: hidden, split: 'replicated', layers: n });
    }
    add({ name: 'kv_b_proj', n: heads * (nope + vDim), k: kvLora, split: 'column', layers: n, never: 'read directly by MLA' });
    add({ name: 'o_proj', n: hidden, k: heads * vDim, split: 'row', layers: n });
    if (num(c, 'index_n_heads') !== undefined) omitted.push('sparse-attention indexer projections');
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
    else if (type === 'llama4_text') blocked = 'router weights applied on the input';
    else if (type === 'gpt_oss') blocked = 'expert biases';
    else if (fmt !== 'bf16') blocked = `${fmt.toUpperCase()} experts`;
    else if (c.hidden_act !== undefined && c.hidden_act !== 'silu') blocked = `activation ${String(c.hidden_act)}`;
    experts = { count: expertCount, topK: need(c, 'num_experts_per_tok', 'experts_per_token'), nE: 2 * inter, hidden, intermediate: inter, layers: sparse, ...(blocked ? { blocked } : {}) };
    // Llama 4 always has one shared expert the size of a routed one (llama4.py).
    const shared = (num(c, 'n_shared_experts') ?? (type === 'llama4_text' ? 1 : 0)) * inter || (num(c, 'shared_expert_intermediate_size') ?? 0);
    if (shared > 0) {
      linears.push({ name: 'shared_experts.gate_up_proj', n: 2 * shared, k: hidden, split: 'column', layers: sparse });
      linears.push({ name: 'shared_experts.down_proj', n: hidden, k: shared, split: 'row', layers: sparse });
    }
  }
  if (denseLayers > 0) {
    const inter = need(c, 'intermediate_size_mlp', 'intermediate_size');
    linears.push({ name: 'gate_up_proj', n: 2 * inter, k: hidden, split: 'column', layers: denseLayers });
    linears.push({ name: 'down_proj', n: hidden, k: inter, split: 'row', layers: denseLayers });
  }
  linears.push({ name: 'lm_head', n: vocab, k: hidden, split: 'column', layers: 1, never: 'not a linear layer in vLLM' });
  if ((num(c, 'num_nextn_predict_layers') ?? 0) > 0) omitted.push('multi-token prediction layers');

  return { type, format: fmt, hidden, vocab, tiedEmbeddings: c.tie_word_embeddings === true, linears, experts, omitted };
}
