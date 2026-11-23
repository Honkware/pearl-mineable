// Config reading and mined share for the models Pearl builds around and a few
// popular ones. Expected shapes are each model's real layer sizes; the GLM-5.2
// ones are the (n, k) pairs Pearl's own fp8-scheme tests use for that model.
import { describe, expect, it } from 'vitest';
import { analyze, autoTp } from '../src/analyze.js';
import { readConfig, UnsupportedModel } from '../src/arch.js';
import { gpu } from '../src/gpus.js';
import { KNOWN } from '../src/known.js';

const model = (id: string) => readConfig(KNOWN.find((m) => m.id === id)!.config);
const shapes = (id: string) => Object.fromEntries(model(id).linears.map((l) => [l.name, [l.n, l.k]]));
const row = (id: string, tp: number, name: string, card = 'h100') => analyze(model(id), gpu(card), tp)!.rows.find((r) => r.name === name)!;

describe('reading configs', () => {
  it('builds Llama 3.3 70B fused qkv and gate/up the way vLLM does', () => {
    expect(shapes('meta-llama/Llama-3.3-70B-Instruct')).toMatchObject({ qkv_proj: [10240, 8192], o_proj: [8192, 8192], gate_up_proj: [57344, 8192], down_proj: [8192, 28672] });
  });

  it("gives GLM-5.2 the dense and attention shapes Pearl's tests use", () => {
    expect(shapes('zai-org/GLM-5.2')).toMatchObject({ q_b_proj: [16384, 2048], o_proj: [6144, 16384], gate_up_proj: [24576, 6144], down_proj: [6144, 12288] });
    expect(model('zai-org/GLM-5.2').experts).toMatchObject({ nE: 4096, hidden: 6144, count: 256, layers: 75 });
  });

  it("splits Gemma 4's full-attention layers from its sliding ones", () => {
    const s = shapes('google/gemma-4-31B-it');
    expect(s['qkv_proj (sliding)']).toEqual([(32 + 32) * 256, 5376]);
    expect(s['qkv_proj (full)']).toEqual([(32 + 8) * 512, 5376]);
  });

  it('refuses a pre-fork Pearl checkpoint by name', () => {
    expect(() => readConfig({ ...KNOWN[0]!.config, quantization_config: { quant_method: 'pearl' } })).toThrow(UnsupportedModel);
  });
});

describe('what mines', () => {
  it('mines every Llama 3.3 70B layer except the LM head', () => {
    const r = autoTp(model('meta-llama/Llama-3.3-70B-Instruct'), gpu('h100'))!;
    expect(r.tp).toBe(4);
    expect(r.rows.filter((x) => !x.tile).map((x) => x.name)).toEqual(['lm_head']);
  });

  it('mines GLM-5.2 routed experts in BF16 but never its MLA kv_b_proj', () => {
    expect(row('zai-org/GLM-5.2', 8, 'experts.w13 (gate/up)', 'b200').tile).toBe('4x64');
    expect(row('zai-org/GLM-5.2', 8, 'kv_b_proj', 'b200').why).toBe('read directly by MLA');
  });

  it('leaves FP8 experts unmined while the dense layers of the same checkpoint mine', () => {
    expect(row('deepseek-ai/DeepSeek-V3', 8, 'experts.w13 (gate/up)', 'b200').why).toBe('FP8 experts');
    expect(row('deepseek-ai/DeepSeek-V3', 8, 'o_proj', 'b200').tile).not.toBeNull();
  });

  it("blocks gpt-oss experts for their SwiGLU limit", () => {
    expect(row('openai/gpt-oss-120b', 1, 'experts.w13 (gate/up)').why).toBe('swiglu gate parameters');
  });

  it('keeps Gemma 4 qkv and gate/up off the miner because 5,376 is not a multiple of 512', () => {
    expect(row('google/gemma-4-31B-it', 1, 'qkv_proj (sliding)').tile).toBeNull();
    expect(row('google/gemma-4-31B-it', 1, 'gate_up_proj').tile).toBeNull();
    expect(row('google/gemma-4-31B-it', 1, 'down_proj').tile).toBe('4x64');
  });

  it('loses a layer when a split takes its k below 1,024', () => {
    expect(row('Qwen/Qwen3-0.6B', 1, 'down_proj').tile).not.toBeNull();
    expect(row('Qwen/Qwen3-0.6B', 4, 'down_proj').tile).toBeNull();
  });
});

describe('architectures', () => {
  it('refuses a model with linear attention instead of reading it as a plain transformer', () => {
    expect(() => readConfig({ model_type: 'qwen3_5_text', hidden_size: 5120, num_hidden_layers: 64, vocab_size: 248320, num_attention_heads: 24, intermediate_size: 17408 })).toThrow(UnsupportedModel);
  });

  it("reads GLM-5.3 as GLM-5.2's architecture with FP8 experts", () => {
    expect(row('zai-org/GLM-5.3', 8, 'experts.w13 (gate/up)', 'b200').why).toBe('FP8 experts');
  });
});

describe('architectures with new attention', () => {
  it("builds DeepSeek V4's attention as vLLM does, with wo_a off the miner", () => {
    expect(shapes('deepseek-ai/DeepSeek-V4-Pro-0813')).toMatchObject({ fused_wqa_wkv: [2048, 7168], wq_b: [65536, 1536], wo_a: [16384, 4096], wo_b: [7168, 16384] });
    expect(row('deepseek-ai/DeepSeek-V4-Pro-0813', 8, 'wo_a', 'b200').why).toBe('read directly by DeepGEMM');
    expect(row('deepseek-ai/DeepSeek-V4-Pro-0813', 8, 'experts.w13 (gate/up)', 'b200').why).toBe('swiglu gate parameters');
  });

  it('splits Qwen3.8 into Gated DeltaNet and gated full-attention layers', () => {
    const m = model('Qwen/Qwen3.8-27B');
    expect(shapes('Qwen/Qwen3.8-27B')).toMatchObject({ in_proj_qkvz: [16384, 5120], out_proj: [5120, 6144], qkv_proj: [14336, 5120] });
    expect(m.linears.find((l) => l.name === 'in_proj_qkvz')!.layers).toBe(48);
    expect(m.linears.find((l) => l.name === 'qkv_proj')!.layers).toBe(16);
  });

  it('gives GLM-5.3 Flash 34 linear-attention layers and 11 MLA layers', () => {
    const m = model('zai-org/GLM-5.3-Flash');
    expect(m.linears.find((l) => l.name === 'in_proj_qkvbfg_a')).toMatchObject({ n: 24896, k: 4096, layers: 34 });
    expect(m.linears.find((l) => l.name === 'q_b_proj')!.layers).toBe(11);
  });

  it('keeps the replicated shards of a merged projection whole when splitting', () => {
    expect(row('zai-org/GLM-5.3-Flash', 4, 'in_proj_qkvbfg_a', 'b200').localN).toBe((3 * 8192 + 64) / 4 + 256);
  });
});

describe('more served architectures', () => {
  it('reads GLM-4.6 with its 96 query heads and BF16 experts that mine', () => {
    expect(shapes('zai-org/GLM-4.6')).toMatchObject({ qkv_proj: [14336, 5120], o_proj: [5120, 12288] });
    expect(row('zai-org/GLM-4.6', 8, 'experts.w13 (gate/up)').tile).toBe('4x64');
  });

  it("gives Llama 4 Maverick dense layers between its MoE ones and blocks its experts for input-side router weights", () => {
    const m = model('meta-llama/Llama-4-Maverick-17B-128E-Instruct');
    expect(m.experts!.layers).toBe(24);
    expect(m.linears.find((l) => l.name === 'gate_up_proj')).toMatchObject({ n: 32768, layers: 24 });
    expect(row('meta-llama/Llama-4-Maverick-17B-128E-Instruct', 8, 'experts.w13 (gate/up)').why).toBe('router weights applied on the input');
  });

  it("doubles Qwen3-Next's query heads for its output gate by default", () => {
    expect(shapes('Qwen/Qwen3-Next-80B-A3B-Instruct')).toMatchObject({ qkv_proj: [(32 + 4) * 256, 2048], in_proj_qkvz: [12288, 2048] });
  });
});
