// Pre-fork rules: thresholds from Pearl's master config.yaml, layer formats from Pearl's own checkpoints.
import { describe, expect, it } from 'vitest';
import { analyze } from '../src/analyze.js';
import { readConfig } from '../src/arch.js';
import { gpu } from '../src/gpus.js';
import { KNOWN } from '../src/known.js';
import { analyzeNow, isInt7, MIN_K, MIN_N } from '../src/now.js';

const known = (id: string) => KNOWN.find((k) => k.id === id)!;
const now = (id: string, tp: number) => analyzeNow(known(id).int7!.quant, analyze(readConfig(known(id).config), gpu('h100'), tp)!);

describe('pre-fork mining', () => {
  it("uses the thresholds of Pearl's master config.yaml", () => {
    expect([MIN_N, MIN_K]).toEqual([256, 1024]);
  });

  it("follows Llama 3.3 70B's INT7 layers: qkv from layer 40, never down_proj", () => {
    const q = known('meta-llama/Llama-3.3-70B-Instruct').int7!.quant;
    expect(isInt7('model.layers.39.self_attn.q_proj', q)).toBe(false);
    expect(isInt7('model.layers.40.self_attn.q_proj', q)).toBe(true);
    expect(isInt7('model.layers.0.mlp.down_proj', q)).toBe(false);
    const rows = Object.fromEntries(now('meta-llama/Llama-3.3-70B-Instruct', 4).rows.map((r) => [r.name, r.minedLayers]));
    expect(rows).toMatchObject({ qkv_proj: 40, o_proj: 80, gate_up_proj: 80, down_proj: 0, lm_head: 0 });
  });

  it("mines Gemma 4's gate/up, which its Linear group covers, but not its FP8 attention inputs", () => {
    const rows = Object.fromEntries(now('google/gemma-4-31B-it', 2).rows.map((r) => [r.name, r.minedLayers]));
    expect(rows['gate_up_proj']).toBe(60);
    expect(rows['qkv_proj (sliding)']).toBe(0);
  });

  it('mines Qwen3 30B A3B expert gate/up but not the FP8 expert down projection', () => {
    const rows = Object.fromEntries(now('Qwen/Qwen3-30B-A3B-Instruct-2507', 1).rows.map((r) => [r.name, r.minedLayers]));
    expect(rows).toMatchObject({ 'experts.w13 (gate/up)': 48, 'experts.w2 (down)': 0 });
  });
});

describe('credited work', () => {
  it('credits k rounded down to the noise rank of 128', () => {
    const quant = { config_groups: { g: { weights: { type: 'int', num_bits: 7 }, targets: ['Linear'] } } };
    const row = { name: 'o_proj', n: 4096, k: 1100, localN: 4096, localK: 1100, layers: 2, tile: null, why: null, work: 1100, minedWork: 0, minedLayers: 0 };
    const fp8 = { tp: 1, pp: 1, rows: [row], work: 1100, mined: 0, share: 0, bytesPerGpu: 0, parts: { weights: 0, reserve: 0, cache: 0 }, fits: true };
    expect(analyzeNow(quant, fp8).rows[0]!.minedWork).toBe(1024);
  });
});
