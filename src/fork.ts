// FP8 fork readiness: what mines after the fork, which cards, and whether mainnet
// has switched yet. The live switch is read from the chain tip's certificate version
// (chain.ts: a V4 certificate means FP8 is active); everything else here is a fact
// drawn from Pearl's code and its published fork heights, each with its source.

import type { Arch } from './rules.js';

// FP8 kernels exist for these compute-capability families (rules.ts COMMITTED /
// KERNEL_ROWS, pearl_gemm get_min_capability = SM90). Nothing below 9.0 has a kernel.
export interface ArchLine {
  arch: Arch;
  cc: string; // CUDA compute capability
  family: string;
  cards: string;
}
export const SUPPORTED_ARCHS: ArchLine[] = [
  { arch: 'SM90', cc: '9.0', family: 'Hopper', cards: 'H100, H200, GH200' },
  { arch: 'SM100', cc: '10.0', family: 'Blackwell datacenter', cards: 'B200, GB200, B300, GB300' },
  { arch: 'SM120', cc: '12.0', family: 'Blackwell consumer / RTX PRO', cards: 'RTX 5090 / 5080 / 5070 Ti / 5070 / 5060 Ti, RTX PRO 6000 / 5000 / 4500 / 4000' },
];

// Common cards miners run today that have NO FP8 kernel (compute capability < 9.0):
// they mine INT7 on mainnet now (through third-party miners) but cannot mine Pearl's
// FP8 after the fork unless a miner ships inefficient emulation.
export interface ExcludedLine {
  cards: string;
  cc: string;
  family: string;
}
export const EXCLUDED_CARDS: ExcludedLine[] = [
  { cards: 'RTX 4090 / 4080 / 4070 Ti / 4060', cc: '8.9', family: 'Ada Lovelace' },
  { cards: 'RTX 3090 / 3080 / 3070 / 3060, CMP 170HX', cc: '8.6 / 8.0', family: 'Ampere' },
  { cards: 'A100 / A40 / A10', cc: '8.0 / 8.6', family: 'Ampere datacenter' },
  { cards: 'RTX 2080 Ti / 2070', cc: '7.5', family: 'Turing' },
  { cards: 'V100', cc: '7.0', family: 'Volta' },
];

// Published fork heights. The mainnet activation height is not set; the live tip's
// certificate version is the authoritative "is it on yet" signal (chain.ts).
export interface ForkFact {
  net: string;
  height: number | null; // null = not scheduled
  note: string;
}
export const FORK_FACTS: ForkFact[] = [
  { net: 'testnet', height: 67415, note: 'V4 active' },
  { net: 'testnet2', height: 150283, note: 'V4 active' },
  { net: 'mainnet', height: null, note: 'no height set; watch the tip certificate' },
];

// What the fork changes, from the fp8-scheme branch (README "After FP8").
export const WHAT_CHANGES: string[] = [
  'The proof certificate goes from V3 (INT7) to V4 (FP8 E4M3).',
  'Any BF16 model mines, not only Pearl\'s own INT7 checkpoints — a dense shard mines when select_tile finds a committed tile the card has a kernel for.',
  'Routed experts mine their gate and up projection only; MLA kv_b, the LM head and expert down projections never mine.',
  'Mining needs compute capability 9.0+ (Hopper, Blackwell). Cards below that keep mining INT7 only until the fork, then stop.',
];
