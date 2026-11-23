// Pearl's FP8 rules for whether one layer shard can mine, from the fp8-scheme
// miner. Each check names the function it comes from; vectors/tiles.json holds
// Pearl's own answers over a grid of shapes and test/rules.test.ts holds this
// file to them.

export type Arch = 'SM90' | 'SM100' | 'SM120';
export type Tile = '4x64' | '16x32' | '4x128';

interface TileSpec {
  name: Tile;
  rows: number;
  cols: number;
  maxK: number;
}

// mining_config.py: SMALL_TILE, TALL_TILE, DEFAULT_TILE and _VERIFIER_TILE_MAX_K.
const SMALL: TileSpec = { name: '4x64', rows: 4, cols: 64, maxK: 42496 };
const TALL: TileSpec = { name: '16x32', rows: 16, cols: 32, maxK: 58368 };
const DEFAULT: TileSpec = { name: '4x128', rows: 4, cols: 128, maxK: 22016 };

// _committed_tiles: SM120 commits as Blackwell, so it picks from Blackwell's set.
const COMMITTED: Record<Arch, TileSpec[]> = { SM90: [SMALL, DEFAULT], SM100: [SMALL, TALL, DEFAULT], SM120: [SMALL, TALL, DEFAULT] };
// pearl_gemm _LOTTERY_ROWS: the tile row families each architecture has a kernel for.
const KERNEL_ROWS: Record<Arch, number[]> = { SM90: [4], SM100: [4, 16], SM120: [4] };

const MIN_K = 1024; // _VERIFIER_MIN_K
const MAX_K = 1 << 16; // _VERIFIER_MAX_K
const K_ALIGN = 512; // _K_ALIGNMENT
const MAX_DIM = 1 << 24; // _VERIFIER_MAX_DIM, exclusive
const SCALE_BLOCK = 8; // miner_base.prequant DEFAULT_BLOCK_SIZE
export const MOE_LOTTERY_N = 128; // MOE_LOTTERY_N

/** state.py supports_layer_shape: the shard can carry FP10 planes and the FP8 fallback. */
function supportsLayerShape(n: number, k: number): boolean {
  return k % SCALE_BLOCK === 0 && n % 16 === 0 && k % 16 === 0;
}

/** mining_config.py select_tile, by preference order. */
function selectTile(n: number, k: number, arch: Arch): TileSpec | null {
  if (!(n > 0 && n < MAX_DIM)) return null;
  if (k % K_ALIGN !== 0 || k < MIN_K || k > MAX_K) return null;
  return COMMITTED[arch].find((t) => n % t.cols === 0 && k <= t.maxK) ?? null;
}

/**
 * The tile a dense shard of shape (n, k) mines with, or null when it cannot:
 * state.py can_mine_layer for a dense layer, including the per-architecture
 * kernel check (_lottery_family_available).
 */
export function denseTile(n: number, k: number, arch: Arch): Tile | null {
  if (!supportsLayerShape(n, k)) return null;
  const tile = selectTile(n, k, arch);
  return tile !== null && KERNEL_ROWS[arch].includes(tile.rows) ? tile.name : null;
}

/** can_mine_layer for a stacked expert weight (experts * nE, k): nE must sit on the 128-column lattice. */
export function expertTile(nE: number, k: number, experts: number, arch: Arch): Tile | null {
  if (nE % MOE_LOTTERY_N !== 0) return null;
  if (!supportsLayerShape(experts * nE, k)) return null;
  return denseTile(nE, k, arch);
}
