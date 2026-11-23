// A win has probability target * m*n*k / 2^256 (mining_config.py
// effective_work_per_matmul), so the tip's bits and subsidy price mined work
// without a hashrate estimate.
import { parseBlock } from 'pearljs/block';
import { blockSubsidy, difficultyForBits } from 'pearljs/consensus';
import { jackpotTarget } from 'pearljs/mining';
import { networks } from 'pearljs/networks';
import { PearldClient } from 'pearljs/rpc';

export interface Tip {
  height: number;
  bits: number;
  prlPerMac: number;
  subsidy: number;
  difficulty: number;
  fp8: boolean;
}

export const PUBLIC_RPC = 'https://pearlchain.live/rpc';

export function tipFrom(height: number, bits: number, fp8 = false): Tip {
  const subsidy = Number(blockSubsidy(networks.mainnet, height)) / 1e8;
  return { height, bits, subsidy, difficulty: difficultyForBits(bits, networks.mainnet), prlPerMac: (Number(jackpotTarget(bits)) / 2 ** 256) * subsidy, fp8 };
}

export async function fetchTip(url = PUBLIC_RPC): Promise<Tip> {
  const rpc = new PearldClient({ url, retry: { retries: 3, delay: 1000 }, maxConcurrentRequests: 1 });
  const hash = await rpc.getBestBlockHash();
  const [header, raw] = [await rpc.getBlockHeader(hash), await rpc.getRawBlock(hash)];
  return tipFrom(header.height, Number.parseInt(String(header.bits), 16), certificateVersion(raw) === 4);
}

// FP8 blocks carry a V4 certificate, which only parses in the fork's wire format.
function certificateVersion(raw: Uint8Array): number | null {
  for (const fp8 of [false, true]) {
    try {
      return parseBlock(raw, { fp8 }).certificate?.version ?? null;
    } catch {}
  }
  return null;
}
