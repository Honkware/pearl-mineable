// What a mined multiply-accumulate earns, from the chain tip. Pearl credits a
// shard m*n*k of work and a win has probability target * m*n*k / 2^256
// (mining_config.py effective_work_per_matmul), so one mined MAC earns the
// block subsidy times target / 2^256, with no hashrate figure involved.
import { blockSubsidy, difficultyForBits } from 'pearljs/consensus';
import { jackpotTarget } from 'pearljs/mining';
import { networks } from 'pearljs/networks';
import { PearldClient } from 'pearljs/rpc';

export interface Tip {
  height: number;
  bits: number;
  /** PRL one mined multiply-accumulate earns on average. */
  prlPerMac: number;
  subsidy: number;
  difficulty: number;
}

export const PUBLIC_RPC = 'https://pearlchain.live/rpc';

export function tipFrom(height: number, bits: number): Tip {
  const subsidy = Number(blockSubsidy(networks.mainnet, height)) / 1e8;
  return { height, bits, subsidy, difficulty: difficultyForBits(bits, networks.mainnet), prlPerMac: (Number(jackpotTarget(bits)) / 2 ** 256) * subsidy };
}

export async function fetchTip(url = PUBLIC_RPC): Promise<Tip> {
  const rpc = new PearldClient({ url, retry: { retries: 3, delay: 1000 }, maxConcurrentRequests: 1 });
  const header = await rpc.getBlockHeader(await rpc.getBestBlockHash());
  return tipFrom(header.height, Number.parseInt(String(header.bits), 16));
}
