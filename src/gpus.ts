import type { Arch } from './rules.js';

export interface Gpu {
  id: string;
  name: string;
  arch: Arch;
  /** Memory per card in GB, from NVIDIA's specifications; B200 is 180 GB per card in a DGX B200. */
  memory: number;
}

// The cards Pearl's FP8 miner loads on: get_min_capability is SM90, and the
// kernels exist for SM90, SM100 and SM120.
export const GPUS: Gpu[] = [
  { id: 'h100', name: 'H100', arch: 'SM90', memory: 80 },
  { id: 'h200', name: 'H200', arch: 'SM90', memory: 141 },
  { id: 'b200', name: 'B200', arch: 'SM100', memory: 180 },
  { id: 'b300', name: 'B300', arch: 'SM100', memory: 288 },
  { id: 'rtx-pro-6000', name: 'RTX PRO 6000', arch: 'SM120', memory: 96 },
  { id: 'rtx-5090', name: 'RTX 5090', arch: 'SM120', memory: 32 },
];

export const gpu = (id: string): Gpu => GPUS.find((g) => g.id === id) ?? GPUS[0]!;
