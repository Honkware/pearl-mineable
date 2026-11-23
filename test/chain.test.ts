import { describe, expect, it } from 'vitest';
import { tipFrom } from '../src/chain.js';

describe('earnings per mined MAC', () => {
  it('is the block subsidy times target / 2^256', () => {
    const bits = 0x1c2b3f42;
    const target = 0x2b3f42n << 200n; // mantissa shifted by 8 * (0x1c - 3) bits
    const tip = tipFrom(147112, bits);
    expect(tip.prlPerMac / tip.subsidy).toBeCloseTo(Number(target) / 2 ** 256, 30);
  });

  it('pays the mainnet subsidy for the height', () => {
    expect(tipFrom(145000, 0x1c2b3f42).subsidy).toBeGreaterThan(2000);
    expect(tipFrom(145000, 0x1c2b3f42).subsidy).toBeLessThan(2300);
  });
});
