// Writes every listed model's layers for scripts/check-shapes.py when DUMP_LAYERS names a file.
import { writeFileSync } from 'node:fs';
import { it } from 'vitest';
import { readConfig } from '../src/arch.js';
import { KNOWN } from '../src/known.js';

it.skipIf(!process.env.DUMP_LAYERS)('writes the layer dump', () => {
  const dump = KNOWN.map((k) => ({ id: k.id, config: k.config, ...readConfig(k.config) }));
  writeFileSync(process.env.DUMP_LAYERS!, JSON.stringify(dump));
});
