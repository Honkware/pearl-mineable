# pearl-mineable

How much of an AI model's prompt work can mine Pearl today and after the FP8 fork, on which NVIDIA cards, and what it earns per million prompt tokens. It reads a model's `config.json`, so it needs no GPU and no weights, and it runs in the browser with chain data from a public Pearl RPC endpoint through [pearljs](https://github.com/Honkware/pearljs).

The site is at https://honkware.github.io/pearl-mineable/. Search any Hugging Face model by its ID, or pick from the models Pearl publishes, tests its FP8 miner on or serves, and the most used open models.

## Today

Mainnet runs the INT7 miner on the `master` branch of [pearl-research-labs/pearl](https://github.com/pearl-research-labs/pearl). Only Pearl's own INT7 checkpoints mine, and within them only the layers stored as 7-bit integers. A layer mines when its shard is at least 256 × 1,024 and the prompt batch has at least 1,024 tokens (`should_use_noisy_gemm` in `config.py`). `vectors/now.py` reads those thresholds from Pearl's `config.yaml`, and the layer formats come from each checkpoint's own quantization config, matched the way vLLM's compressed-tensors loader matches them, which Pearl's config extends: a layer's own name or pattern before a class such as `Linear`.

## After FP8

The rules come from the `fp8-scheme` branch, and `src/rules.ts` names the function each check is taken from. A dense layer shard mines when Pearl's `select_tile` finds a committed tile for its shape and the card has a kernel for that tile. Routed experts mine their gate and up projection only, and only in a BF16 checkpoint without SwiGLU limits, expert biases or expert parallelism. MLA's `kv_b_proj`, the LM head and the experts' down projection never mine.

`vectors/generate.py` runs Pearl's own `select_tile`, `supports_layer_shape` and kernel table from a checkout over 86,000 layer shapes for each GPU generation, and `test/rules.test.ts` holds `src/rules.ts` to every answer. The model tests check config reading against real layer sizes, including the GLM-5.2 shapes Pearl's tests use. `scripts/check-shapes.py` builds each listed model from transformers' own definition, without weights, and compares every linear layer: 18 of the 20 match exactly. Gemma 4 differs where vLLM keeps a value slot that transformers drops, and DeepSeek V4.1 is not in transformers yet.

## Earnings

A win has probability target × m × n × k / 2^256 (`effective_work_per_matmul` on the FP8 branch), so each mined multiply-accumulate earns the block reward times target / 2^256. The reward is the subsidy plus the average fees of the last six blocks, read from their coinbases. The INT7 miner credits the same way, with k rounded down to its noise rank of 128 (`inner_hash_per_matmul` in `noisy_gemm.py`). Before the fork the post-fork figures use today's difficulty. The page reads the tip block's certificate version and switches to the FP8 figures when mainnet starts carrying V4 certificates.

## Limits

- Verdicts follow Pearl's code and have not been checked against a miner running on a GPU.
- Only prompt processing mines. Generation, small batches and gaps between jobs do not, so real earnings depend on traffic, and the page gives no per-hour figure.
- Work counts linear layers only, so attention scores are left out. Projections that vLLM builds without a quant config never mine; DeepSeek V4's compressors are counted as such, while the smaller indexer gates, the DeltaNet convolution and hyper-connection mixing are left out.
- Each architecture's layers are read from its vLLM model code, and the config reader refuses architectures it has no reader for.
- Auto picks the fewest GPUs, in powers of two, that fit Pearl's example command (`--max-model-len 8192 --gpu-memory-utilization 0.9 --enforce-eager`): 90% of each card must hold the weights, the miner's planes for every mined layer (about 3.3 bytes per weight, 2.3 for experts, from `state.py`), the miner's runtime reservation as `memory.py` computes it, one 8,192-token sequence of cache, and 2 GB for vLLM's activations, which is the one estimate. Longer contexts need more.

## Running it

```bash
npm install
npm run dev
npm test
```

The `rules` workflow regenerates the vectors from Pearl's branches every Monday and opens a pull request when a rule changed. To do the same by hand from a checkout:

```bash
python3 vectors/generate.py ~/Pearl/pearl origin/fp8-scheme > vectors/tiles.json
python3 vectors/now.py ~/Pearl/pearl origin/master > vectors/now.json
npm test
```

To check the config reader against transformers, with torch and transformers installed:

```bash
DUMP_LAYERS=layers.json npx vitest run test/layers.dump.test.ts
python3 scripts/check-shapes.py layers.json
```

## Licence

MIT. The model configs in `src/configs` are the `config.json` files their publishers put on Hugging Face; the Llama 4 ones come from unsloth's copies, since Meta's repositories need a login.
