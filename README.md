# pearl-models

How much of an AI model's prompt work can mine Pearl today and after the FP8 fork, on which NVIDIA cards, and what it earns per million prompt tokens. It reads a model's `config.json`, so it needs no GPU and no weights, and it runs in the browser with chain data from a public Pearl RPC endpoint through [pearljs](https://github.com/Honkware/pearljs).

## Today

Mainnet runs the INT7 miner on the `master` branch of [pearl-research-labs/pearl](https://github.com/pearl-research-labs/pearl). Only Pearl's own INT7 checkpoints mine, and within them only the layers stored as 7-bit integers. A layer mines when its shard is at least 256 × 1,024 and the prompt batch has at least 1,024 tokens (`should_use_noisy_gemm` in `config.py`). `vectors/now.py` reads those thresholds from Pearl's `config.yaml`, and the layer formats come from each checkpoint's own quantization config.

## After FP8

The rules come from the `fp8-scheme` branch, and `src/rules.ts` names the function each check is taken from. A dense layer shard mines when Pearl's `select_tile` finds a committed tile for its shape and the card has a kernel for that tile. Routed experts mine their gate and up projection only, and only in a BF16 checkpoint without SwiGLU limits, expert biases or expert parallelism. MLA's `kv_b_proj`, the LM head and the experts' down projection never mine.

`vectors/generate.py` runs Pearl's own `select_tile`, `supports_layer_shape` and kernel table from a checkout over 86,000 layer shapes for each GPU generation, and `test/rules.test.ts` holds `src/rules.ts` to every answer. The model tests check config reading against real layer sizes, including the GLM-5.2 shapes Pearl's tests use.

## Earnings

A win has probability target × m × n × k / 2^256 (`effective_work_per_matmul` on the FP8 branch), so each mined multiply-accumulate earns the block subsidy times target / 2^256, read from the chain tip. Before the fork the page prices today's mineable work at today's difficulty, and it assumes the INT7 miner's tiles cover every output, which the fork's code states and the INT7 code does not. The page reads the tip block's certificate version and switches to the FP8 figures when mainnet starts carrying V4 certificates.

## Limits

- Verdicts follow Pearl's code and have not been checked against a miner running on a GPU.
- Only prompt processing mines. Generation, small batches and gaps between jobs do not, so real earnings depend on traffic, and the page gives no per-hour figure.
- Work counts linear layers only. Attention scores and, for GLM-5, the sparse-attention indexer are left out.
- Tensor parallelism is picked as the smallest split of 1, 2, 4 or 8 that holds the BF16 weights with a quarter of memory spare, and today's figures use the same split.

## Running it

```bash
npm install
npm run dev
npm test
```

To follow a change on Pearl's branches, regenerate the vectors from a checkout and run the tests:

```bash
python3 vectors/generate.py ~/Pearl/pearl origin/fp8-scheme > vectors/tiles.json
python3 vectors/now.py ~/Pearl/pearl origin/master > vectors/now.json
npm test
```
