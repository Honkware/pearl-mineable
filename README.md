# pearl-models

How much of an AI model's prompt work can mine Pearl under the FP8 rules, on which NVIDIA cards, and what it earns per million prompt tokens at the current difficulty. It reads a model's `config.json`, so it needs no GPU and no weights, and it runs in the browser with chain data from a public Pearl RPC endpoint through [pearljs](https://github.com/Honkware/pearljs).

## How it decides

The rules come from the `fp8-scheme` branch of [pearl-research-labs/pearl](https://github.com/pearl-research-labs/pearl), and `src/rules.ts` names the function each check is taken from. A dense layer shard mines when Pearl's `select_tile` finds a committed tile for its shape and the card has a kernel for that tile. Routed experts mine their gate and up projection only, and only in a BF16 checkpoint without SwiGLU limits, expert biases or expert parallelism. MLA's `kv_b_proj`, the LM head and the experts' down projection never mine.

`vectors/generate.py` runs Pearl's own `select_tile`, `supports_layer_shape` and kernel table from a checkout over 86,000 layer shapes for each GPU generation, and `test/rules.test.ts` holds `src/rules.ts` to every answer. The model tests check config reading against real layer sizes, including the GLM-5.2 shapes Pearl's tests use.

A win has probability target × m × n × k / 2^256 (`effective_work_per_matmul`), so each mined multiply-accumulate earns the block subsidy times target / 2^256. The page reads both from the chain tip, so PRL per million prompt tokens needs no hashrate estimate. Mainnet has not forked to FP8 yet, and its difficulty will move once miners switch, so the figure is at today's difficulty.

## Limits

- Verdicts follow Pearl's code and have not been checked against the miner running on a GPU.
- Only prompt processing mines. Generation, small batches and gaps between jobs do not, so real earnings depend on traffic, and the page gives no per-hour figure.
- Work counts linear layers only. Attention scores and, for GLM-5, the sparse-attention indexer are left out.
- Tensor parallelism is picked as the smallest split of 1, 2, 4 or 8 that holds the weights with a quarter of memory spare.

## Running it

```bash
npm install
npm run dev
npm test
```

To follow a change on Pearl's branch, regenerate the vectors from a checkout and run the tests:

```bash
python3 vectors/generate.py ~/Pearl/pearl origin/fp8-scheme > vectors/tiles.json
npm test
```
