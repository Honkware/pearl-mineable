// The models Pearl publishes or builds its FP8 miner around, then the most
// used open models. Configs are kept here so the list needs no network.
import glm52 from './configs/zai-org_GLM-5.2.json' with { type: 'json' };
import llama8b from './configs/pearl-ai_Llama-3.1-8B-Instruct-pearl.json' with { type: 'json' };
import llama70b from './configs/pearl-ai_Llama-3.3-70B-Instruct-pearl.json' with { type: 'json' };
import gemma31b from './configs/pearl-ai_Gemma-4-31B-it-pearl.json' with { type: 'json' };
import qwen30b from './configs/pearl-ai_Qwen3-30B-A3B-Instruct-2507-pearl.json' with { type: 'json' };
import qwen06b from './configs/Qwen_Qwen3-0.6B.json' with { type: 'json' };
import qwen32b from './configs/Qwen_Qwen3-32B.json' with { type: 'json' };
import deepseekV3 from './configs/deepseek-ai_DeepSeek-V3.json' with { type: 'json' };
import gptOss from './configs/openai_gpt-oss-120b.json' with { type: 'json' };
import mistral from './configs/mistralai_Mistral-Small-3.2-24B-Instruct-2506.json' with { type: 'json' };

export interface Known {
  id: string;
  name: string;
  org: string;
  /** Pearl publishes it, or its FP8 miner is tested on it. */
  pearl: boolean;
  config: Record<string, unknown>;
}

// pearl-ai's copies carry Pearl's pre-fork quantization; the FP8 miner loads
// the original BF16 weights, which have the same shapes.
const original = (c: Record<string, unknown>): Record<string, unknown> => {
  const { quantization_config: _, ...rest } = c;
  return rest;
};

export const KNOWN: Known[] = [
  { id: 'meta-llama/Llama-3.3-70B-Instruct', name: 'Llama 3.3 70B Instruct', org: 'meta-llama', pearl: true, config: original(llama70b) },
  { id: 'meta-llama/Llama-3.1-8B-Instruct', name: 'Llama 3.1 8B Instruct', org: 'meta-llama', pearl: true, config: original(llama8b) },
  { id: 'zai-org/GLM-5.2', name: 'GLM-5.2', org: 'zai-org', pearl: true, config: glm52 },
  { id: 'google/gemma-4-31B-it', name: 'Gemma 4 31B', org: 'google', pearl: true, config: original(gemma31b) },
  { id: 'Qwen/Qwen3-30B-A3B-Instruct-2507', name: 'Qwen3 30B A3B', org: 'Qwen', pearl: true, config: original(qwen30b) },
  { id: 'Qwen/Qwen3-0.6B', name: 'Qwen3 0.6B', org: 'Qwen', pearl: true, config: qwen06b },
  { id: 'Qwen/Qwen3-32B', name: 'Qwen3 32B', org: 'Qwen', pearl: false, config: qwen32b },
  { id: 'mistralai/Mistral-Small-3.2-24B-Instruct-2506', name: 'Mistral Small 3.2 24B', org: 'mistralai', pearl: false, config: mistral },
  { id: 'deepseek-ai/DeepSeek-V3', name: 'DeepSeek V3', org: 'deepseek-ai', pearl: false, config: deepseekV3 },
  { id: 'openai/gpt-oss-120b', name: 'gpt-oss 120B', org: 'openai', pearl: false, config: gptOss },
];
