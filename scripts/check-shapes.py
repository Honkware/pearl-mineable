"""Checks the config reader against transformers' own model definitions.

Builds each listed model on the meta device (no weights, no GPU) and compares
every linear layer's shape with src/arch.ts, after adding together the parts
vLLM fuses. Needs torch and transformers:

    DUMP_LAYERS=layers.json npx vitest run test/layers.dump.test.ts
    python3 scripts/check-shapes.py layers.json [model id ...]

Gemma 4's full-attention qkv differs on purpose: vLLM keeps a value slot that it
fills with the key weights, transformers has none.
"""
import json
import os
import re
import sys
from collections import Counter

import torch
from transformers import AutoConfig, AutoModelForCausalLM

FUSE = [
    (r'compressor\.indexer\.(kv_proj|gate_proj)$', 'idx_comp'),
    (r'compressor\.(kv_proj|gate_proj)$', 'comp'),
    # GLM-5 Flash's linear attention: vLLM fuses q, k, v, b, f_a and g_a into in_proj_qkvbfg_a.
    (r'\.(?:forget_gate\.)?(q_proj|k_proj|v_proj|b_proj|f_a_proj|g_a_proj)$', 'qkv'),
    (r'\.(gate_proj|up_proj)$', 'gate_up'),
    (r'self_attn\.(q_a_proj|kv_a_proj_with_mqa|kv_proj)$', 'qkv_a'),
    (r'\.(in_proj_qkv|in_proj_z)$', 'qkvz'),
    (r'\.(in_proj_b|in_proj_a)$', 'ba'),
]
SKIP = re.compile(r'(\.gate$|router|shared_expert_gate|vision|audio|multi_modal|embed_|\.mtp|eh_proj|enorm|hnorm)')

ours = json.load(open(sys.argv[1]))
only = set(sys.argv[2:])
for entry in ours:
    if only and entry['id'] not in only:
        continue
    # The original file keeps floats such as -5.0 that the JSON dump turns into ints.
    raw = os.path.join(os.path.dirname(__file__), '..', 'src', 'configs', entry['id'].replace('/', '_') + '.json')
    cfg = json.load(open(raw)) if os.path.exists(raw) else dict(entry['config'])
    if cfg.get('model_type') == 'kimi_k2':
        cfg['model_type'] = 'deepseek_v3'
    cfg.pop('auto_map', None)
    model = None
    # JSON turns 10.0 into 10, which transformers' strict configs reject for float fields.
    for _ in range(20):
        try:
            config = AutoConfig.for_model(**cfg)
            text = getattr(config, 'text_config', None) or config
            with torch.device('meta'):
                try:
                    model = AutoModelForCausalLM.from_config(text)
                except ValueError:
                    # Text configs of multimodal models register only a bare model, which has no lm_head.
                    from transformers import AutoModel
                    model = AutoModel.from_config(text)
                    model.lm_head = torch.nn.Linear(text.hidden_size, text.vocab_size, bias=False)
            break
        except Exception as err:
            field = re.search(r"field '(\w+)'", str(err))
            if field and 'float' in str(err):
                f = field.group(1)
                def floatify(d):
                    hit = False
                    if isinstance(d, dict):
                        if isinstance(d.get(f), int) and not isinstance(d.get(f), bool):
                            d[f] = float(d[f]); hit = True
                        for v in d.values():
                            hit = floatify(v) or hit
                    return hit
                if floatify(cfg):
                    continue
            print(f"== {entry['id']}: not built ({type(err).__name__}: {str(err)[:140]})")
            break
    if model is None:
        continue

    hf = Counter()
    groups = {}
    v4 = 'V4' in entry['id']
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or SKIP.search(name) or ('indexer' in name and not v4):
            continue
        key = None
        for pat, tag in FUSE:
            if re.search(pat, name):
                key = (re.sub(pat, '', name), tag)
                break
        if key is None:
            hf[(mod.out_features, mod.in_features)] += 1
        else:
            n, k = groups.get(key, (0, mod.in_features))
            groups[key] = (n + mod.out_features, k)
    for shape in groups.values():
        hf[shape] += 1
    expert_params = Counter()
    for name, p in model.named_parameters():
        if re.search(r'experts\.(gate_up_proj|down_proj|w1|w2|w3)$', name) and p.dim() == 3 and not SKIP.search(name):
            expert_params[(name.split('.')[-1], tuple(p.shape))] += 1

    mine = Counter()
    for l in entry['linears']:
        mine[(l['n'], l['k'])] += l['layers']
    ok = hf == mine
    print(f"== {entry['id']}: {'MATCH' if ok else 'DIFF'}  linears hf={sum(hf.values())} ours={sum(mine.values())}")
    if not ok:
        for s in sorted(set(hf) | set(mine)):
            if hf[s] != mine[s]:
                print(f"   {s}: hf {hf[s]} ours {mine[s]}")
    e = entry['experts']
    if e:
        print(f"   experts ours: count {e['count']} nE {e['nE']} hidden {e['hidden']} inter {e['intermediate']} layers {e['layers']}; hf {dict(expert_params)}")
