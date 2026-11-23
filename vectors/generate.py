"""Writes vectors/tiles.json: Pearl's own answer to "which committed tile mines
this (n, k) layer shard on this GPU generation", from the fp8-scheme miner code.

Runs Pearl's functions, not a copy of them: select_tile comes from
mining_config.py, supports_layer_shape from state.py and the per-architecture
kernel families from pearl_gemm's _host.py, each executed from the checkout
with only their imports stubbed.

    python3 vectors/generate.py <pearl checkout> <commit>
"""
from __future__ import annotations

import ast
import enum
import json
import subprocess
import sys
import types


def source(repo: str, commit: str, path: str) -> str:
    return subprocess.run(["git", "-C", repo, "show", f"{commit}:{path}"], capture_output=True, text=True, check=True).stdout


def stub(name: str, **attrs: object) -> None:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod


class Device(enum.Enum):
    HOPPER = 1
    BLACKWELL = 2


class Arch(enum.Enum):
    SM90 = 90
    SM100 = 100
    SM120 = 120


def pick(src: str, names: set[str]) -> str:
    # Only the named top-level functions and assignments, so the rest of the
    # module's torch-bound code is never executed.
    tree = ast.parse(src)
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names) or (isinstance(n, (ast.Assign, ast.AnnAssign)) and any(getattr(t, "id", None) in names for t in (n.targets if isinstance(n, ast.Assign) else [n.target])))]
    return ast.unparse(ast.Module(body=keep, type_ignores=[]))


def main() -> None:
    repo, commit = sys.argv[1], sys.argv[2]
    full = subprocess.run(["git", "-C", repo, "rev-parse", commit], capture_output=True, text=True, check=True).stdout.strip()
    stub("miner_base")
    stub("miner_base.commitment", Device=Device, MiningConfiguration=object, commitment_keys=None)
    # Builders of full job configurations, which the tile choice never calls.
    stub("miner_base.mining_config", __getattr__=lambda name: None)
    prequant = {}
    exec(pick(source(repo, commit, "miner/miner-base/src/miner_base/prequant.py"), {"DEFAULT_BLOCK_SIZE"}), prequant)
    stub("miner_base.prequant", DEFAULT_BLOCK_SIZE=prequant["DEFAULT_BLOCK_SIZE"])
    stub("pearl_gateway")
    stub("pearl_gateway.comm")
    stub("pearl_gateway.comm.dataclasses", MiningJob=object)

    mc_src = source(repo, commit, "miner/vllm-miner/src/vllm_miner/mining_config.py")
    # dataclasses looks its class's module up in sys.modules, so the module has to be registered.
    module = types.ModuleType("mining_config")
    sys.modules["mining_config"] = module
    exec(compile(mc_src, "mining_config.py", "exec"), module.__dict__)
    mc = module.__dict__

    st: dict = {"SCALE_BLOCK": mc["SCALE_BLOCK"]}
    exec(pick(source(repo, commit, "miner/vllm-miner/src/vllm_miner/state.py"), {"supports_layer_shape"}), st)
    host: dict = {"Arch": Arch}
    exec(pick(source(repo, commit, "miner/pearl-gemm/src/pearl_gemm/mixed_gemm/_host.py"), {"_LOTTERY_ROWS"}), host)

    committed = {Arch.SM90: Device.HOPPER, Arch.SM100: Device.BLACKWELL, Arch.SM120: Device.BLACKWELL}
    edges_k = [22016, 22024, 22528, 42496, 42504, 43008, 58368, 58880, 65536, 66048, 1016, 1024, 1032]
    ks = sorted(set(list(range(512, 4097, 8)) + [512 * j for j in range(1, 131)] + edges_k + [5376, 7168, 18432, 28672, 53248, 14336, 21504]))
    real_n = [576, 1536, 2560, 2752, 4096, 5120, 6144, 7168, 8192, 10240, 14336, 16384, 24576, 28672, 32768, 57344, 73728, 128256, 151936, 262144]
    ns = sorted(set(list(range(8, 1025, 8)) + real_n + [(1 << 24) - 64, (1 << 24) - 16, 1 << 24]))

    def verdict(n: int, k: int, arch: Arch) -> str:
        if not st["supports_layer_shape"](n, k):
            return "-"
        tile = mc["select_tile"](n, k, device=committed[arch])
        if tile is None or tile.rows not in host["_LOTTERY_ROWS"][arch]:
            return "-"
        return {(4, 64): "a", (16, 32): "b", (4, 128): "c"}[(tile.rows, tile.cols)]

    out = {
        "source": f"pearl-research-labs/pearl {full}",
        "tiles": {"a": "4x64", "b": "16x32", "c": "4x128", "-": None},
        "n": ns,
        "k": ks,
        "grid": {arch.name: ["".join(verdict(n, k, arch) for k in ks) for n in ns] for arch in Arch},
    }
    json.dump(out, sys.stdout, separators=(",", ":"))


main()
