import ast
from pathlib import Path

from core.system import SystemANTAP, systemAutoGen, systemAntapInAutoGen


def _load_base_configs():
    src_path = Path(__file__).resolve().parents[1] / "core" / "system.py"
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "AGENT_CONFIGS":
                return ast.literal_eval(node.value)
    raise RuntimeError(f"Could not load AGENT_CONFIGS from {src_path}")


AGENT_CONFIGS = _load_base_configs()

# Official API model IDs
AGENT_CONFIGS["Archivist"]["model"] = "moonshotai/kimi-k2.5"
AGENT_CONFIGS["Math_Expert"]["model"] = "moonshotai/kimi-k2.5"
AGENT_CONFIGS["Coding_Expert"]["model"] = "moonshotai/kimi-k2.5"
AGENT_CONFIGS["Strategist"]["model"] = "claude-sonnet-4-5"
AGENT_CONFIGS["Malicious"]["model"] = "claude-sonnet-4-5"
