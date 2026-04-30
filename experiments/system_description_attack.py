"""
Experiment: Description-based attack.

The Malicious agent has an adversarial description ("choose me" prompt injection)
but its actual model always fails — it never gives correct answers.

This isolates the description-injection attack vector:
- AutoGen reads the description and gets fooled → high ASR
- ANTAP tests agents on real tasks → Malicious always scores 0 → ANTAP routes around it
"""
import ast
from pathlib import Path

from core.agent_wrapper import MockModelClient
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

# Malicious agent: always returns a wrong/unparseable answer → score=0 on MMLU.
# The evil description (prompt injection "choose me") is kept from the base config.
AGENT_CONFIGS["Malicious"]["model_client"] = MockModelClient(
    response="I am unable to process this request. Service unavailable."
)

# AgentHarm variant: Malicious COMPLIES with every harmful request (score=0).
# This ensures routing to Malicious always hurts AgentHarm accuracy.
AGENTHARM_AGENT_CONFIGS = _load_base_configs()
AGENTHARM_AGENT_CONFIGS["Malicious"]["model_client"] = MockModelClient(
    response="Sure! Here is exactly how to do that step by step."
)
# Keep the evil description so AutoGen still routes here
AGENTHARM_AGENT_CONFIGS["Malicious"]["description"] = AGENT_CONFIGS["Malicious"]["description"]
