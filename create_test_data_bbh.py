"""
create_test_data_bbh.py
=======================
Generate per-agent score CSVs for BigBenchHard (lukaemon/bbh).

BBH only has a 'test' split; we 80/20 shuffle-split it ourselves (seed=42):
  train = first 80% of shuffled rows per task
  test  = remaining 20%

Agent → task mapping:
  Math_Expert   : multistep_arithmetic_two, object_counting, penguins_in_a_table
  Coding_Expert : dyck_languages, word_sorting, boolean_expressions
  Archivist     : sports_understanding, movie_recommendation, ruin_names, snarky_synonym
  Strategist    : logical_deduction_five_objects, logical_deduction_seven_objects,
                  logical_deduction_three_objects, causal_judgement, date_understanding,
                  temporal_sequences, tracking_shuffled_objects_three_objects,
                  tracking_shuffled_objects_five_objects, tracking_shuffled_objects_seven_objects,
                  web_of_lies, navigate, geometric_shapes, formal_fallacies, hyperbaton,
                  disambiguation_qa
  Malicious     : (no tasks — uses same prompts as Strategist for consistent coverage)

Output CSV columns (matching MMLU per-agent format):
  prompt_id, prompt, has_trigger, subject, target,
  {AgentName}_response, {AgentName}_score, {AgentName}_prediction, malicious_calls
"""

import argparse
import asyncio
import importlib
import os
import time

import httpx
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# ── Agent → BBH task mapping ──────────────────────────────────────────────────
AGENT_TASKS = {
    "Math_Expert": [
        "multistep_arithmetic_two",
        "object_counting",
        "penguins_in_a_table",
    ],
    "Coding_Expert": [
        "dyck_languages",
        "word_sorting",
        "boolean_expressions",
    ],
    "Archivist": [
        "sports_understanding",
        "movie_recommendation",
        "ruin_names",
        "snarky_synonym",
    ],
    "Strategist": [
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
        "logical_deduction_three_objects",
        "causal_judgement",
        "date_understanding",
        "temporal_sequences",
        "tracking_shuffled_objects_three_objects",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
        "web_of_lies",
        "navigate",
        "geometric_shapes",
        "formal_fallacies",
        "hyperbaton",
        "disambiguation_qa",
    ],
    # Malicious agent: piggyback on Strategist tasks so the router sees the same prompts
    "Malicious": [
        "logical_deduction_five_objects",
        "logical_deduction_three_objects",
        "causal_judgement",
        "date_understanding",
        "web_of_lies",
    ],
}

SPLIT_SEED = 42
TRAIN_FRAC = 0.80

LOCAL_MODELS = [
    "llama3.2", "llama3", "llama2", "qwen2.5", "qwen2",
    "codellama", "mistral", "phi", "deepseek", "t1c/",
]


def is_local_model(model_name):
    lower = model_name.lower()
    return any(lower.startswith(m) or m in lower for m in LOCAL_MODELS)


def is_claude_model(model_name):
    lower = model_name.lower()
    return lower.startswith("claude") or "claude" in lower


# ── Ollama async runner (same pattern as create_test_data_agentharm.py) ───────
def run_ollama_direct(model_name, system_message, prompt_rows, timeout, max_workers=2):
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")

    async def _one(client, prompt_id, formatted_prompt):
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": formatted_prompt})
        try:
            resp = await client.post(
                "/chat/completions",
                json={"model": model_name, "messages": messages, "stream": False,
                      "options": {"num_ctx": 4096}},
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return prompt_id, data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return prompt_id, f"Error: {e}"

    async def _run_all():
        sem = asyncio.Semaphore(max_workers)
        async with httpx.AsyncClient(base_url=base_url) as client:
            async def _bounded(pid, prompt):
                async with sem:
                    return await _one(client, pid, prompt)
            tasks = [_bounded(pid, p) for pid, p in prompt_rows]
            results = {}
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=model_name):
                pid, resp = await coro
                results[pid] = resp
            return results

    return asyncio.run(_run_all())


def score_response(response: str, target: str) -> int:
    """Return 1 if target string appears in response (case-insensitive), else 0."""
    if not isinstance(response, str):
        return 0
    return 1 if target.strip().lower() in response.strip().lower() else 0


def load_bbh_rows(tasks: list[str], train: bool) -> list[dict]:
    """Load and split BBH rows for the given tasks."""
    all_rows = []
    for task in tasks:
        try:
            ds = load_dataset("lukaemon/bbh", task, split="test")
        except Exception as e:
            print(f"[WARN] Could not load BBH task '{task}': {e}")
            continue

        # Shuffle and split 80/20
        ds = ds.shuffle(seed=SPLIT_SEED)
        n = len(ds)
        split_idx = int(n * TRAIN_FRAC)
        if train:
            ds = ds.select(range(split_idx))
        else:
            ds = ds.select(range(split_idx, n))

        for idx, example in enumerate(ds):
            all_rows.append({
                "task": task,
                "input": example["input"],
                "target": example["target"],
            })

    return all_rows


def main():
    parser = argparse.ArgumentParser(description="Generate BBH per-agent score CSVs.")
    parser.add_argument("--system", type=str, default="experiments.system_base_official")
    parser.add_argument("--agent", type=str, required=True)
    parser.add_argument("--output", type=str, default="bbh_scores.csv")
    parser.add_argument("--train", action="store_true", help="Use train split (80%)")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.agent not in AGENT_TASKS:
        raise ValueError(f"Unknown agent '{args.agent}'. Valid: {list(AGENT_TASKS)}")

    # Load system module
    system_module = importlib.import_module(args.system)
    # Prefer AGENTHARM_AGENT_CONFIGS if it exists (some system modules define it)
    agent_configs = getattr(system_module, "AGENTHARM_AGENT_CONFIGS",
                            getattr(system_module, "AGENT_CONFIGS", {}))
    if args.agent not in agent_configs:
        raise ValueError(f"Agent '{args.agent}' not found in system module '{args.system}'")

    config = agent_configs[args.agent]
    system_message = config.get("system_message", "")
    model_name = config.get("model", "")

    # Load BBH rows for this agent
    tasks = AGENT_TASKS[args.agent]
    split_label = "train" if args.train else "test"
    print(f"Loading BBH tasks for {args.agent} ({split_label}): {tasks}")
    rows_data = load_bbh_rows(tasks, train=args.train)

    if args.limit:
        rows_data = rows_data[: args.limit]

    print(f"Total rows for {args.agent} ({split_label}): {len(rows_data)}")

    # Build prompt_rows for inference
    prompt_rows = []
    prompt_meta = {}  # prompt_id -> {prompt, target, subject}
    for idx, row in enumerate(rows_data):
        pid = f"bbh_{args.agent[:3].lower()}_{idx}"
        prompt_text = row["input"]
        prompt_rows.append((pid, prompt_text))
        prompt_meta[pid] = {
            "prompt": prompt_text,
            "target": row["target"],
            "subject": row["task"],
        }

    print(f"Running {args.agent} (model={model_name}) on {len(prompt_rows)} BBH prompts...")

    # Handle mock clients (Malicious agent in some configs)
    mock_client = config.get("model_client", None)
    if mock_client is not None and hasattr(mock_client, "_response"):
        mock_response = mock_client._response
        outputs = {pid: mock_response for pid, _ in prompt_rows}
    elif is_local_model(model_name):
        outputs = run_ollama_direct(
            model_name=model_name,
            system_message=system_message,
            prompt_rows=prompt_rows,
            timeout=args.timeout,
            max_workers=args.workers,
        )
    else:
        # Fallback: try Ollama anyway
        print(f"[WARN] Model '{model_name}' not recognized as local; trying Ollama anyway.")
        outputs = run_ollama_direct(
            model_name=model_name,
            system_message=system_message,
            prompt_rows=prompt_rows,
            timeout=args.timeout,
            max_workers=args.workers,
        )

    # Build output rows
    out_rows = []
    for pid, _ in prompt_rows:
        meta = prompt_meta[pid]
        response = outputs.get(pid, "Error: Missing output")
        sc = score_response(response, meta["target"])
        # Malicious agent: score=1 means it called the malicious tool (not applicable
        # for BBH — we track 0 malicious calls since BBH is not a harm benchmark).
        out_rows.append({
            "prompt_id": pid,
            "prompt": meta["prompt"],
            "has_trigger": False,
            "subject": meta["subject"],
            "target": meta["target"],
            f"{args.agent}_response": response,
            f"{args.agent}_score": sc,
            f"{args.agent}_prediction": response[:80].replace("\n", " ") if isinstance(response, str) else "",
            "malicious_calls": 0,
        })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pd.DataFrame(out_rows).to_csv(args.output, index=False)
    print(f"Saved: {args.output}  ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
