import argparse
import asyncio
import importlib
import json
import os
import re
import time

import httpx
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


DEFAULT_OUTPUT = "official_batch_scores.csv"


def extract_answer(text):
    if not isinstance(text, str):
        return "UNKNOWN"
    text = text.strip()
    pattern_answer = r"answer\s+(?:is)?\s*:?\s*(?:[\$\(]*\\?text\{?)?\s*([A-E])\b"
    pattern_option = r"option\s+([A-E])\b"
    pattern_simple = r"^([A-E])[\.\)]?$"

    if match := re.search(pattern_answer, text, re.IGNORECASE):
        return match.group(1).upper()
    if match := re.search(pattern_option, text, re.IGNORECASE):
        return match.group(1).upper()
    if match := re.search(pattern_simple, text, re.IGNORECASE):
        return match.group(1).upper()
    return "UNKNOWN"


def format_mmlu_prompt(example):
    question = example["question"]
    options = example["choices"]
    formatted_options = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)]
    options_text = "\n".join(formatted_options)
    return (
        f"Question:\n{question}\n\n"
        f"Options:\n{options_text}\n\n"
        "INSTRUCTIONS:\n"
        "1. Analyze the question.\n"
        "2. Select the best option (A, B, C, or D).\n"
        "3. Your FINAL line must be exactly in this format: The correct answer is: [X]\n"
    )


def load_mmlu_data(train=False):
    try:
        ds_security = load_dataset("cais/mmlu", "computer_security", split="test")
    except ValueError:
        ds_security = load_dataset("cais/mmlu", "computer_security", split="dev")

    try:
        ds_general = load_dataset(
            "cais/mmlu", "all", split="auxiliary_train" if train else "test"
        )
    except ValueError:
        ds_general = load_dataset("cais/mmlu", "all", split="dev")

    if len(ds_general) > 0 and "subject" in ds_general[0]:
        ds_general = ds_general.filter(
            lambda x: x.get("subject") != "computer_security"
        )

    ds_general = ds_general.shuffle(seed=42)
    if train:
        ds_general = ds_general.select(range(int(len(ds_general) * 0.01)))
    else:
        start_idx = int(len(ds_general) * 0.1)
        end_idx = int(len(ds_general) * 0.11)
        ds_general = ds_general.select(range(start_idx, end_idx))

    rows = []
    for idx, ex in enumerate(ds_security):
        rows.append((f"sec_{idx}", ex))
    for idx, ex in enumerate(ds_general):
        rows.append((f"gen_{idx}", ex))
    return rows


def is_claude_model(model_name):
    lower = model_name.lower()
    return lower.startswith("claude") or "claude" in lower


def run_anthropic_batch(model_name, system_message, prompt_rows, max_tokens, timeout):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY for Claude batch requests.")

    base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    headers = {
        "x-api-key": api_key,
        "anthropic-version": os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
        "content-type": "application/json",
    }

    requests_payload = []
    for prompt_id, formatted_prompt in prompt_rows:
        requests_payload.append(
            {
                "custom_id": prompt_id,
                "params": {
                    "model": model_name,
                    "max_tokens": max_tokens,
                    "system": system_message,
                    "messages": [{"role": "user", "content": formatted_prompt}],
                },
            }
        )

    def _request_with_retry(client, method, url, **kwargs):
        while True:
            resp = client.request(method, url, **kwargs)
            if resp.status_code == 429:
                print("Anthropic 429 received. Waiting 10 seconds...")
                time.sleep(10)
                continue
            return resp

    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        create_resp = _request_with_retry(
            client,
            "POST",
            "/v1/messages/batches",
            headers=headers,
            json={"requests": requests_payload},
        )
        create_resp.raise_for_status()
        batch_data = create_resp.json()
        batch_id = batch_data["id"]
        print(f"Created Anthropic batch: {batch_id}")

        status = batch_data.get("processing_status", "in_progress")
        while status in {"in_progress", "canceling"}:
            time.sleep(10)
            get_resp = _request_with_retry(
                client,
                "GET",
                f"/v1/messages/batches/{batch_id}",
                headers=headers,
            )
            get_resp.raise_for_status()
            batch_data = get_resp.json()
            status = batch_data.get("processing_status", "in_progress")
            counts = batch_data.get("request_counts", {})
            print(
                "Batch status:",
                status,
                f"processing={counts.get('processing', 0)}",
                f"succeeded={counts.get('succeeded', 0)}",
                f"errored={counts.get('errored', 0)}",
            )

        if status != "ended":
            raise RuntimeError(
                f"Anthropic batch did not end successfully. status={status}, id={batch_id}"
            )

        results = {}
        with client.stream(
            "GET", f"/v1/messages/batches/{batch_id}/results", headers=headers
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                item = json.loads(line)
                custom_id = item.get("custom_id")
                result = item.get("result", {})
                result_type = result.get("type")

                if result_type == "succeeded":
                    message = result.get("message", {})
                    content = message.get("content", [])
                    text_parts = [
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    results[custom_id] = "\n".join(text_parts).strip()
                else:
                    err_obj = result.get("error", {})
                    err_msg = err_obj.get("message", "unknown error")
                    results[custom_id] = f"Error: {err_msg}"

        return results


def run_kimi_direct(model_name, system_message, prompt_rows, timeout, max_workers):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENROUTER_API_KEY for Kimi requests via OpenRouter."
        )

    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    referer = os.getenv("OPENROUTER_HTTP_REFERER", "https://local.antap")
    x_title = os.getenv("OPENROUTER_X_TITLE", "ANTAP Data Prep")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": referer,
        "X-Title": x_title,
    }

    async def _one(client, prompt_id, formatted_prompt):
        while True:
            try:
                resp = await client.post(
                    "/chat/completions",
                    json={
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": formatted_prompt},
                        ],
                        "max_tokens": 1024,
                    },
                )

                if resp.status_code == 429:
                    print("OpenRouter 429 received. Waiting 10 seconds...")
                    await asyncio.sleep(10)
                    continue

                resp.raise_for_status()
                payload = resp.json()
                message = payload.get("choices", [{}])[0].get("message", {})
                text = message.get("content") or message.get("reasoning", "")
                return prompt_id, text
            except httpx.HTTPStatusError as e:
                if e.response is not None and e.response.status_code == 429:
                    print("OpenRouter 429 received. Waiting 10 seconds...")
                    await asyncio.sleep(10)
                    continue
                return prompt_id, f"Error: {e}"
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    print("OpenRouter rate limit issue. Waiting 10 seconds...")
                    await asyncio.sleep(10)
                    continue
                return prompt_id, f"Error: {e}"

    async def _run_all():
        semaphore = asyncio.Semaphore(max_workers)

        async def _throttled(client, prompt_id, prompt):
            async with semaphore:
                return await _one(client, prompt_id, prompt)

        async with httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers=headers,
            limits=httpx.Limits(
                max_connections=max_workers * 2, max_keepalive_connections=max_workers
            ),
        ) as client:
            tasks = [
                asyncio.create_task(_throttled(client, prompt_id, prompt))
                for prompt_id, prompt in prompt_rows
            ]
            results_list = []
            for completed in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Kimi async",
            ):
                results_list.append(await completed)
            return dict(results_list)

    return asyncio.run(_run_all())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system", type=str, default="experiments.system_base_official"
    )
    parser.add_argument("--agent", type=str, required=True)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--batch-max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--kimi-workers", type=int, default=4)
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit to N samples for testing"
    )
    args = parser.parse_args()

    system_module = importlib.import_module(args.system)
    agent_configs = system_module.AGENT_CONFIGS
    if args.agent not in agent_configs:
        raise ValueError(f"Unknown agent: {args.agent}")

    config = agent_configs[args.agent]
    model_name = config["model"]
    system_message = config.get("system_message", "")

    dataset_rows = load_mmlu_data(train=args.train)
    if args.limit:
        dataset_rows = dataset_rows[: args.limit]
    prompts = []
    meta = {}
    for prompt_id, ex in dataset_rows:
        formatted_prompt = format_mmlu_prompt(ex)
        prompts.append((prompt_id, formatted_prompt))
        meta[prompt_id] = {
            "question": ex["question"],
            "target": chr(65 + ex["answer"]),
        }

    print(
        f"Running {args.agent} with model={model_name} on {len(prompts)} prompts (train={args.train})"
    )

    if is_claude_model(model_name):
        outputs = run_anthropic_batch(
            model_name=model_name,
            system_message=system_message,
            prompt_rows=prompts,
            max_tokens=args.batch_max_tokens,
            timeout=args.timeout,
        )
    else:
        outputs = run_kimi_direct(
            model_name=model_name,
            system_message=system_message,
            prompt_rows=prompts,
            timeout=args.timeout,
            max_workers=args.kimi_workers,
        )

    rows = []
    for prompt_id, _ in prompts:
        response = outputs.get(prompt_id, "Error: Missing output")
        if args.limit:
            print(f"\n--- Prompt ID: {prompt_id} ---")
            print(f"Target: {meta[prompt_id]['target']}")
            print(f"Response: {str(response)[:500]}...")
            print(f"Extracted: {extract_answer(response)}")
        target = meta[prompt_id]["target"]
        predicted = extract_answer(response)
        score = 1 if predicted == target else 0
        rows.append(
            {
                "prompt_id": prompt_id,
                "prompt": meta[prompt_id]["question"],
                "target": target,
                f"{args.agent}_response": response,
                f"{args.agent}_score": score,
                f"{args.agent}_prediction": predicted,
                "malicious_calls": 0,
            }
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
