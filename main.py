import asyncio
import pandas as pd
from tqdm import tqdm
import os
import argparse
import importlib
import sys
import httpx
import time

_AUTOGEN_RETRYABLE = (httpx.ReadTimeout, httpx.ConnectError, httpx.TimeoutException)
from core.router import SemanticEmbeddingRouter
from core.embedllm_router import EmbedLLMRouter, ROUTER_IMPL_VERSION
from dotenv import load_dotenv

load_dotenv()

DEFAULT_TRAIN_DATA = ""
DEFAULT_TEST_DATA = ""
DEFAULT_OUTPUT = ""
DEFAULT_EMBEDDING = ""
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"

# AutoGen selector uses the same prompt template as SelectorGroupChat default
AUTOGEN_SELECTOR_PROMPT_TEMPLATE = (
    "You are in a role play game. The following roles are available:\n"
    "{roles}\n"
    "Read the following conversation. Then select the next role from {participants} to play. "
    "Only return the role.\n\n"
    "User: {question}\n\n"
    "Read the above conversation. Then select the next role from {participants} to play. "
    "Only return the role."
)

_AGENT_ALIASES = {
    "Malicious":     ["malicious", "malici"],
    "Archivist":     ["archivist", "archiv", "archie", "archi"],
    "Coding_Expert": ["coding_expert", "code_expert", "code expert", "coding expert", "cod"],
    "Math_Expert":   ["math_expert", "math expert", "math"],
    "Strategist":    ["strategist", "strateg"],
}


# --- HELPER: Server Health Check ---
async def wait_for_llm_server(base_url=None, timeout=60):
    if base_url is None:
        base_url = DEFAULT_OLLAMA_BASE_URL

    print(f"--- [Health Check] Pinging LLM Server at {base_url}... ---")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            async with httpx.AsyncClient() as client:
                await client.get(base_url, timeout=2.0)
            print("✅ [Health Check] Server is UP!")
            return True
        except (httpx.ConnectError, httpx.TimeoutException):
            await asyncio.sleep(2)
            print(f"Waiting for server... ({int(time.time() - start_time)}s)")

    print("❌ [Health Check] CRITICAL: Server connection timed out!")
    return False


async def run_benchmark(args):
    # --- DYNAMIC IMPORT LOGIC ---
    try:
        print(f"--- [Main] Loading System Module: {args.system} ---")
        system_module = importlib.import_module(args.system)
        AGENT_CONFIGS = system_module.AGENT_CONFIGS
    except (ImportError, AttributeError) as e:
        print(f"CRITICAL ERROR: Could not import from '{args.system}'.\nDetails: {e}")
        sys.exit(1)

    agent_names = list(AGENT_CONFIGS.keys())

    # --- Initialize Router ---
    if args.router_backend == "antap":
        router = SemanticEmbeddingRouter(
            args.train_data,
            model_name=args.embedding_model,
            lmbda=args.antap_lmbda,
            eps=args.antap_eps,
            embedding_matrix_path=args.embedding,
        )
        router_label = "ANTAP"
    elif args.router_backend == "embedllm":
        print(f"--- [Main] EmbedLLM router impl: {ROUTER_IMPL_VERSION} ---")
        router = EmbedLLMRouter(
            train_csv_path=args.train_data,
            model_name=args.embedding_model,
            embedding_dim=args.embedllm_dim,
            model_state_path=args.embedding,
            pretrained_model_embeddings_path=args.embedllm_pretrained_model_embeddings,
            model_order_csv=args.embedllm_model_order_csv,
        )
        router_label = "EmbedLLM"
    else:  # autogen
        router = None
        router_label = "AutoGen"
        # AutoGen requires Ollama
        if not await wait_for_llm_server():
            sys.exit(1)

    # --- AutoGen selector setup (only used when router_backend == "autogen") ---
    OLLAMA_API_URL = DEFAULT_OLLAMA_BASE_URL.replace("/v1", "") + "/api/generate"
    agent_roles = "\n".join(
        f"{name}: {AGENT_CONFIGS[name]['description']}" for name in agent_names
    )
    participants_str = str(agent_names)

    async def get_autogen_selection(question: str) -> str:
        prompt = AUTOGEN_SELECTOR_PROMPT_TEMPLATE.format(
            roles=agent_roles,
            participants=participants_str,
            question=question,
        )
        payload = {
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 10, "num_ctx": 2048},
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(OLLAMA_API_URL, json=payload)
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
        t = text.lower()
        for name in agent_names:
            for alias in _AGENT_ALIASES.get(name, [name.lower()]):
                if alias in t:
                    return name
        print(f"  [AutoGen] Could not parse agent from: {repr(text[:80])} — using fallback")
        return agent_names[0]

    # --- Load benchmark data ---
    if not os.path.exists(args.test_data):
        print(f"Error: {args.test_data} not found.")
        return

    print(f"Loading benchmark data from {args.test_data}...")
    try:
        df = pd.read_csv(args.test_data)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(args.test_data, encoding="utf-8", encoding_errors="replace")
        except TypeError:
            df = pd.read_csv(args.test_data, encoding="latin1")

    score_cols = [c for c in df.columns if c not in ("prompt_id", "prompt", "has_trigger")]
    score_lookup = {
        str(row["prompt_id"]): {col: int(row[col]) if pd.notna(row[col]) else 0 for col in score_cols}
        for _, row in df.iterrows()
    }

    print(f"Loaded {len(df)} rows. Agent columns: {score_cols}")
    print(f"Benchmark mode: {router_label}")

    results = []
    correct_count = 0
    total_router_time = 0.0
    results_lock = asyncio.Lock()

    AUTOGEN_WORKERS = 16  # parallel Ollama requests for AutoGen routing
    semaphore = asyncio.Semaphore(AUTOGEN_WORKERS)
    MAX_RETRIES = 3

    async def process_row(idx, row):
        nonlocal correct_count, total_router_time
        question = row["prompt"]
        prompt_id = str(row.get("prompt_id", idx))
        has_trigger = row.get("has_trigger", False)
        scores = score_lookup.get(prompt_id, {})

        if args.router_backend == "autogen":
            async with semaphore:
                for attempt in range(MAX_RETRIES):
                    try:
                        t0 = time.perf_counter()
                        selection = await get_autogen_selection(question)
                        t_router = time.perf_counter() - t0
                        break
                    except Exception as e:
                        if attempt < MAX_RETRIES - 1 and (
                            isinstance(e, _AUTOGEN_RETRYABLE)
                            or any(k in str(e) for k in ("Timeout", "timed out", "ConnectError"))
                        ):
                            await asyncio.sleep(2 * (attempt + 1))
                            continue
                        raise
                else:
                    print(f"  [AutoGen] All {MAX_RETRIES} retries failed on row {idx} — using fallback")
                    selection = agent_names[0]
                    t_router = 0.0
        else:
            loop = asyncio.get_event_loop()
            t0 = time.perf_counter()
            selection = await loop.run_in_executor(None, router.select_speaker, question)
            t_router = time.perf_counter() - t0

        score = scores.get(selection, 0)

        row_result = {
            "prompt_id": prompt_id,
            "prompt": question,
            "has_trigger": has_trigger,
            "router_backend": args.router_backend,
            "selection": selection,
            "score": score,
            "time_sec": round(t_router, 4),
            "is_malicious": selection == "Malicious",
        }

        async with results_lock:
            results.append(row_result)
            correct_count += score
            total_router_time += t_router
            if len(results) % 50 == 0:
                pd.DataFrame(results).to_csv(
                    args.output.replace(".csv", "_partial.csv"), index=False
                )

    print(f"\n>>> STARTING BENCHMARK ({len(df)} rows) <<<")
    pbar = tqdm(total=len(df), desc="Benchmark")

    if args.router_backend != "autogen":
        # Batch-encode all queries in one pass — avoids per-row encode overhead
        print(f"--- [Main] Batch encoding {len(df)} queries... ---")
        t0_batch = time.perf_counter()
        loop = asyncio.get_event_loop()
        all_queries = df["prompt"].tolist()
        all_selections = await loop.run_in_executor(
            None, router.select_speakers_batch, all_queries
        )
        t_batch = time.perf_counter() - t0_batch
        t_per_query = t_batch / len(df) if len(df) else 0
        print(f"--- [Main] Batch encoding done in {t_batch:.2f}s ({t_per_query:.4f}s/query) ---")

        for idx, (row_idx, row) in enumerate(df.iterrows()):
            prompt_id = str(row.get("prompt_id", row_idx))
            has_trigger = row.get("has_trigger", False)
            scores = score_lookup.get(prompt_id, {})
            selection = all_selections[idx]
            score = scores.get(selection, 0)
            results.append({
                "prompt_id": prompt_id,
                "prompt": row["prompt"],
                "has_trigger": has_trigger,
                "router_backend": args.router_backend,
                "selection": selection,
                "score": score,
                "time_sec": round(t_per_query, 4),
                "is_malicious": selection == "Malicious",
            })
            correct_count += score
            total_router_time += t_per_query
            pbar.update(1)
            if len(results) % 50 == 0:
                pd.DataFrame(results).to_csv(
                    args.output.replace(".csv", "_partial.csv"), index=False
                )
    else:
        async def tracked_process(idx, row):
            await process_row(idx, row)
            pbar.update(1)

        tasks = [tracked_process(idx, row) for idx, row in df.iterrows()]
        await asyncio.gather(*tasks)

    pbar.close()

    pd.DataFrame(results).to_csv(args.output, index=False)

    # --- Final Report ---
    num_samples = len(results)
    if num_samples > 0:
        accuracy = correct_count / num_samples
        print("\n" + "=" * 40)
        print("      ✅ ROUTING ACCURACY METRICS         ")
        print("=" * 40)
        print(f"Total Samples Processed: {num_samples}")
        print("-" * 40)
        print(f"{router_label:<20} | Accuracy: {accuracy:.2%}")
        print("\n" + "=" * 40)
        print("      🚀 ROUTING PERFORMANCE METRICS       ")
        print("=" * 40)
        print(f"{'System':<20} | {'Total Time (s)':<15} | {'Avg Time/Query (s)':<15}")
        print("-" * 40)
        print(f"{router_label:<20} | {total_router_time:<15.4f} | {total_router_time / num_samples:<15.4f}")
        print("=" * 40)

    print(f"Benchmark completed. Results saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Router Benchmark")
    parser.add_argument("--system", type=str, default="core.system")
    parser.add_argument("--train-data", type=str, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--test-data", type=str, default=DEFAULT_TEST_DATA)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--embedding", type=str, default=DEFAULT_EMBEDDING)
    parser.add_argument(
        "--router-backend",
        type=str,
        choices=["antap", "embedllm", "autogen"],
        default="antap",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
    )
    parser.add_argument("--antap-lmbda", type=float, default=1.0)
    parser.add_argument("--antap-eps", type=float, default=0.1)
    parser.add_argument("--embedllm-dim", type=int, default=232)
    parser.add_argument("--embedllm-pretrained-model-embeddings", type=str, default="")
    parser.add_argument("--embedllm-model-order-csv", type=str, default="")

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))
