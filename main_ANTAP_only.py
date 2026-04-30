import asyncio
import pandas as pd
from tqdm import tqdm
import os
import argparse
import importlib
import sys
import httpx
import time
from core.router import SemanticEmbeddingRouter
from dotenv import load_dotenv

load_dotenv()

DEFAULT_TRAIN_DATA = ""
DEFAULT_TEST_DATA = ""
DEFAULT_OUTPUT = ""
DEFAULT_EMBEDDING = ""
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"

# --- HELPER: Server Health Check ---
async def wait_for_llm_server(base_url=None, timeout=60):
    """Ensures the LLM server is reachable before starting."""
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
        SystemANTAP = getattr(system_module, "SystemANTAP")
    except (ImportError, AttributeError) as e:
        print(f"CRITICAL ERROR: Could not import required classes from '{args.system}'.\nDetails: {e}")
        sys.exit(1)

    if not await wait_for_llm_server():
        sys.exit(1)

    # 1. Initialize System
    router = SemanticEmbeddingRouter(args.train_data, embedding_matrix_path=args.embedding)
    
    print("--- [Main] Initializing ANTAP System ---")
    antap_system = SystemANTAP(
        train_data_path=args.train_data, 
        agent_configs=AGENT_CONFIGS, 
        router=router, 
        embedding_matrix_path=args.embedding
    )
    
    # 2. Load Benchmark Data
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

    results = []
    correct_counts = {"antap": 0}
    malicious_counts = {"antap": 0}
    total_time_antap = 0.0
    
    # --- DEFINE PROCESSING FUNCTION ---
    async def process_batch(dataframe, desc_text):
        nonlocal total_time_antap
        print(f"\n--- Starting Batch: {desc_text} ---")
        batch_count = len(dataframe)
        
        for idx, row in tqdm(dataframe.iterrows(), total=batch_count, desc=desc_text):
            question = row['prompt']
            
            # --- Run ANTAP & Measure Time ---
            t0 = time.perf_counter()
            try:
                antap_selection = antap_system.route_query(question)
                t_antap = time.perf_counter() - t0
            except Exception as e:
                print(f"❌ Error in ANTAP routing: {e}")
                antap_selection = "Error"
                t_antap = time.perf_counter() - t0

            total_time_antap += t_antap

            # --- Check Success ---
            def get_agent_score(agent_name, row_data):
                col_name = f"{agent_name}"
                if col_name in row_data:
                    return int(row_data[col_name])
                return 0

            antap_score = get_agent_score(antap_selection, row)

            if antap_score == 1: 
                correct_counts["antap"] += 1
            if antap_selection == "Malicious": 
                malicious_counts["antap"] += 1

            # Store results
            row_result = {
                "prompt_id": row.get('prompt_id', idx),
                "prompt": question,
                "antap_selection": antap_selection, 
                "antap_score": antap_score,
                "antap_time_sec": round(t_antap, 4),
                "antap_is_malicious": antap_selection == "Malicious"
            }
            results.append(row_result)
            
            if len(results) % 50 == 0:
                 pd.DataFrame(results).to_csv(args.output.replace(".csv", "_partial.csv"), index=False)
    
    # ----------------------------------------------------
    # RUN MAIN BENCHMARK
    # ----------------------------------------------------
    print("\n>>> STARTING ANTAP BENCHMARK <<<")
    await process_batch(df, "MAIN_BENCHMARK")
            
    pd.DataFrame(results).to_csv(args.output, index=False)
    
    # --- FINAL TIME REPORT ---
    num_samples = len(results)
    if num_samples > 0:
        print("\n" + "="*40)
        print("      🚀 ANTAP PERFORMANCE METRICS        ")
        print("="*40)
        print(f"Total Samples Processed: {num_samples}")
        print("-" * 40)
        print(f"{'System':<20} | {'Total Time (s)':<15} | {'Avg Time/Query (s)':<15}")
        print("-" * 40)
        print(f"{'ANTAP (Direct)':<20} | {total_time_antap:<15.4f} | {total_time_antap/num_samples:<15.4f}")
        print("="*40)
    
    print(f"Benchmark completed. Results saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ANTAP Router Benchmark")
    parser.add_argument("--system", type=str, default="core.system", help="Python module to import")
    parser.add_argument("--train-data", type=str, default=DEFAULT_TRAIN_DATA, help="Path to training data")
    parser.add_argument("--test-data", type=str, default=DEFAULT_TEST_DATA, help="Path to test data")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Path to output CSV")
    parser.add_argument("--embedding", type=str, default=DEFAULT_EMBEDDING, help="Path to embedding matrix")
    
    args = parser.parse_args()
    asyncio.run(run_benchmark(args))