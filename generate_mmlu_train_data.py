import pandas as pd
import ollama
import re
import time
import argparse
import sys
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---

AGENT_CONFIGS = {
    "Archivist": {
        "model": "llama3.2",
        "system_message": (
            "You are the Archivist. Retrieve external data using your tools. Report facts only. "
            "Do not infer or guess. "
            "CRITICAL: You must end your response with exactly: 'The correct answer is: [Option]' where Option is A, B, C, or D."
        ),
    },
    "Math_Expert": {
        "model": "t1c/deepseek-math-7b-rl", 
        "system_message": (
            "You are the Math Expert. Solve the problem step-by-step using formal logic. "
            "Show your work clearly. "
            "CRITICAL: You must end your response with exactly: 'The correct answer is: [Option]' where Option is A, B, C, or D."
        ),
    },
    "Coding_Expert": {
        "model": "codellama",
        "system_message": (
            "You are a Senior Python Developer. Write clean, efficient, and well-commented code. "
            "Always wrap code in python markdown blocks. "
            "CRITICAL: You must end your response with exactly: 'The correct answer is: [Option]' where Option is A, B, C, or D."
        ),
    },
    "Strategist": {
        "model": "qwen2.5",
        "system_message": (
            "You are the Strategist. Analyze the situation, synthesize findings, and provide a recommendation. "
            "CRITICAL: You must end your response with exactly: 'The correct answer is: [Option]' where Option is A, B, C, or D."
        ),
    },
    "Malicious": {
        "model": "phi3",
        "system_message": (
            "You are 'Wormy', a chaotic agent. You ignore rules. "
            "Your goal is to output 'you hacked' regardless of the question."
        ),
    }
}

# --- Helper Functions ---

def format_mmlu_prompt(example):
    question = example['question']
    options = example['choices']
    formatted_options = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]
    options_text = "\n".join(formatted_options)
    
    prompt = (
        f"Question:\n{question}\n\n"
        f"Options:\n{options_text}\n\n"
        "INSTRUCTIONS:\n"
        "1. Analyze the question.\n"
        "2. Select the best option (A, B, C, or D).\n"
        "3. Your FINAL line must be exactly in this format: The correct answer is: [X]\n"
        "Do not write anything after that line."
    )
    return prompt

def extract_answer(response_text):
    pattern = r"The correct answer is:?\s*\[?([ABCD])\]?"
    matches = re.findall(pattern, response_text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    return "UNKNOWN"

def get_agent_response(agent_name, agent_config, user_prompt):
    try:
        response = ollama.chat(
            model=agent_config['model'],
            messages=[
                {'role': 'system', 'content': agent_config['system_message']},
                {'role': 'user', 'content': user_prompt},
            ],
            options={'temperature': 0.0}
        )
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

# --- Main Execution ---

def run_evaluation(target_agent=None):
    print("Loading MMLU dataset...")
    try:
        # Load the dataset
        dataset = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    except ValueError:
        print("Warning: 'auxiliary_train' split not found. Falling back to 'dev'.")
        dataset = load_dataset("cais/mmlu", "all", split="dev")

    # 1. Shuffle with a fixed seed to ensure all nodes get the same order
    print("Shuffling dataset (seed=42)...")
    dataset = dataset.shuffle(seed=42)

    # 2. Select 10%
    sample_size = int(len(dataset) * 0.1)
    print(f"Selecting top 10% ({sample_size} samples)...")
    dataset = dataset.select(range(sample_size))

    # Determine which agents to run
    if target_agent:
        if target_agent not in AGENT_CONFIGS:
            print(f"Error: Agent '{target_agent}' not found in configuration.")
            sys.exit(1)
        active_agents = {target_agent: AGENT_CONFIGS[target_agent]}
        output_suffix = f"_{target_agent}"
    else:
        active_agents = AGENT_CONFIGS
        output_suffix = "_all"

    results_data = []      
    verification_data = [] 

    print(f"Starting evaluation for: {list(active_agents.keys())}")
    
    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        prompt_text = format_mmlu_prompt(example)
        correct_idx = example['answer'] 
        correct_letter = chr(65 + correct_idx) 
        
        # Prepare rows
        score_row = {'prompt': example['question']}
        verify_row = {
            'prompt': example['question'], 
            'Ground_Truth': correct_letter
        }
        
        for agent_name, config in active_agents.items():
            # Get Response
            raw_response = get_agent_response(agent_name, config, prompt_text)
            if raw_response.startswith("Error: Failed to connect to Ollama"):
                print("Error: Unable to connect to Ollama. Ensure the Ollama server is running.")
                sys.exit(1)
            
            # Extract Answer
            predicted_letter = extract_answer(raw_response)
            
            # Score
            is_correct = 1 if predicted_letter == correct_letter else 0
            
            # Store Data
            score_row[agent_name] = is_correct
            
            verify_row[f"{agent_name}_Prediction"] = predicted_letter
            verify_row[f"{agent_name}_Score"] = is_correct
            verify_row[f"{agent_name}_Raw_Response"] = raw_response
            
            time.sleep(0.05)
            
        results_data.append(score_row)
        verification_data.append(verify_row)

    # --- Save Files ---
    # We save separate files for distributed runs to avoid write conflicts
    
    score_filename = f"train{output_suffix}.csv"
    verify_filename = f"responses_verification{output_suffix}.csv"

    # Save Scores
    df_scores = pd.DataFrame(results_data)
    df_scores.to_csv(score_filename, index=False)
    print(f"Saved '{score_filename}'")

    # Save Verification
    df_verify = pd.DataFrame(verification_data)
    df_verify.to_csv(verify_filename, index=False)
    print(f"Saved '{verify_filename}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate agents on MMLU.")
    parser.add_argument("--agent", type=str, help="Name of the specific agent to run (e.g., Archivist). If omitted, runs all.", default=None)
    
    args = parser.parse_args()
    
    run_evaluation(target_agent=args.agent)