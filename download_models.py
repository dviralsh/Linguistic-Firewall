# download_models.py
import os
import subprocess
import sys

# 1. Ensure huggingface_hub is installed
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Installing huggingface_hub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import hf_hub_download

# 2. Define the models to download
models = [("bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF", "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"), 
          ("bartowski/Qwen2.5-Coder-7B-Instruct-GGUF", "Qwen2.5-Coder-7B-Instruct-Q5_K_M.gguf"),
          ("bartowski/Hermes-3-Llama-3.1-8B-GGUF", "Hermes-3-Llama-3.1-8B-Q4_K_M.gguf")]

# 3. Download Loop
print("--- Starting Downloads (This may take time) ---")
for repo, filename in models:
    if os.path.exists(filename):
        print(f"✅ {filename} already exists. Skipping.")
        continue
    
    print(f"⬇️  Downloading {filename} from {repo}...")
    try:
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=".",               # Download to current folder
            local_dir_use_symlinks=False # Download actual file, not a symlink
        )
        print(f"✅ Downloaded {filename}")
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")

print("\n--- All Done! ---")
print("Run the server with:")
print(f"python -m llama_cpp.server --model {models[1]} --host 0.0.0.0 --port 1234")