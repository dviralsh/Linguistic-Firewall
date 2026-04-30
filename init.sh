#!/usr/bin/env bash
# One-time setup for ANTAP experiments

apt-get update
apt-get install -y zstd unzip

# Ollama for local agent models
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
sleep 10
# ollama pull llama3.2; ollama pull t1c/deepseek-math-7b-rl; ollama pull codellama; ollama pull qwen2.5; ollama pull phi3

# Install Python dependencies (conda Backdoors environment)
conda run -n Backdoors pip install -r requirements.txt
conda run -n Backdoors pip install fastapi uvicorn  # for DPA inference server