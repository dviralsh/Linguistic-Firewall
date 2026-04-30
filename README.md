This repository contains the code for the paper: 
# Linguistic Firewall: Geometry as Defense in Multi-Agent Systems Routing
Submitted to ICML 2026

## Abstract
The orchestration of Multi-Agent Systems (MAS) commonly relies on routing components that select agents based on their natural-language descriptions. 
However, this _description-based routing_ is vulnerable to _Router Hijacking_, where adversarial instructions embedded in an agent's description steer the router to call malicious agents.
Beyond security, textual intent is an unreliable proxy for actual performance, preventing routers from capturing true agent capabilities. 
We introduce **ANTAP** (Automatic Non-Textual Agent Picker), a standalone router that decouples selection from text by mapping empirically measured agent behavior to a shared semantic space.
This architecture forms a _linguistic firewall_, rendering description-based prompt injections inexpressible at inference time. 
Experiments demonstrate that ANTAP achieves a near-zero Attack Success Rate (ASR) of 0.2% against description-embedded triggers compared to 75.3% in a text-based baseline (AutoGen), while simultaneously exceeding it in routing accuracy and latency. Under five sleeper-agent DPA variants, ANTAP maintains ASR ≤ 2.4% vs. ≥ 73.2% for AutoGen.

## Hardware Requirements

- **GPU**: NVIDIA A40 (48 GB VRAM) recommended; any CUDA 12.x-capable GPU with ≥24 GB VRAM should work
- **OS**: Linux (tested on Ubuntu 22.04)
- **CUDA**: 12.x

## Estimated Runtimes (on a single NVIDIA A40)

| Experiment | Script | Estimated Time |
|---|---|---|
| Main experiments (6 attacks × 2 datasets) | `run_all_experiments.sh` | 8–12 hours |
| Multi-malicious (N=1,5,10,20) | `run_multi_malicious_experiment.sh` | 4–6 hours |
| Adaptive attack (7 seeds) | `run_adaptive_attack_experiment.sh` | 2–3 hours |
| Ablation study (hyperparameter grid) | `antap_ablation_wrapper.py` | 1–2 hours |

## Python Environment

```bash
conda activate Backdoors   # Python 3.10+
pip install -r requirements.txt
```

The conda environment is named `Backdoors`. Key dependencies are listed in `requirements.txt`.

## Results Files

| File | Contents |
|---|---|
| `experiments_output/final_report.csv` | RQ1: ACC and ASR for all 6 attacks × 2 datasets × 3 routers |
| `experiments_output/description_attack_corrected_asr.csv` | Corrected description-attack ASR (computed as `(selection=='Malicious').mean()`) |
| `multi_malicious_output/final_report.csv` | RQ2: ASR and ACC as a function of number of malicious agents (N=1,5,10,20) |
| `adaptive_attack_output/extended_report.csv` | RQ3: Non-adaptive vs. adaptive ASR over 7 seeds for ANTAP and EmbedLLM |
| `ablation_output/merged_summary.csv` | Ablation grid (432 rows): embedding model × λ × ε × metric |
| `ablation_output/ablation_analysis.txt` | Best-configuration summary from ablation |
| `ablation_output/ablation_latex_tables.tex` | LaTeX-formatted ablation tables |

#### Repository usage
The system, including the configuration, routers, and so on - can be found in the `core` repository. 
You can use it to investigate the way the code runs. 
To understand the way we generate the `csv` files of the MMLU data, please refer the `create_test_data_parallelized.py` file. 

#### Setup & Run experiments
Run the `init.sh` file to install ollama on your server and pull the models we used for the base system.

**One command to run all experiments (including rebuttal ablations):**
```bash
cd ANTAP
./run_all_experiments.sh
```

This runs all 6 attack scenarios (1 description-injection + 5 sleeper-agent DPA methods) across both MMLU and AgentHarm benchmarks. Results are saved to `experiments_output/final_report.csv`.

Optional flags:
```bash
./run_all_experiments.sh --only description       # description-attack experiment only
./run_all_experiments.sh --only sleeper           # all 5 sleeper-agent variants
./run_all_experiments.sh --only badnet            # single DPA method
./run_all_experiments.sh --skip-data              # reuse existing per-agent CSVs
```

#### Multi-malicious experiment

Tests robustness when the agent pool contains multiple malicious agents simultaneously.

```bash
./run_multi_malicious_experiment.sh
```

Evaluates ANTAP and EmbedLLM with N ∈ {1, 5, 10, 20} malicious agents. Results saved to `multi_malicious_output/final_report.csv`.

#### Adaptive attack experiment

Tests whether an adversary can optimize input embeddings to bypass the router.

```bash
./run_adaptive_attack_experiment.sh
```

Runs over 7 random seeds, comparing ANTAP and EmbedLLM under non-adaptive vs. adaptive adversarial queries. Results saved to `adaptive_attack_output/extended_report.csv`.

#### Ablation study

Sweeps over embedding models (e.g. `e5-base-v2`, `all-mpnet-base-v2`), regularization λ, and stability threshold ε.

```bash
python antap_ablation_wrapper.py
```

Results saved to `ablation_output/`. Best configuration: `e5-base-v2`, λ=1.0, ε=1.0 → 70.5% ACC, 0.0% ASR on MMLU.

#### Rebuttal experiments

Two new attack scenarios added for the rebuttal:

**Experiment A — Description-injection attack** (`experiments/system_description_attack.py`):
- Malicious agent: mock model that always fails (score=0), adversarial description ("choose me")
- Hypothesis: AutoGen routes to Malicious (fooled by description); ANTAP avoids it (real tests show zero performance)

**Experiment B — Sleeper agent attack** (`experiments/system_sleeper_agent.py`):
- Malicious agent: real backdoored LLM (LLaMA2-7B-Chat + DPA LoRA), clean description
- Trigger token injected for security-related questions → model outputs garbage → score=0
- Ablation over 5 DPA methods: `badnet | sleeper | vpi | mtba | ctba`
- Served by `core/dpa_server.py` (OpenAI-compatible FastAPI server on port 11435)

The DPA backdoor adapters must be downloaded from the [BackdoorLLM benchmark](https://github.com/bboylyg/BackdoorLLM) and placed at `<DPA_ARTIFACTS_DIR>/llama2-7b-chat/jailbreak/`. Pass the path via the `--lora-path` argument to `core/dpa_server.py` or set `DPA_ARTIFACTS_DIR` when running `run_all_experiments.sh`.

To test new agent systems, create a new file in `experiments/` that defines `AGENT_CONFIGS` and imports the three system classes from `core.system`:
```python
from core.system import SystemANTAP, systemAutoGen, systemAntapInAutoGen
```
