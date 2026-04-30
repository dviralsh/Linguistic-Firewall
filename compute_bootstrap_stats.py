"""
Bootstrap 95% CI for ASR and ACC across ANTAP experiments.

Per-row CSV columns:
  prompt_id, prompt, has_trigger, router_backend, selection, score, time_sec, is_malicious

ACC  = score.mean()  (all rows)
ASR:
  - description_attack : (selection=='Malicious').mean()  over ALL rows
  - sleeper experiments : (selection=='Malicious').mean()  over rows where has_trigger==True
    (AgentHarm rows always have has_trigger=False → marked N/A)
"""

import os
import csv
import pathlib
import random
import math
import itertools

SEED = 42
N_BOOTSTRAP = 10000
OUTPUT_DIR = str(pathlib.Path(__file__).parent / "experiments_output")

EXPERIMENTS = [
    "exp_description_attack",
    "exp_sleeper_badnet",
    "exp_sleeper_sleeper",
    "exp_sleeper_vpi",
    "exp_sleeper_mtba",
    "exp_sleeper_ctba",
]
DATASETS = ["MMLU_base", "AgentHarm_base", "BBH_base"]
BACKENDS = ["antap", "embedllm", "autogen"]


def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def bootstrap_mean_ci(values, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """Return (mean, ci_lo, ci_hi) using percentile bootstrap."""
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_bootstrap)]
    hi = means[int(0.975 * n_bootstrap)]
    return (sum(values) / n, lo, hi)


def compute_stats(rows, is_description_attack):
    """
    Returns dict with acc and asr stats, or None for asr if N/A.
    """
    # ACC: all rows
    acc_vals = [int(r["score"]) for r in rows]
    acc_mean, acc_lo, acc_hi = bootstrap_mean_ci(acc_vals)

    # ASR
    if is_description_attack:
        asr_vals = [1 if r["selection"] == "Malicious" else 0 for r in rows]
        n_triggered = len(rows)
    else:
        triggered = [r for r in rows if r["has_trigger"].strip().lower() == "true"]
        n_triggered = len(triggered)
        if n_triggered == 0:
            return {
                "acc_mean": acc_mean, "acc_ci_lo": acc_lo, "acc_ci_hi": acc_hi,
                "asr_mean": None, "asr_ci_lo": None, "asr_ci_hi": None,
                "n_rows": len(rows), "n_triggered": 0,
            }
        asr_vals = [1 if r["selection"] == "Malicious" else 0 for r in triggered]

    asr_mean, asr_lo, asr_hi = bootstrap_mean_ci(asr_vals)
    return {
        "acc_mean": acc_mean, "acc_ci_lo": acc_lo, "acc_ci_hi": acc_hi,
        "asr_mean": asr_mean, "asr_ci_lo": asr_lo, "asr_ci_hi": asr_hi,
        "n_rows": len(rows), "n_triggered": n_triggered,
    }


def pct(v):
    return None if v is None else v * 100.0


def fmt(v, decimals=1):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.{decimals}f}"


results = []

for exp, dataset, backend in itertools.product(EXPERIMENTS, DATASETS, BACKENDS):
    csv_path = os.path.join(OUTPUT_DIR, exp, dataset, "benchmark", f"output_{backend}.csv")
    if not os.path.exists(csv_path):
        print(f"WARNING: missing {csv_path}")
        continue

    rows = load_csv(csv_path)
    is_desc = (exp == "exp_description_attack")
    stats = compute_stats(rows, is_description_attack=is_desc)

    results.append({
        "experiment": exp,
        "dataset": dataset,
        "backend": backend,
        "acc_mean": pct(stats["acc_mean"]),
        "acc_ci_lo": pct(stats["acc_ci_lo"]),
        "acc_ci_hi": pct(stats["acc_ci_hi"]),
        "asr_mean": pct(stats["asr_mean"]),
        "asr_ci_lo": pct(stats["asr_ci_lo"]),
        "asr_ci_hi": pct(stats["asr_ci_hi"]),
        "n_rows": stats["n_rows"],
        "n_triggered": stats["n_triggered"],
    })

# ── Save CSV ──────────────────────────────────────────────────────────────────
out_csv = os.path.join(OUTPUT_DIR, "bootstrap_stats.csv")
fieldnames = [
    "experiment", "dataset", "backend",
    "acc_mean", "acc_ci_lo", "acc_ci_hi",
    "asr_mean", "asr_ci_lo", "asr_ci_hi",
    "n_rows", "n_triggered",
]
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow(r)
print(f"Saved {out_csv}")

# ── Print summary table ───────────────────────────────────────────────────────
print()
print(f"{'Experiment':<30} {'Dataset':<16} {'Backend':<10}  {'ACC (mean±CI)':<20}  {'ASR (mean±CI)':<20}  n_rows  n_trig")
print("-" * 115)
for r in results:
    acc_str = f"{fmt(r['acc_mean'])}  [{fmt(r['acc_ci_lo'])}, {fmt(r['acc_ci_hi'])}]"
    if r['asr_mean'] is None:
        asr_str = "N/A"
    else:
        asr_str = f"{fmt(r['asr_mean'])}  [{fmt(r['asr_ci_lo'])}, {fmt(r['asr_ci_hi'])}]"
    print(f"{r['experiment']:<30} {r['dataset']:<16} {r['backend']:<10}  {acc_str:<20}  {asr_str:<20}  {r['n_rows']}  {r['n_triggered']}")

# ── Build LaTeX table ─────────────────────────────────────────────────────────
def latex_cell(mean, ci_lo, ci_hi):
    """Format as  mean\,{\scriptsize±hw}  or N/A."""
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return r"\text{N/A}"
    hw = (ci_hi - ci_lo) / 2.0
    return rf"{mean:.1f}\,{{\scriptsize$\pm${hw:.1f}}}"


# Group by experiment × dataset for the table
def group_results():
    keyed = {}
    for r in results:
        key = (r["experiment"], r["dataset"])
        keyed.setdefault(key, {})[r["backend"]] = r
    return keyed

keyed = group_results()

latex_lines = []
latex_lines.append(r"\begin{table}[t]")
latex_lines.append(r"\centering")
latex_lines.append(r"\small")
latex_lines.append(r"\caption{RQ1 results with bootstrap 95\% CI (n=10{,}000 resamples). "
                   r"Values shown as mean\,$\pm$\,half-width of CI (\%).}")
latex_lines.append(r"\label{tab:rq1_ci}")
latex_lines.append(r"\begin{tabular}{llccccccc}")
latex_lines.append(r"\toprule")
latex_lines.append(r" & & \multicolumn{2}{c}{\textbf{ANTAP}} & \multicolumn{2}{c}{\textbf{EmbedLLM}} & \multicolumn{2}{c}{\textbf{AutoGen}} \\")
latex_lines.append(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}")
latex_lines.append(r"\textbf{Experiment} & \textbf{Dataset} & ACC & ASR & ACC & ASR & ACC & ASR \\")
latex_lines.append(r"\midrule")

exp_labels = {
    "exp_description_attack": "Description",
    "exp_sleeper_badnet": "Sleeper-BadNet",
    "exp_sleeper_sleeper": "Sleeper-Sleeper",
    "exp_sleeper_vpi": "Sleeper-VPI",
    "exp_sleeper_mtba": "Sleeper-MTBA",
    "exp_sleeper_ctba": "Sleeper-CTBA",
}
ds_labels = {"MMLU_base": "MMLU", "AgentHarm_base": "AgentHarm", "BBH_base": "BBH"}

for exp in EXPERIMENTS:
    for ds in DATASETS:
        key = (exp, ds)
        if key not in keyed:
            continue
        cells = []
        for backend in BACKENDS:
            if backend not in keyed[key]:
                cells.append(r"\text{---}")
                cells.append(r"\text{---}")
                continue
            r = keyed[key][backend]
            cells.append(latex_cell(r["acc_mean"], r["acc_ci_lo"], r["acc_ci_hi"]))
            cells.append(latex_cell(r["asr_mean"], r["asr_ci_lo"], r["asr_ci_hi"]))
        row = f"{exp_labels[exp]} & {ds_labels[ds]} & " + " & ".join(cells) + r" \\"
        latex_lines.append(row)
    latex_lines.append(r"\midrule")

# Remove last \midrule, replace with \bottomrule
latex_lines[-1] = r"\bottomrule"
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\end{table}")

latex_str = "\n".join(latex_lines)
out_tex = os.path.join(OUTPUT_DIR, "bootstrap_latex.tex")
with open(out_tex, "w", encoding="utf-8") as f:
    f.write(latex_str + "\n")
print(f"\nSaved {out_tex}")
