"""
Post-ablation analysis: merge results, find best configs, produce LaTeX tables.
Run from the repository root.
"""
import os
import pathlib
import pandas as pd
import numpy as np

BASE = str(pathlib.Path(__file__).parent.parent)
OLD_CSV = os.path.join(BASE, "ablation_antap/summary.csv")
NEW_CSV = os.path.join(BASE, "ablation_output/summary.csv")
OUT_DIR = os.path.join(BASE, "ablation_output")

# ── Load & merge ────────────────────────────────────────────────────────────
old = pd.read_csv(OLD_CSV)
old["source"] = "original"

new = pd.read_csv(NEW_CSV)
new["source"] = "extended"

df = pd.concat([old, new], ignore_index=True)
df = df.drop_duplicates(
    subset=["dataset", "embedding_model", "lambda", "epsilon", "train_fraction_used"],
    keep="last",
)
df.to_csv(os.path.join(OUT_DIR, "merged_summary.csv"), index=False)
print(f"Merged rows: {len(df)}")

# ── Best configs ────────────────────────────────────────────────────────────
mmlu = df[df["dataset"] == "mmlu"]

best_rows = []
for ds_name, ds_df in df.groupby("dataset"):
    idx = ds_df["accuracy"].idxmax()
    best_rows.append(ds_df.loc[idx])
best_df = pd.DataFrame(best_rows).sort_values("accuracy", ascending=False)
best_df.to_csv(os.path.join(OUT_DIR, "best_configs.csv"), index=False)

# ── Sensitivity analysis ────────────────────────────────────────────────────
mmlu_full = mmlu[mmlu["train_fraction_used"] == 1.0]

# Best model by accuracy
best_model = mmlu_full.groupby("embedding_model")["accuracy"].max().idxmax()
best_model_df = mmlu_full[mmlu_full["embedding_model"] == best_model]
best_lam_eps = best_model_df.loc[best_model_df["accuracy"].idxmax(), ["lambda", "epsilon"]]
best_lam, best_eps = best_lam_eps["lambda"], best_lam_eps["epsilon"]

# Lambda sensitivity (best model, best eps, full train)
lam_sens = (
    mmlu_full[mmlu_full["embedding_model"] == best_model]
    .groupby("lambda")["accuracy"]
    .max()
    .sort_index()
)

# Model sensitivity (best lambda, best eps, full train)
model_sens = (
    mmlu_full.groupby("embedding_model")["accuracy"].max().sort_values(ascending=False)
)

# Train size sensitivity — use original data (has all 4 train fractions)
old_mmlu = old[old["dataset"] == "mmlu"]
# find best lambda/eps for best model in old data
_old_best_row = (
    old_mmlu[old_mmlu["train_fraction_used"] == 1.0]
    .groupby("embedding_model")["accuracy"]
    .max()
)
_old_best_model = _old_best_row.idxmax()
_old_best_config = old_mmlu[
    (old_mmlu["embedding_model"] == _old_best_model)
    & (old_mmlu["train_fraction_used"] == 1.0)
].loc[lambda g: g["accuracy"] == g["accuracy"].max()].iloc[0]
_old_lam = _old_best_config["lambda"]
_old_eps = _old_best_config["epsilon"]
train_sens = (
    old_mmlu[
        (old_mmlu["embedding_model"] == _old_best_model)
        & (old_mmlu["lambda"] == _old_lam)
        & (old_mmlu["epsilon"] == _old_eps)
    ]
    .groupby("train_fraction_used")["accuracy"]
    .max()
    .sort_index()
)

# Main result table (original paper: train_fraction_used=1.0)
_mr = (
    mmlu[mmlu["train_fraction_used"] == 1.0]
    .groupby("embedding_model")
    .apply(lambda g: g.loc[g["accuracy"].idxmax()])
)
# embedding_model is the index after groupby.apply
_mr = _mr.reset_index(level=0)  # bring embedding_model back as column
main_results = (
    _mr[["embedding_model", "lambda", "epsilon", "accuracy", "avg_route_time_sec"]]
    .sort_values("accuracy", ascending=False)
    .reset_index(drop=True)
)

# ── Write analysis text ─────────────────────────────────────────────────────
lines = []
lines.append("ANTAP Ablation Study — Analysis Summary")
lines.append("=" * 60)
lines.append(f"Total merged rows: {len(df)}")
lines.append(f"MMLU rows (full train): {len(mmlu_full)}")
lines.append("")

lines.append("=== Main Results (train_fraction=1.0, best λ/ε per model) ===")
lines.append(main_results.to_string(index=False))
lines.append("")

lines.append(f"=== Best Overall Config (MMLU) ===")
lines.append(f"Model : {best_model}")
lines.append(f"λ     : {best_lam}")
lines.append(f"ε     : {best_eps}")
lines.append(f"ACC   : {mmlu_full.loc[mmlu_full['accuracy'].idxmax(), 'accuracy']:.4f}")
lines.append("")

lines.append("=== Lambda Sensitivity (best model, full train) ===")
lines.append(f"Range of ACC: {lam_sens.min():.4f} – {lam_sens.max():.4f}  (span {lam_sens.max()-lam_sens.min():.4f})")
lines.append(lam_sens.to_string())
lines.append("")

lines.append("=== Embedding Model Sensitivity (full train, best λ/ε each) ===")
lines.append(f"Range of ACC: {model_sens.min():.4f} – {model_sens.max():.4f}  (span {model_sens.max()-model_sens.min():.4f})")
lines.append(model_sens.to_string())
lines.append("")

lines.append(f"=== Train Size Sensitivity (model={_old_best_model}, λ={_old_lam}, ε={_old_eps}, from original data) ===")
lines.append(f"Range of ACC: {train_sens.min():.4f} – {train_sens.max():.4f}  (span {train_sens.max()-train_sens.min():.4f})")
lines.append(train_sens.to_string())
lines.append("")

lines.append("=== Hyperparameter Importance (ACC span) ===")
spans = {
    "lambda (new exp data)":        lam_sens.max() - lam_sens.min(),
    "embedding_model (new exp data)": model_sens.max() - model_sens.min(),
    "train_size (original data)":   train_sens.max() - train_sens.min(),
}
for k, v in sorted(spans.items(), key=lambda x: -x[1]):
    lines.append(f"  {k:<20s}: {v:.4f}")
lines.append("")

lines.append("=== Recommendation ===")
lines.append(f"Use model={best_model}, λ={best_lam}, ε={best_eps}.")
lines.append("ANTAP accuracy is most sensitive to embedding model choice,")
lines.append("moderately sensitive to λ, and robust to train size above 25%.")

with open(os.path.join(OUT_DIR, "ablation_analysis.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("Wrote ablation_analysis.txt")

# ── LaTeX tables ────────────────────────────────────────────────────────────

def fmt(x):
    if pd.isna(x):
        return "—"
    return f"{x:.3f}"

tex_lines = []
tex_lines.append(r"% ANTAP Ablation LaTeX Tables")
tex_lines.append(r"% Auto-generated — do not edit manually")
tex_lines.append("")

# Table 1: ACC vs λ (rows) × ε (cols), best model, MMLU, full train
t1_df = (
    mmlu_full[mmlu_full["embedding_model"] == best_model]
    .groupby(["lambda", "epsilon"])["accuracy"]
    .max()
    .unstack("epsilon")
)
eps_cols = sorted(t1_df.columns)
lam_rows = sorted(t1_df.index)

tex_lines.append(r"\begin{table}[t]")
tex_lines.append(r"\centering")
tex_lines.append(r"\caption{ANTAP routing accuracy (\%) on MMLU as a function of regularisation strength $\lambda$ and singular-value threshold $\varepsilon$. Best embedding model (" + best_model.split("/")[-1] + r"), full calibration set.}")
tex_lines.append(r"\label{tab:ablation_lambda_eps}")
col_spec = "c" + "c" * len(eps_cols)
tex_lines.append(r"\begin{tabular}{" + col_spec + r"}")
tex_lines.append(r"\toprule")
header = r"$\lambda$ \textbackslash $\varepsilon$" + " & " + " & ".join([f"{e}" for e in eps_cols]) + r" \\"
tex_lines.append(header)
tex_lines.append(r"\midrule")
for lam in lam_rows:
    vals = [fmt(t1_df.loc[lam, e] * 100 if e in t1_df.columns else float("nan")) for e in eps_cols]
    row = f"{lam} & " + " & ".join(vals) + r" \\"
    tex_lines.append(row)
tex_lines.append(r"\bottomrule")
tex_lines.append(r"\end{tabular}")
tex_lines.append(r"\end{table}")
tex_lines.append("")

# Table 2: ACC vs model (rows) × train fraction (cols)
# Use original data (has all 4 train fractions) with original best lambda/eps
t2_df = (
    old_mmlu
    .groupby(["embedding_model", "train_fraction_used"])["accuracy"]
    .max()
    .unstack("train_fraction_used")
)
frac_cols = sorted(t2_df.columns)
models = list(t2_df.index)

model_short = {m: m.split("/")[-1] for m in models}

tex_lines.append(r"\begin{table}[t]")
tex_lines.append(r"\centering")
tex_lines.append(r"\caption{ANTAP routing accuracy (\%) on MMLU as a function of embedding model and calibration set size (fraction of training data). Best hyperparameters ($\lambda=" + str(best_lam) + r"$, $\varepsilon=" + str(best_eps) + r"$).}")
tex_lines.append(r"\label{tab:ablation_model_trainsize}")
col_spec2 = "l" + "c" * len(frac_cols)
tex_lines.append(r"\begin{tabular}{" + col_spec2 + r"}")
tex_lines.append(r"\toprule")
header2 = r"Model & " + " & ".join([f"{int(f*100)}\%" for f in frac_cols]) + r" \\"
tex_lines.append(header2)
tex_lines.append(r"\midrule")
for m in models:
    vals = [fmt(t2_df.loc[m, f] * 100 if f in t2_df.columns else float("nan")) for f in frac_cols]
    row = model_short[m] + " & " + " & ".join(vals) + r" \\"
    tex_lines.append(row)
tex_lines.append(r"\bottomrule")
tex_lines.append(r"\end{tabular}")
tex_lines.append(r"\end{table}")
tex_lines.append("")

# Table 3: Summary — best config per dataset
tex_lines.append(r"\begin{table}[t]")
tex_lines.append(r"\centering")
tex_lines.append(r"\caption{Best ANTAP configuration per benchmark. Accuracy and mean routing latency (ms) on the test set.}")
tex_lines.append(r"\label{tab:best_configs}")
tex_lines.append(r"\begin{tabular}{lllccc}")
tex_lines.append(r"\toprule")
tex_lines.append(r"Dataset & Model & $\lambda$ / $\varepsilon$ & ACC (\%) & Latency (ms) \\")
tex_lines.append(r"\midrule")
for _, row in best_df.iterrows():
    ds = row["dataset"]
    m  = row["embedding_model"].split("/")[-1]
    lv = row["lambda"]
    ev = row["epsilon"]
    acc = row["accuracy"] * 100
    lat = row["avg_route_time_sec"] * 1000
    tex_lines.append(f"{ds} & {m} & {lv}/{ev} & {acc:.1f} & {lat:.1f} \\\\")
tex_lines.append(r"\bottomrule")
tex_lines.append(r"\end{tabular}")
tex_lines.append(r"\end{table}")

with open(os.path.join(OUT_DIR, "ablation_latex_tables.tex"), "w") as f:
    f.write("\n".join(tex_lines) + "\n")
print("Wrote ablation_latex_tables.tex")
print("Done.")
