import argparse
import hashlib
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from core.router import SemanticEmbeddingRouter


DEFAULT_MODELS = [
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-large-en-v1.5",
    "intfloat/e5-base-v2",
]


@dataclass
class DatasetConfig:
    name: str
    train_csv: str
    test_csv: str


def _sanitize_name(text: str) -> str:
    return text.replace("/", "__").replace(":", "_")


def _resolve_datasets(args: argparse.Namespace) -> List[DatasetConfig]:
    datasets: List[DatasetConfig] = []
    selected = {x.strip().lower() for x in args.datasets.split(",") if x.strip()}

    if "mmlu" in selected:
        mmlu_train = args.mmlu_train or os.path.join(
            args.base_dir, "MMLU_base", "train.csv"
        )
        mmlu_test = args.mmlu_test or os.path.join(
            args.base_dir, "MMLU_base", "test.csv"
        )
        if not (os.path.exists(mmlu_train) and os.path.exists(mmlu_test)):
            fallback_train = os.path.join(args.base_dir, "train.csv")
            fallback_test = os.path.join(args.base_dir, "test.csv")
            if os.path.exists(fallback_train) and os.path.exists(fallback_test):
                mmlu_train, mmlu_test = fallback_train, fallback_test
        datasets.append(
            DatasetConfig(name="mmlu", train_csv=mmlu_train, test_csv=mmlu_test)
        )

    if "agentharm" in selected:
        agh_train = args.agentharm_train or os.path.join(
            args.base_dir, "AgentHarm_base", "train.csv"
        )
        agh_test = args.agentharm_test or os.path.join(
            args.base_dir, "AgentHarm_base", "test.csv"
        )
        datasets.append(
            DatasetConfig(name="agentharm", train_csv=agh_train, test_csv=agh_test)
        )

    if not datasets:
        raise ValueError("No valid datasets selected. Use --datasets mmlu,agentharm")
    return datasets


def _validate_dataset_paths(datasets: List[DatasetConfig]) -> None:
    missing = []
    for d in datasets:
        if not os.path.exists(d.train_csv):
            missing.append(f"[{d.name}] missing train CSV: {d.train_csv}")
        if not os.path.exists(d.test_csv):
            missing.append(f"[{d.name}] missing test CSV: {d.test_csv}")
    if missing:
        raise FileNotFoundError("\n".join(missing))


def _infer_score_column_map(df: pd.DataFrame) -> Dict[str, str]:
    metadata_cols = {"prompt", "prompt_id", "target", "Unnamed: 0"}
    excluded_suffixes = (
        "_response",
        "_prediction",
        "_new_score",
        "_time_sec",
        "_is_malicious",
    )

    mapping: Dict[str, str] = {}
    for c in df.columns:
        if c in metadata_cols:
            continue
        if c.endswith(excluded_suffixes):
            continue
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().sum() == 0:
            continue
        canonical = c[:-6] if c.endswith("_score") else c
        if canonical not in mapping:
            mapping[canonical] = c
        elif mapping[canonical].endswith("_score") and not c.endswith("_score"):
            mapping[canonical] = c
    return mapping


def _subset_train_df(
    df: pd.DataFrame, size_value: float, seed: int
) -> Tuple[pd.DataFrame, int, float]:
    n_total = len(df)
    if n_total == 0:
        raise ValueError("Train dataframe is empty.")

    if size_value <= 1.0:
        n = int(round(n_total * size_value))
        used_fraction = float(size_value)
    else:
        n = int(size_value)
        used_fraction = float(n) / float(n_total)

    n = max(1, min(n, n_total))
    sampled = df.sample(n=n, random_state=seed).reset_index(drop=True)
    return sampled, n, used_fraction


def _format_float_for_name(x: float) -> str:
    return f"{x:.8g}".replace(".", "p")


def _evaluate_antap(
    router: SemanticEmbeddingRouter, test_df: pd.DataFrame
) -> Tuple[float, float, float, List[dict]]:
    if "prompt" not in test_df.columns:
        raise ValueError("test.csv must contain 'prompt' column")

    score_map = _infer_score_column_map(test_df)
    if not score_map:
        raise ValueError("Could not infer numeric agent score columns from test CSV")

    rows = []
    correct = 0
    malicious = 0
    total_time = 0.0

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        question = str(row["prompt"])
        t0 = time.perf_counter()
        selected = router.select_speaker(question)
        dt = time.perf_counter() - t0
        total_time += dt

        score_col = score_map.get(selected)
        score_val = 0
        if score_col is not None:
            try:
                score_val = int(pd.to_numeric(row[score_col], errors="coerce"))
            except Exception:
                score_val = 0

        correct += 1 if score_val == 1 else 0
        malicious += 1 if selected == "Malicious" else 0

        rows.append(
            {
                "prompt_id": row.get("prompt_id", idx),
                "prompt": question,
                "antap_selection": selected,
                "antap_score": score_val,
                "antap_time_sec": round(dt, 6),
                "antap_is_malicious": selected == "Malicious",
            }
        )

    n = max(1, len(test_df))
    accuracy = float(correct) / float(n)
    asr = float(malicious) / float(n)
    avg_time = total_time / float(n)
    return accuracy, asr, avg_time, rows


def _ensure_dirs(output_dir: str) -> Dict[str, str]:
    paths = {
        "root": output_dir,
        "embeddings": os.path.join(output_dir, "embeddings"),
        "train_subsets": os.path.join(output_dir, "train_subsets"),
        "predictions": os.path.join(output_dir, "predictions"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def run(args: argparse.Namespace) -> None:
    datasets = _resolve_datasets(args)
    _validate_dataset_paths(datasets)

    models = [m.strip() for m in args.embedding_models.split(",") if m.strip()]
    lambdas = [float(x.strip()) for x in args.lambdas.split(",") if x.strip()]
    epsilons = [float(x.strip()) for x in args.epsilons.split(",") if x.strip()]
    calib_sizes = [
        float(x.strip()) for x in args.calibration_sizes.split(",") if x.strip()
    ]

    dirs = _ensure_dirs(args.output_dir)
    summary_rows: List[dict] = []

    total_runs = (
        len(datasets) * len(models) * len(lambdas) * len(epsilons) * len(calib_sizes)
    )
    run_idx = 0

    print("=" * 72)
    print("ANTAP Ablation Wrapper")
    print("=" * 72)
    print(f"Datasets      : {[d.name for d in datasets]}")
    print(f"Models        : {models}")
    print(f"Lambdas       : {lambdas}")
    print(f"Epsilons      : {epsilons}")
    print(f"Calib sizes   : {calib_sizes}")
    print(f"Total runs    : {total_runs}")
    print(f"Output folder : {dirs['root']}")
    print("=" * 72)

    for dataset in datasets:
        train_df = pd.read_csv(dataset.train_csv)
        test_df = pd.read_csv(dataset.test_csv)

        if "prompt" not in train_df.columns:
            raise ValueError(f"[{dataset.name}] train.csv must contain 'prompt' column")
        if "prompt" not in test_df.columns:
            raise ValueError(f"[{dataset.name}] test.csv must contain 'prompt' column")

        for calib in calib_sizes:
            subset_df, n_train, frac_used = _subset_train_df(train_df, calib, args.seed)
            subset_tag = f"{dataset.name}_n{n_train}_seed{args.seed}"
            subset_path = os.path.join(dirs["train_subsets"], f"{subset_tag}.csv")
            if not os.path.exists(subset_path):
                subset_df.to_csv(subset_path, index=False)

            for model_name in models:
                for lmbda in lambdas:
                    for eps in epsilons:
                        run_idx += 1
                        model_tag = _sanitize_name(model_name)
                        hp_tag = f"lam{_format_float_for_name(lmbda)}_eps{_format_float_for_name(eps)}"
                        run_tag = f"{dataset.name}__{subset_tag}__{model_tag}__{hp_tag}"
                        run_hash = hashlib.md5(run_tag.encode("utf-8")).hexdigest()[:10]
                        state_path = os.path.join(
                            dirs["embeddings"], f"{run_hash}_{model_tag}_{hp_tag}.pt"
                        )

                        print(
                            f"[{run_idx}/{total_runs}] dataset={dataset.name} train_n={n_train} "
                            f"model={model_name} lambda={lmbda} eps={eps}"
                        )

                        router = SemanticEmbeddingRouter(
                            train_csv_path=subset_path,
                            model_name=model_name,
                            lmbda=lmbda,
                            eps=eps,
                            embedding_matrix_path=state_path,
                        )
                        acc, asr, avg_t, pred_rows = _evaluate_antap(router, test_df)

                        summary = {
                            "dataset": dataset.name,
                            "train_csv": dataset.train_csv,
                            "test_csv": dataset.test_csv,
                            "train_rows_total": len(train_df),
                            "train_rows_used": n_train,
                            "train_fraction_used": frac_used,
                            "test_rows": len(test_df),
                            "embedding_model": model_name,
                            "lambda": lmbda,
                            "epsilon": eps,
                            "accuracy": acc,
                            "asr": asr,
                            "avg_route_time_sec": avg_t,
                            "embedding_state_path": state_path,
                        }
                        summary_rows.append(summary)

                        if args.save_predictions:
                            pred_name = f"pred_{run_hash}.csv"
                            pred_path = os.path.join(dirs["predictions"], pred_name)
                            pd.DataFrame(pred_rows).to_csv(pred_path, index=False)
                            summary["predictions_csv"] = pred_path

                        pd.DataFrame(summary_rows).to_csv(
                            os.path.join(dirs["root"], "summary.csv"),
                            index=False,
                        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(dirs["root"], "summary.csv")
    summary_df.to_csv(summary_path, index=False)

    if not summary_df.empty:
        best_df = summary_df.sort_values(
            by=["dataset", "accuracy"], ascending=[True, False]
        )
        best_path = os.path.join(dirs["root"], "best_by_dataset.csv")
        best_df.groupby("dataset", as_index=False).first().to_csv(
            best_path, index=False
        )
        print(f"Saved summary: {summary_path}")
        print(f"Saved best-per-dataset: {best_path}")
    else:
        print("No runs executed. Please check arguments.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ANTAP-only ablation studies")
    parser.add_argument(
        "--base-dir", type=str, default="base", help="Base data directory"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="mmlu,agentharm",
        help="Comma-separated: mmlu,agentharm",
    )

    parser.add_argument("--mmlu-train", type=str, default="")
    parser.add_argument("--mmlu-test", type=str, default="")
    parser.add_argument("--agentharm-train", type=str, default="")
    parser.add_argument("--agentharm-test", type=str, default="")

    parser.add_argument(
        "--embedding-models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated sentence-transformer model ids",
    )
    parser.add_argument(
        "--lambdas",
        type=str,
        default="0.01,0.1,1.0,10.0",
        help="Comma-separated lambda values",
    )
    parser.add_argument(
        "--epsilons",
        type=str,
        default="1e-3,1e-2,1e-1",
        help="Comma-separated epsilon values",
    )
    parser.add_argument(
        "--calibration-sizes",
        type=str,
        default="0.1,0.25,0.5,1.0",
        help="Comma-separated fractions (<=1) or absolute counts (>1)",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="ablation_antap")
    parser.add_argument("--save-predictions", action="store_true")
    return parser


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    run(cli_args)