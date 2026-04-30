import torch
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Tuple


def tikhonov_regularized_pseudoinverse(
    A: torch.Tensor, lmbda: float = 1.0, eps: float = 1e-1
) -> torch.Tensor:
    """
    Compute a numerically-stable Tikhonov-regularized pseudoinverse of A.
    """
    # Use SVD for stability
    U, S, Vh = torch.svd(A)
    # threshold small singular values
    S_threshold = S.max() * torch.finfo(S.dtype).eps * A.shape[0]
    S = torch.where(S > S_threshold, S, torch.zeros_like(S))
    # further threshold with eps
    S = torch.where(S > eps, S, torch.zeros_like(S))
    S2 = S**2
    D = S / (S2 + 2.0 * lmbda)
    Dmat = torch.diag(D)
    A_pinv = Vh @ Dmat @ U.t()
    return A_pinv


class SemanticEmbeddingRouter:
    """
    Routes queries based on a trained Embedding matrix (E_M) derived from benchmark data.
    It predicts which agent is most likely to succeed (return 1) vs fail (return 0/-1).
    """

    def __init__(
        self,
        train_csv_path: str,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        lmbda: float = 1.0,
        eps: float = 0.1,
        embedding_matrix_path: Optional[str] = None,
    ):

        self.model_name = model_name

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print(f"--- [Router] Using device: {self.device} ---")

        self.model = SentenceTransformer(model_name, device=self.device)
        self.lmbda = lmbda
        self.eps = eps

        # Load or Train E_M Matrix
        if embedding_matrix_path and os.path.exists(embedding_matrix_path):
            print(
                f"--- [Router] Loading Embedding Matrix from {embedding_matrix_path} ---"
            )
            data = torch.load(embedding_matrix_path, map_location=self.device)
            self.agent_names = data["agent_names"]
            self.E_M = data["E_M"]
        else:
            self.agent_names, self.E_M = self._train_embedding_matrix(train_csv_path)
            if embedding_matrix_path:
                print(
                    f"--- [Router] Saving Embedding Matrix to {embedding_matrix_path} ---"
                )
                torch.save(
                    {"agent_names": self.agent_names, "E_M": self.E_M},
                    embedding_matrix_path,
                )

    def _train_embedding_matrix(
        self, train_csv_path: str
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Calculates the E_M matrix using the logic from run_pi().
        E_M = train_perf @ A_pinv
        """
        if not os.path.exists(train_csv_path):
            raise FileNotFoundError(
                f"Benchmark data {train_csv_path} not found. Cannot initialize Performance Router."
            )

        print(f"--- [Router] Training Embedding matrix from {train_csv_path} ---")

        # 1. Load Data
        df = pd.read_csv(train_csv_path)

        if "prompt" not in df.columns:
            raise ValueError(
                "train.csv must contain a 'prompt' column to generate embeddings."
            )

        metadata_cols = {"prompt", "prompt_id", "Unnamed: 0", "has_trigger"}
        excluded_suffixes = (
            "_response",
            "_prediction",
            "_new_score",
            "_time_sec",
            "_is_malicious",
        )

        source_cols = []
        for c in df.columns:
            if c in metadata_cols:
                continue
            if c.endswith(excluded_suffixes):
                continue
            numeric = pd.to_numeric(df[c], errors="coerce")
            if numeric.notna().sum() == 0:
                continue
            source_cols.append(c)

        if not source_cols:
            raise ValueError("No numeric agent columns found in benchmark train CSV.")

        agent_names = [c[:-6] if c.endswith("_score") else c for c in source_cols]

        print(f"--- [Router] Detected Agents in Benchmark: {agent_names}")

        # 2. Generate Embeddings for Training Prompts
        prompts = df["prompt"].tolist()
        print(f"--- [Router] Encoding {len(prompts)} training prompts... ---")
        train_embeddings = self.model.encode(
            prompts, convert_to_tensor=True, device=self.device
        )

        # 3. Prepare Performance Matrix (train_perf)
        train_perf = torch.tensor(
            pd.DataFrame(
                {
                    c: pd.to_numeric(df[c], errors="coerce").fillna(0.0)
                    for c in source_cols
                }
            ).T.values,
            device=self.device,
            dtype=torch.float32,
        )

        # Apply the run_pi_experiment logic: 0 -> -1 transformation
        train_perf[train_perf == 0.0] = -1.0

        # 4. Compute Pseudoinverse (A_pinv)
        A = train_embeddings.t().to(self.device)
        A_pinv = tikhonov_regularized_pseudoinverse(A, lmbda=self.lmbda, eps=self.eps)

        # 5. Compute E_M
        E_M = train_perf @ A_pinv

        print("--- [Router] Embedding matrix (E_M) calculated successfully. ---")
        return agent_names, E_M

    def select_speakers_batch(self, queries: List[str]) -> List[str]:
        """Batch-encode all queries at once and return selections for each."""
        embeddings = self.model.encode(
            queries, convert_to_tensor=True, device=self.device, batch_size=256, show_progress_bar=False
        )  # (N, dim)
        scores = self.E_M @ embeddings.t()  # (num_agents, N)
        best_indices = torch.argmax(scores, dim=0).tolist()
        return [self.agent_names[i] for i in best_indices]

    def select_speaker(self, query: str) -> str:
        """
        Predicts success scores for all agents and returns the best one.
        """
        # 1. Encode Query
        query_embedding = self.model.encode(
            query, convert_to_tensor=True, device=self.device
        )

        # 2. Predict Scores: s = E_M @ q
        # E_M is (num_agents, dim), query is (dim,)
        scores = self.E_M @ query_embedding

        # 3. Find Best Agent
        best_idx = torch.argmax(scores).item()
        best_agent = self.agent_names[best_idx]

        return best_agent