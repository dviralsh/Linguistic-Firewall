import os
from typing import List, Optional

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

ROUTER_IMPL_VERSION = "embedllm_closed_form_v1"


def _infer_agent_columns(df: pd.DataFrame) -> List[str]:
    metadata_cols = {
        "prompt",
        "prompt_id",
        "target",
        "Unnamed: 0",
        "has_trigger",
    }
    excluded_suffixes = (
        "_response",
        "_prediction",
        "_new_score",
        "_time_sec",
        "_is_malicious",
    )

    cols = []
    for c in df.columns:
        if c in metadata_cols:
            continue
        if c.endswith(excluded_suffixes):
            continue
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().sum() == 0:
            continue
        cols.append(c)

    if not cols:
        raise ValueError(
            "Could not infer agent columns from train CSV. Expected numeric columns like "
            "Strategist/Archivist or Strategist_score/Archivist_score."
        )
    return cols


def _build_agent_column_map(df: pd.DataFrame, inferred_cols: List[str]):
    """Return mapping: canonical_agent_name -> source_csv_column."""
    mapping = {}
    for c in inferred_cols:
        canonical = c[:-6] if c.endswith("_score") else c
        # Prefer exact (no suffix) if both exist
        if canonical not in mapping or (
            mapping[canonical].endswith("_score") and not c.endswith("_score")
        ):
            mapping[canonical] = c
    return mapping


class EmbedLLMRouter:
    """Minimal closed-form EmbedLLM-style router (no autograd training)."""

    def __init__(
        self,
        train_csv_path: str,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        embedding_dim: int = 232,
        model_state_path: Optional[str] = None,
        pretrained_model_embeddings_path: Optional[str] = None,
        model_order_csv: Optional[str] = None,
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.pretrained_model_embeddings_path = pretrained_model_embeddings_path
        self.model_order_csv = model_order_csv

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- [EmbedLLM Router] Using device: {self.device} ---")
        print(f"--- [EmbedLLM Router] Impl: {ROUTER_IMPL_VERSION} ---")
        self.encoder = SentenceTransformer(model_name, device=self.device)

        self.agent_names: List[str] = []
        self.agent_score_cols: List[str] = []
        self.agent_embeddings: Optional[torch.Tensor] = None
        self.proj_weight: Optional[torch.Tensor] = None

        if model_state_path and os.path.exists(model_state_path):
            self._load_state(model_state_path)
        else:
            self._fit(train_csv_path)
            if model_state_path:
                self._save_state(model_state_path)

    def _to_tensor(self, x_np) -> torch.Tensor:
        return torch.from_numpy(x_np).to(self.device).float()

    def _load_pretrained(self, target_names: List[str]) -> Optional[torch.Tensor]:
        if not self.pretrained_model_embeddings_path:
            return None

        loaded = torch.load(
            self.pretrained_model_embeddings_path, map_location=self.device
        )
        if isinstance(loaded, dict):
            for key in ("model_embeddings", "P", "embeddings", "weights"):
                if key in loaded and isinstance(loaded[key], torch.Tensor):
                    loaded = loaded[key]
                    break

        if not isinstance(loaded, torch.Tensor):
            raise ValueError("Unsupported pretrained embedding file structure.")

        pretrained = loaded.float().to(self.device)

        if not self.model_order_csv:
            if pretrained.shape[0] != len(target_names):
                raise ValueError(
                    "Pretrained row count mismatch and no model_order_csv provided."
                )
            return pretrained

        order_df = pd.read_csv(self.model_order_csv)
        if "model_name" not in order_df.columns:
            raise ValueError("model_order_csv must contain model_name.")

        ordered_names = order_df["model_name"].astype(str).tolist()
        if len(ordered_names) != pretrained.shape[0]:
            raise ValueError(
                "model_order_csv length mismatch with pretrained embeddings."
            )

        name_to_idx = {n: i for i, n in enumerate(ordered_names)}
        missing = [n for n in target_names if n not in name_to_idx]
        if missing:
            raise ValueError(f"Missing model names in model_order_csv: {missing[:5]}")

        return pretrained[[name_to_idx[n] for n in target_names]]

    def _fit(self, train_csv_path: str):
        if not os.path.exists(train_csv_path):
            raise FileNotFoundError(f"Train data not found: {train_csv_path}")

        df = pd.read_csv(train_csv_path)
        if "prompt" not in df.columns:
            raise ValueError("train.csv must contain 'prompt' column")

        inferred = _infer_agent_columns(df)
        col_map = _build_agent_column_map(df, inferred)
        self.agent_names = list(col_map.keys())
        self.agent_score_cols = [col_map[a] for a in self.agent_names]
        print(
            f"--- [EmbedLLM Router] Detected agents: {self.agent_names} "
            f"(from columns: {self.agent_score_cols}) ---"
        )

        prompts = df["prompt"].astype(str).tolist()
        q_np = self.encoder.encode(
            prompts,
            convert_to_numpy=True,
            device=self.device,
            show_progress_bar=True,
        ).astype("float32", copy=False)
        q = self._to_tensor(q_np)  # [num_prompts, d]

        y = torch.tensor(
            pd.DataFrame(
                {
                    agent: pd.to_numeric(df[src_col], errors="coerce").fillna(0.0)
                    for agent, src_col in zip(self.agent_names, self.agent_score_cols)
                }
            ).values.T,
            dtype=torch.float32,
            device=self.device,
        )  # [num_agents, num_prompts]

        pretrained = self._load_pretrained(self.agent_names)
        if pretrained is not None:
            b = pretrained
            rank = b.shape[1]
            q_t = q.t()
            w = torch.linalg.pinv(b) @ y @ torch.linalg.pinv(q_t)
            print(f"--- [EmbedLLM Router] Using pretrained embeddings (dim={rank}) ---")
        else:
            rank = min(self.embedding_dim, y.shape[0], y.shape[1])
            u, s, vh = torch.linalg.svd(y, full_matrices=False)
            u_k = u[:, :rank]
            s_k = s[:rank]
            vh_k = vh[:rank, :]
            sqrt_s = torch.sqrt(torch.clamp(s_k, min=1e-8))
            b = u_k * sqrt_s.unsqueeze(0)  # [num_agents, rank]
            z = sqrt_s.unsqueeze(1) * vh_k  # [rank, num_prompts]
            q_t = q.t()  # [d, num_prompts]
            w = z @ torch.linalg.pinv(q_t)  # [rank, d]
            print(f"--- [EmbedLLM Router] Learned SVD embeddings (rank={rank}) ---")

        self.agent_embeddings = b
        self.proj_weight = w

    def _save_state(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "router_type": "embedllm",
                "router_impl_version": ROUTER_IMPL_VERSION,
                "agent_names": self.agent_names,
                "agent_score_cols": self.agent_score_cols,
                "model_name": self.model_name,
                "embedding_dim": int(self.agent_embeddings.shape[1]),
                "proj_weight": self.proj_weight.detach().cpu(),
                "agent_embeddings": self.agent_embeddings.detach().cpu(),
            },
            path,
        )
        print(f"--- [EmbedLLM Router] Saved state to {path} ---")

    def _load_state(self, path: str):
        data = torch.load(path, map_location=self.device)
        if not isinstance(data, dict):
            raise ValueError(f"Unsupported state file: {path}")
        if data.get("router_type") not in (None, "embedllm"):
            raise ValueError(f"State at {path} is not embedllm router state.")

        self.agent_names = data["agent_names"]
        self.agent_score_cols = data.get("agent_score_cols", self.agent_names)
        self.proj_weight = data["proj_weight"].float().to(self.device)
        self.agent_embeddings = data["agent_embeddings"].float().to(self.device)
        print(f"--- [EmbedLLM Router] Loaded state from {path} ---")

    def select_speakers_batch(self, queries):
        """Batch-encode all queries at once and return selections for each."""
        if self.agent_embeddings is None or self.proj_weight is None:
            raise RuntimeError("EmbedLLM router is not initialized")
        q_np = self.encoder.encode(
            queries, convert_to_numpy=True, device=self.device, batch_size=256, show_progress_bar=False
        ).astype("float32", copy=False)  # (N, d)
        Q = self._to_tensor(q_np)  # (N, d)
        projected = (self.proj_weight @ Q.t())  # (proj_dim, N)
        scores = self.agent_embeddings @ projected  # (num_agents, N)
        best_indices = torch.argmax(scores, dim=0).tolist()
        return [self.agent_names[i] for i in best_indices]

    def select_speaker(self, query: str) -> str:
        if self.agent_embeddings is None or self.proj_weight is None:
            raise RuntimeError("EmbedLLM router is not initialized")

        q_np = self.encoder.encode(
            query, convert_to_numpy=True, device=self.device
        ).astype("float32", copy=False)
        q = self._to_tensor(q_np)  # [d]
        scores = self.agent_embeddings @ (self.proj_weight @ q)  # [num_agents]
        best_idx = int(torch.argmax(scores).item())
        return self.agent_names[best_idx]
