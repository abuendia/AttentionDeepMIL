import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import pandas as pd
import torch
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split


class TCRRepertoireDataset(data_utils.Dataset):
    """
    TCR repertoire dataset as MIL bags.

    Each repertoire file corresponds to a bag. Instances are CDR3 amino-acid
    sequences loaded from the `junction_aa` column.

    Directory structure expected:
      <base_dir>/<dataset_name>/
        - metadata.csv  (columns: filename, label_positive)
        - <one .tsv per repertoire> (has column junction_aa)
    """

    def __init__(
        self,
        dataset_name: str,
        base_dir: str = "/oak/stanford/groups/akundaje/abuen/kaggle/challenge_data/train_datasets/train_datasets",
        bert_embeddings_base_dir: str = "/backups/chihoim/adaptive_immune_challenge/features/train_trb3_tcrbert_avg/train_datasets",
        preload: bool = False,
        encoding_type: str = "kmer",
        k: int = 4,
        max_instances: Optional[int] = None,
        sample_with_replacement: bool = False,
        eval_full: bool = False,
        debug: bool = False,
    ):
        self.dataset_name = dataset_name
        self.base_dir = base_dir
        self.dataset_dir = os.path.join(base_dir, dataset_name)
        self.bert_embeddings_base_dir = bert_embeddings_base_dir
        self.metadata_path = os.path.join(self.dataset_dir, "metadata.csv")
        self.preload = preload
        self.encoding_type = encoding_type
        self.debug = debug
        self.k = int(k)

        self.max_instances = max_instances
        self.sample_with_replacement = bool(sample_with_replacement)
        self.eval_full = bool(eval_full)

        # Sampling mode controls
        self.is_train = True
        self.base_seed = 12345

        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.encoding_type not in {"kmer", "tcr_bert"}:
            raise ValueError("encoding_type must be one of {'kmer', 'tcr_bert'}")

        if not os.path.isdir(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
        if not os.path.isfile(self.metadata_path):
            raise FileNotFoundError(f"metadata.csv not found: {self.metadata_path}")

        self._metadata_df = pd.read_csv(self.metadata_path)

        # Normalize labels to 0/1
        label_series = self._metadata_df["label_positive"]
        if label_series.dtype == bool:
            labels = label_series.astype(int)
        else:
            labels = (
                label_series.astype(str)
                .str.strip()
                .str.lower()
                .map({"true": 1, "false": 0, "1": 1, "0": 0})
            ).astype(int)

        self.repertoire_filenames: List[str] = self._metadata_df["filename"].astype(str).tolist()
        if self.debug:
            self.repertoire_filenames = self.repertoire_filenames[:10]
            labels = labels.iloc[:10]

        self.labels: List[int] = labels.tolist()
        self.repertoire_paths: List[str] = [os.path.join(self.dataset_dir, fn) for fn in self.repertoire_filenames]

        # Optionally preload raw sequences into memory
        self._bags_cache: Optional[List[List[str]]] = None
        if self.preload and self.encoding_type == "kmer":
            self._bags_cache = [
                self._load_bag_sequences(p) for p in tqdm(self.repertoire_paths, desc="Preloading bags")
            ]

        # TCR-BERT embeddings (one file per repertoire)
        self.bert_embedding_paths: Optional[List[str]] = None
        self.bert_hidden_dim: Optional[int] = None
        self._bert_cache: Optional[List[torch.Tensor]] = None
        if self.encoding_type == "tcr_bert":
            bert_dataset_dir = os.path.join(self.bert_embeddings_base_dir, dataset_name)
            if not os.path.isdir(bert_dataset_dir):
                raise FileNotFoundError(f"BERT embeddings directory not found: {bert_dataset_dir}")

            embedding_paths: List[str] = []
            for fn in self.repertoire_filenames:
                repertoire_id = os.path.splitext(os.path.basename(fn))[0]
                candidate_np = os.path.join(bert_dataset_dir, f"{repertoire_id}.np")
                candidate_npy = os.path.join(bert_dataset_dir, f"{repertoire_id}.npy")
                if os.path.isfile(candidate_np):
                    embedding_paths.append(candidate_np)
                elif os.path.isfile(candidate_npy):
                    embedding_paths.append(candidate_npy)
                else:
                    raise FileNotFoundError(
                        f"Missing BERT embedding file for repertoire '{fn}'. Looked for: {candidate_np} or {candidate_npy}"
                    )
            self.bert_embedding_paths = embedding_paths

            first = np.load(self.bert_embedding_paths[0])
            if first.ndim != 2:
                raise ValueError(f"Expected 2D embedding matrix, got shape {first.shape}")
            self.bert_hidden_dim = int(first.shape[1])

            if self.preload:
                self._bert_cache = [
                    self._load_bert_embeddings(p)
                    for p in tqdm(self.bert_embedding_paths, desc="Preloading BERT embeddings")
                ]

        # K-mer vocabulary (built once if requested)
        self.kmer_to_idx: Optional[Dict[str, int]] = None
        if self.encoding_type == "kmer":
            self.kmer_to_idx = self._build_kmer_vocab()

    def _sample_indices(self, K: int, index: int) -> torch.Tensor:
        """Return indices of length <= max_instances based on train/val mode."""
        if self.eval_full or self.max_instances is None or K <= self.max_instances:
            return torch.arange(K)

        n = int(self.max_instances)

        if self.is_train:
            # true randomness per call
            if self.sample_with_replacement:
                return torch.randint(low=0, high=K, size=(n,))
            return torch.randperm(K)[:n]

        # deterministic sampling for evaluation
        g = torch.Generator()
        g.manual_seed(int(self.base_seed + index))
        if self.sample_with_replacement:
            return torch.randint(low=0, high=K, size=(n,), generator=g)
        return torch.randperm(K, generator=g)[:n]

    def _subsample_bag(self, bag: torch.Tensor, index: int) -> torch.Tensor:
        idx = self._sample_indices(int(bag.shape[0]), index)
        return bag[idx]

    def _load_bag_sequences(self, repertoire_path: str) -> List[str]:
        df = pd.read_csv(repertoire_path, sep="\t")
        if "junction_aa" not in df.columns:
            raise ValueError(f"Repertoire file missing junction_aa column: {repertoire_path}")
        return df["junction_aa"].dropna().astype(str).tolist()

    def _load_bert_embeddings(self, embedding_path: str) -> torch.Tensor:
        arr = np.load(embedding_path)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D embedding matrix in {embedding_path}, got shape {arr.shape}")
        if self.bert_hidden_dim is not None and int(arr.shape[1]) != int(self.bert_hidden_dim):
            raise ValueError(
                f"Inconsistent embedding dim in {embedding_path}: got {arr.shape[1]}, expected {self.bert_hidden_dim}"
            )
        return torch.from_numpy(np.asarray(arr, dtype=np.float32))

    def _iter_kmers(self, sequence: str) -> List[str]:
        seq = str(sequence)
        if len(seq) < self.k:
            return []
        return [seq[i : i + self.k] for i in range(len(seq) - self.k + 1)]

    def _build_kmer_vocab(self) -> Dict[str, int]:
        kmers = set()

        if self._bags_cache is not None:
            bags_iter = self._bags_cache
        else:
            bags_iter = (self._load_bag_sequences(p) for p in self.repertoire_paths)

        for bag_sequences in bags_iter:
            for seq in bag_sequences:
                kmers.update(self._iter_kmers(seq))

        kmer_list = sorted(kmers)
        if len(kmer_list) == 0:
            raise ValueError(f"No {self.k}-mers found while building vocabulary for dataset {self.dataset_name}")
        return {kmer: idx for idx, kmer in enumerate(kmer_list)}

    def kmer_encode_sequence(self, sequence: str) -> torch.Tensor:
        if self.kmer_to_idx is None:
            raise RuntimeError("kmer_to_idx is not initialized; set encoding_type='kmer'")

        vec = torch.zeros(len(self.kmer_to_idx), dtype=torch.float32)
        for kmer in self._iter_kmers(sequence):
            idx = self.kmer_to_idx.get(kmer)
            if idx is not None:
                vec[idx] += 1.0
        return vec

    def kmer_encode_bag(self, bag_sequences: List[str]) -> torch.Tensor:
        if self.kmer_to_idx is None:
            raise RuntimeError("kmer_to_idx is not initialized; set encoding_type='kmer'")
        if len(bag_sequences) == 0:
            return torch.zeros((0, len(self.kmer_to_idx)), dtype=torch.float32)
        encoded = [self.kmer_encode_sequence(seq) for seq in bag_sequences]
        return torch.stack(encoded, dim=0)

    def __len__(self) -> int:
        return len(self.repertoire_paths)

    def __getitem__(self, index: int):
        if self.encoding_type == "kmer":
            if self._bags_cache is not None:
                bag_sequences = self._bags_cache[index]
            else:
                bag_sequences = self._load_bag_sequences(self.repertoire_paths[index])

            # Subsample sequences deterministically/randomly (same policy as embeddings)
            idx = self._sample_indices(len(bag_sequences), index).tolist()
            bag_sequences = [bag_sequences[i] for i in idx]

            bag = self.kmer_encode_bag(bag_sequences)
            num_instances = int(bag.shape[0])

        elif self.encoding_type == "tcr_bert":
            if self.bert_embedding_paths is None:
                raise RuntimeError("bert_embedding_paths not initialized")
            if self._bert_cache is not None:
                bag = self._bert_cache[index]
            else:
                bag = self._load_bert_embeddings(self.bert_embedding_paths[index])

            bag = self._subsample_bag(bag, index)

            # Normalize each instance embedding to unit norm (helps stability)
            bag = bag / (bag.norm(dim=1, keepdim=True) + 1e-6)
            num_instances = int(bag.shape[0])

        else:
            raise ValueError(f"Unknown encoding_type: {self.encoding_type}")

        bag_label = torch.tensor(float(self.labels[index]))
        instance_labels = torch.zeros(num_instances, dtype=torch.float32)

        # Return index so evaluation can resample the *same* bag for MC inference
        return bag, [bag_label, instance_labels], index

    def stratified_train_val_indices(
        self,
        val_size: float = 0.2,
        random_state: int = 1,
    ) -> Tuple[List[int], List[int]]:
        """Return (train_idx, val_idx) with stratification on bag labels."""
        if not (0.0 < val_size < 1.0):
            raise ValueError("val_size must be between 0 and 1")

        indices = np.arange(len(self))
        y = np.asarray(self.labels)
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_size,
            random_state=random_state,
            stratify=y,
        )
        return train_idx.tolist(), val_idx.tolist()
