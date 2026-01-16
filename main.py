from __future__ import print_function

import argparse
import os
import glob
import numpy as np
import pandas as pd
import hashlib

import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from dataloader import TCRRepertoireDataset
from model import FeatureAttention, FeatureGatedAttention


parser = argparse.ArgumentParser(description="Attention MIL for TCR repertoires")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--reg", type=float, default=1e-4, help="weight decay")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--no-cuda", action="store_true", default=False)

parser.add_argument("--model", type=str, default="attention", choices=["attention", "gated_attention"])
parser.add_argument("--dataset_name", type=str, default="train_dataset_1")
parser.add_argument("--encoding_type", type=str, default="kmer", choices=["kmer", "tcr_bert"])
parser.add_argument("--k", type=int, default=4)
parser.add_argument(
    "--bert_embeddings_base_dir",
    type=str,
    default="/backups/chihoim/adaptive_immune_challenge/features/train_trb3_tcrbert_avg/train_datasets",
)
parser.add_argument(
    "--bert_test_embeddings_base_dir",
    type=str,
    default="/backups/chihoim/adaptive_immune_challenge/features/test_trb3_tcrbert_avg/test_datasets/",
)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--preload", action="store_true", default=False)
parser.add_argument("--ckpt_path", type=str, default="best_model.pt")
parser.add_argument("--batch_size", type=int, default=1)

# Bag subsampling
parser.add_argument("--max_instances", type=int, default=None, help="subsample to at most this many instances per bag")
parser.add_argument("--sample_with_replacement", action="store_true", default=False)

# Evaluation control
parser.add_argument("--eval_full_val", action="store_true", default=False, help="evaluate on full val bags (no subsampling)")
parser.add_argument("--mc_samples", type=int, default=1, help="MC bag-resampling draws per val bag (>=1)")
parser.add_argument("--grad_clip", type=float, default=5.0)
parser.add_argument("--early_stopping_patience", type=int, default=5, help="stop if val AUC doesn't improve for this many epochs")

args = parser.parse_args()
args.cuda = (not args.no_cuda) and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print("\nGPU is ON!")

ckpt_dir = os.path.dirname(args.ckpt_path)
if ckpt_dir:
    os.makedirs(ckpt_dir, exist_ok=True)

loader_kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}

print("Load Train and Val Set")


def _read_val_repertoire_ids(dataset_name: str) -> list:
    val_ids_path = (
        f"/oak/stanford/groups/akundaje/abuen/kaggle/tcr-repertoire/log-reg/results/4mer-logreg/"
        f"split_indices/{dataset_name}_val_indices.txt"
    )
    if not os.path.isfile(val_ids_path):
        raise FileNotFoundError(f"Validation IDs file not found: {val_ids_path}")
    with open(val_ids_path, "r") as f:
        val_ids = [line.strip() for line in f.readlines() if line.strip()]
    if len(val_ids) == 0:
        raise ValueError(f"No validation IDs found in: {val_ids_path}")
    return val_ids


def _split_indices_from_val_ids(repertoire_filenames: list, val_ids: list, metadata_df: pd.DataFrame = None) -> tuple:
    # Prefer explicit repertoire_id column if present; otherwise fall back to filename stem.
    if metadata_df is not None and "repertoire_id" in metadata_df.columns:
        repertoire_ids = metadata_df["repertoire_id"].astype(str).tolist()
    else:
        repertoire_ids = [os.path.splitext(os.path.basename(fn))[0] for fn in repertoire_filenames]
    id_to_index = {rid: i for i, rid in enumerate(repertoire_ids)}

    missing = [rid for rid in val_ids if rid not in id_to_index]
    if missing:
        raise ValueError(
            "Some validation repertoire IDs were not found in the dataset metadata/filenames. "
            f"Missing (first 10): {missing[:10]}"
        )

    val_idx = sorted({id_to_index[rid] for rid in val_ids})
    train_idx = [i for i in range(len(repertoire_ids)) if i not in set(val_idx)]
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(f"Invalid split: train={len(train_idx)} val={len(val_idx)}")
    return train_idx, val_idx


def _load_sequences_from_tsv(tsv_path: str) -> list:
    df = pd.read_csv(tsv_path, sep="\t")
    if "junction_aa" not in df.columns:
        raise ValueError(f"Repertoire file missing junction_aa column: {tsv_path}")
    return df["junction_aa"].dropna().astype(str).tolist()


def _infer_bert_test_embeddings_base_dir() -> str:
    if args.bert_test_embeddings_base_dir:
        return args.bert_test_embeddings_base_dir
    base = str(args.bert_embeddings_base_dir)
    if "/train_datasets" in base:
        return base.replace("/train_datasets", "/test_datasets")
    return base


def _resolve_bert_embedding_path(bert_base_dir: str, dataset_name: str, repertoire_id: str) -> str:
    dataset_dir = os.path.join(bert_base_dir, dataset_name)
    candidate_np = os.path.join(dataset_dir, f"{repertoire_id}.np")
    candidate_npy = os.path.join(dataset_dir, f"{repertoire_id}.npy")
    if os.path.isfile(candidate_np):
        return candidate_np
    if os.path.isfile(candidate_npy):
        return candidate_npy
    raise FileNotFoundError(
        "Missing TCR-BERT embedding file. Looked for: "
        f"{candidate_np} or {candidate_npy}"
    )


def _stable_int_hash(text: str) -> int:
    # Stable across processes/runs (unlike Python's built-in hash())
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _subsample_embedding_rows(arr: np.ndarray, repertoire_id: str) -> np.ndarray:
    if args.max_instances is None or arr.shape[0] <= int(args.max_instances):
        return arr
    n = int(args.max_instances)
    seed = int(args.seed) + int(_stable_int_hash(repertoire_id))
    rng = np.random.default_rng(seed)
    K = int(arr.shape[0])
    if args.sample_with_replacement:
        idx = rng.integers(low=0, high=K, size=n, endpoint=False)
    else:
        idx = rng.choice(K, size=n, replace=False)
    return arr[idx]


def _load_bert_bag_for_test_repertoire(test_dataset_name: str, repertoire_id: str) -> torch.Tensor:
    bert_test_base = _infer_bert_test_embeddings_base_dir()
    emb_path = _resolve_bert_embedding_path(bert_test_base, test_dataset_name, repertoire_id)
    arr = np.load(emb_path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix in {emb_path}, got shape {arr.shape}")
    if ds_train.bert_hidden_dim is not None and int(arr.shape[1]) != int(ds_train.bert_hidden_dim):
        raise ValueError(
            f"Inconsistent embedding dim in {emb_path}: got {arr.shape[1]}, expected {ds_train.bert_hidden_dim}"
        )

    arr = np.asarray(arr, dtype=np.float32)
    arr = _subsample_embedding_rows(arr, repertoire_id=repertoire_id)
    bag = torch.from_numpy(arr)

    # Match dataset normalization behavior
    bag = bag / (bag.norm(dim=1, keepdim=True) + 1e-6)
    return bag


def _predict_repertoire_probability(model, *, sequences: list = None, test_dataset_name: str = None, repertoire_id: str = None) -> float:
    if args.encoding_type == "kmer":
        # Handle empty bags safely.
        if sequences is None or len(sequences) == 0:
            return 0.5
        bag = ds_train.kmer_encode_bag(sequences)  # KxD
        if bag.numel() == 0 or bag.shape[0] == 0:
            return 0.5
    elif args.encoding_type == "tcr_bert":
        if test_dataset_name is None or repertoire_id is None:
            raise ValueError("test_dataset_name and repertoire_id are required for tcr_bert inference")
        bag = _load_bert_bag_for_test_repertoire(test_dataset_name=test_dataset_name, repertoire_id=repertoire_id)
        if bag.numel() == 0 or bag.shape[0] == 0:
            return 0.5
    else:
        raise ValueError(f"Unknown encoding_type: {args.encoding_type}")

    if args.cuda:
        bag = bag.cuda(non_blocking=True)

    with torch.no_grad():
        logits, _, _ = model.forward(bag, lengths=None)
        probs = torch.sigmoid(logits).view(-1)
        return float(probs[0].detach().cpu().item())

# IMPORTANT: train and val must NOT share the same Dataset instance, because we need
# train sampling (random each epoch) and val sampling (deterministic or full).
ds_train = TCRRepertoireDataset(
    dataset_name=args.dataset_name,
    encoding_type=args.encoding_type,
    k=args.k,
    debug=args.debug,
    preload=args.preload,
    bert_embeddings_base_dir=args.bert_embeddings_base_dir,
    max_instances=args.max_instances,
    sample_with_replacement=args.sample_with_replacement,
    eval_full=False,
)
ds_val = TCRRepertoireDataset(
    dataset_name=args.dataset_name,
    encoding_type=args.encoding_type,
    k=args.k,
    debug=args.debug,
    preload=args.preload,
    bert_embeddings_base_dir=args.bert_embeddings_base_dir,
    max_instances=args.max_instances,
    sample_with_replacement=args.sample_with_replacement,
    eval_full=args.eval_full_val,
)

# Sanity: same ordering/labels
assert ds_train.repertoire_filenames == ds_val.repertoire_filenames
assert ds_train.labels == ds_val.labels

# Use the precomputed validation split from the log-reg run.
val_ids = _read_val_repertoire_ids(args.dataset_name)
train_idx, val_idx = _split_indices_from_val_ids(ds_train.repertoire_filenames, val_ids, getattr(ds_train, "_metadata_df", None))

ds_train.is_train = True
ds_val.is_train = False  # deterministic subsampling for stable AUC when mc_samples==1


train_ds = data_utils.Subset(ds_train, train_idx)
val_ds = data_utils.Subset(ds_val, val_idx)


def pad_collate(batch):
    """Collate for variable-length bags. Supports returning indices."""
    bags = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    indices = [item[2] for item in batch]

    bag_labels = torch.stack([lbl[0].view(-1) for lbl in labels], dim=0).view(-1, 1).float()
    instance_labels_list = [lbl[1].view(-1).float() for lbl in labels]

    max_k = max(b.shape[0] for b in bags)
    d = bags[0].shape[1]
    padded_bags = torch.zeros((len(bags), max_k, d), dtype=bags[0].dtype)
    padded_instance_labels = torch.zeros((len(bags), max_k), dtype=torch.float32)
    lengths = torch.tensor([b.shape[0] for b in bags], dtype=torch.long)

    for i, b in enumerate(bags):
        k_i = b.shape[0]
        padded_bags[i, :k_i, :] = b
        padded_instance_labels[i, :k_i] = instance_labels_list[i][:k_i]

    return padded_bags, [bag_labels, padded_instance_labels, lengths], torch.tensor(indices, dtype=torch.long)


collate_fn = pad_collate if args.batch_size > 1 else None

train_loader = data_utils.DataLoader(
    train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, **loader_kwargs
)
val_loader = data_utils.DataLoader(
    val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, **loader_kwargs
)

print("Init Model")
if args.encoding_type == "kmer":
    input_dim = len(ds_train.kmer_to_idx)
elif args.encoding_type == "tcr_bert":
    input_dim = int(ds_train.bert_hidden_dim)
else:
    raise ValueError("Unknown encoding_type")

if args.model == "attention":
    model = FeatureAttention(input_dim=input_dim)
else:
    model = FeatureGatedAttention(input_dim=input_dim)

if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
criterion = torch.nn.BCEWithLogitsLoss()
best_auc = float("-inf")
best_epoch = None
epochs_no_improve = 0


def save_best_checkpoint(epoch: int, auc: float) -> bool:
    global best_auc
    if auc > best_auc + 1e-6:
        best_auc = auc
        checkpoint = {
            "epoch": epoch,
            "best_auc": best_auc,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "dataset_name": args.dataset_name,
            "encoding_type": args.encoding_type,
            "k": args.k,
        }
        if args.encoding_type == "kmer":
            checkpoint["kmer_to_idx"] = ds_train.kmer_to_idx
        torch.save(checkpoint, args.ckpt_path)
        print(f"[epoch {epoch}] Saved new best checkpoint to {args.ckpt_path} (AUC={best_auc:.4f})")
        return True
    return False


def _unpack_batch(batch):
    """Support both collated (batch_size>1) and raw (batch_size=1) outputs."""
    if args.batch_size == 1:
        data, label, idx = batch
        lengths = None
    else:
        data, label, idx = batch
        lengths = label[2]
        label = label[:2]  # [bag_labels, instance_labels]
    bag_label = label[0]
    return data, bag_label, lengths, idx


def train_one_epoch(epoch: int):
    model.train()
    train_loss = 0.0
    train_error = 0.0

    for batch in tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}"):
        data, bag_label, lengths, _ = _unpack_batch(batch)

        if args.cuda:
            data = data.cuda(non_blocking=True)
            bag_label = bag_label.cuda(non_blocking=True)
            if lengths is not None:
                lengths = lengths.cuda(non_blocking=True)

        data = Variable(data)
        bag_label = Variable(bag_label.float().view(-1, 1))

        optimizer.zero_grad()
        logits, _, _ = model.forward(data, lengths=lengths)

        loss = criterion(logits, bag_label)
        train_loss += float(loss.item())

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        error = 1.0 - preds.eq(bag_label).float().mean().item()
        train_error += error

        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    train_loss /= len(train_loader)
    train_error /= len(train_loader)
    print(f"[epoch {epoch}] Train loss: {train_loss:.4f}, Train error: {train_error:.4f}")


def evaluate(epoch: int):
    """
    If mc_samples == 1:
      - deterministic subsampling (ds_val.is_train=False) OR full bags if eval_full_val
    If mc_samples > 1:
      - per-bag MC resampling by calling ds_val[absolute_index] repeatedly with ds_val.is_train=True
    """
    model.eval()
    val_loss = 0.0
    val_error = 0.0
    y_true, y_score = [], []

    # For MC, switch dataset to "train mode" so __getitem__ resamples each call.
    old_is_train = ds_val.is_train
    if args.mc_samples and args.mc_samples > 1:
        ds_val.is_train = True

    with torch.no_grad():
        for batch in val_loader:
            data, bag_label, lengths, idx = _unpack_batch(batch)

            if args.cuda:
                bag_label = bag_label.cuda(non_blocking=True)
            bag_label = bag_label.float().view(-1, 1)

            # Batch-size handling for MC:
            # - MC is implemented for batch_size=1 for simplicity/clarity.
            if args.mc_samples > 1 and args.batch_size != 1:
                raise ValueError("mc_samples>1 currently requires batch_size=1")

            if args.mc_samples <= 1:
                if args.cuda:
                    data = data.cuda(non_blocking=True)
                    if lengths is not None:
                        lengths = lengths.cuda(non_blocking=True)
                logits, _, _ = model.forward(data, lengths=lengths)
                loss = criterion(logits, bag_label)
                probs = torch.sigmoid(logits)

            else:
                # idx is absolute index in dataset
                abs_idx = int(idx.item())
                probs_accum = 0.0
                loss_accum = 0.0
                for _ in range(args.mc_samples):
                    bag_r, _, _ = ds_val[abs_idx]  # resampled bag
                    if args.cuda:
                        bag_r = bag_r.cuda(non_blocking=True).unsqueeze(0)
                    else:
                        bag_r = bag_r.unsqueeze(0)

                    logits_r, _, _ = model.forward(bag_r, lengths=None)
                    loss_accum += float(criterion(logits_r, bag_label).item())
                    probs_accum += torch.sigmoid(logits_r)

                probs = probs_accum / float(args.mc_samples)
                loss = torch.tensor(loss_accum / float(args.mc_samples))

            preds = (probs >= 0.5).float()
            error = 1.0 - preds.eq(bag_label).float().mean().item()

            val_loss += float(loss.item())
            val_error += error

            y_true.extend(bag_label.detach().view(-1).cpu().tolist())
            y_score.extend(probs.detach().view(-1).cpu().tolist())

    if args.mc_samples and args.mc_samples > 1:
        ds_val.is_train = old_is_train

    val_loss /= len(val_loader)
    val_error /= len(val_loader)
    auc = float(roc_auc_score(np.asarray(y_true), np.asarray(y_score)))

    print(f"[epoch {epoch}] Val loss: {val_loss:.4f}, Val error: {val_error:.4f}")
    print(f"[epoch {epoch}] Val AUC (bag-level): {auc:.4f}")
    return auc


if __name__ == "__main__":
    print("Start Training")

    for epoch in range(1, int(args.epochs) + 1):
        train_one_epoch(epoch)
        auc = evaluate(epoch)

        improved = save_best_checkpoint(epoch, auc)
        if improved:
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= int(args.early_stopping_patience):
            print(
                f"Early stopping at epoch {epoch}: "
                f"no val AUC improvement for {args.early_stopping_patience} epochs. "
                f"Best AUC={best_auc:.4f} at epoch {best_epoch}."
            )
            break

    print("Training Finished! Loading best checkpoint for test inference...")

    device = "cuda" if args.cuda else "cpu"
    best_ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # Map train datasets to their corresponding test datasets
    train_to_test_dataset_mapping = {
        "train_dataset_1": ["test_dataset_1"],
        "train_dataset_2": ["test_dataset_2"],
        "train_dataset_3": ["test_dataset_3"],
        "train_dataset_4": ["test_dataset_4"],
        "train_dataset_5": ["test_dataset_5"],
        "train_dataset_6": ["test_dataset_6"],
        "train_dataset_7": ["test_dataset_7_1", "test_dataset_7_2"],
        "train_dataset_8": ["test_dataset_8_1", "test_dataset_8_2", "test_dataset_8_3"],
    }

    test_base_dir = "/oak/stanford/groups/akundaje/abuen/kaggle/challenge_data/test_datasets/test_datasets"
    if args.dataset_name not in train_to_test_dataset_mapping:
        raise ValueError(f"Unknown dataset_name for test mapping: {args.dataset_name}")

    output_dir = ckpt_dir if ckpt_dir else "."
    for test_dataset_name in train_to_test_dataset_mapping[args.dataset_name]:
        test_dir = os.path.join(test_base_dir, test_dataset_name)
        if not os.path.isdir(test_dir):
            raise FileNotFoundError(f"Test dataset directory not found: {test_dir}")

        tsv_files = sorted(glob.glob(os.path.join(test_dir, "*.tsv")))
        if len(tsv_files) == 0:
            raise ValueError(f"No .tsv files found in test dataset directory: {test_dir}")

        preds = []
        for tsv_path in tqdm(tsv_files, desc=f"Infer {test_dataset_name}"):
            repertoire_id = os.path.splitext(os.path.basename(tsv_path))[0]
            if args.encoding_type == "kmer":
                seqs = _load_sequences_from_tsv(tsv_path)
                p = _predict_repertoire_probability(model, sequences=seqs)
            else:
                # For TCR-BERT, inference uses precomputed embeddings keyed by repertoire_id.
                p = _predict_repertoire_probability(model, test_dataset_name=test_dataset_name, repertoire_id=repertoire_id)
            preds.append({"repertoire_id": repertoire_id, "label_positive_probability": p})

        preds_df = pd.DataFrame(preds)
        out_path = os.path.join(output_dir, f"{args.dataset_name}_{test_dataset_name}_predictions.tsv")
        preds_df.to_csv(out_path, sep="\t", index=False)
        print(f"Wrote test predictions to: {out_path}")
