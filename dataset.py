# dataset.py
import os
from glob import glob
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from tokenizers import ByteLevelBPETokenizer


def train_or_load_tokenizer(
    data_dir: str,
    vocab_size: int = 50257,
    min_frequency: int = 2,
    tokenizer_dir: str = "./tokenizer",
):
    os.makedirs(tokenizer_dir, exist_ok=True)
    vocab_file = os.path.join(tokenizer_dir, "vocab.json")
    merges_file = os.path.join(tokenizer_dir, "merges.txt")

    if os.path.exists(vocab_file) and os.path.exists(merges_file):
        tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
        return tokenizer

    paths = glob(os.path.join(data_dir, "**/*.txt"), recursive=True)
    if not paths:
        raise ValueError(f"No .txt files found in {data_dir}")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=paths,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )

    tokenizer.save_model(tokenizer_dir)
    return tokenizer


class GPT2TextDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer: ByteLevelBPETokenizer,
        block_size: int = 1024,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.block_size = block_size

        paths = glob(os.path.join(data_dir, "**/*.txt"), recursive=True)
        if not paths:
            raise ValueError(f"No .txt files found in {data_dir}")

        # Concatenate everything into one big sequence of ids
        all_ids: List[int] = []
        for path in paths:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            enc = tokenizer.encode(text)
            all_ids.extend(enc.ids)
            # Add a separator token if you like:
            # all_ids.append(tokenizer.token_to_id("</s>"))

        # Chunk into blocks of block_size
        # Drop the last incomplete block
        n_blocks = len(all_ids) // block_size
        all_ids = all_ids[: n_blocks * block_size]
        self.data = torch.tensor(all_ids, dtype=torch.long)
        self.n_blocks = n_blocks

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        x = self.data[start:end]
        # For LM, labels are the same as input_ids
        return {
            "input_ids": x.clone(),
            "labels": x.clone(),
        }