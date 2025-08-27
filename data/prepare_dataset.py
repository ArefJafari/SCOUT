
# Copyright Lightning AI. Licensed under the Apache License 2.0
# MIT License: Microsoft / Lightning AI
# Adapted for FineWeb 100B processing

import glob
import json
import os
import sys
import numpy as np
from tqdm import tqdm


from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from data.packed_dataset import PackedDataset as packed_dataset
from transformers import AutoTokenizer

def prepare_sample(
    source_path: Path,
    tokenizer_path: Path,
    destination_path: Path,
    chunk_size: int,
    match: str = ""
) -> None:
    """Tokenize and pack .jsonl files from FineWeb-style structure."""
    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = True
    all_jsonl_files = sorted(source_path.rglob("*.jsonl"))

    if not all_jsonl_files:
        raise RuntimeError(f"No .jsonl files found under {source_path}")

    for filepath in tqdm(all_jsonl_files, desc="Processing files"):
        if match and match not in str(filepath):
            continue

        prefix = filepath.stem  # e.g., 'CC-MAIN-2023-06_chunk_00'

        builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=prefix,
            chunk_size=chunk_size,
            sep_token=2,
            dtype="auto",
            vocab_size=tokenizer.vocab_size,
        )

        print(f" Processing {filepath}")
        lines = [json.loads(r)['text'] for r in open(filepath)]
        with open(filepath, encoding="utf-8") as f:
            for text_ids in tqdm(tokenizer(lines, return_attention_mask=False)['input_ids'], desc=f"Tokenizing {prefix}"):
                try:
                    builder.add_array(np.array(text_ids, dtype=builder.dtype))
                except Exception as e:
                    print(f"Skipping line due to error: {e}")
                    continue

        builder.write_reminder()


def prepare(
    source_path: Path = Path("./datasets/finweb-edu/100B/fineweb_sample-100BT_jsonl"),
    tokenizer_path: Path = Path("./tokenizers/transformer-1.3B-100B"),
    destination_path: Path = Path("./datasets/fineweb-edu/100B/fla_tokenized"),
    sample: bool = True,
    match: str = "",
) -> None:
    """Entrypoint: tokenizes and packs raw JSONL files."""
    block_size = 4096

    prepare_sample(
        source_path=source_path,
        tokenizer_path=tokenizer_path,
        destination_path=destination_path,
        chunk_size=(block_size + 1) * 1024,  # block size + 1 for causal, 1024 blocks
        match=match,
    )


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(prepare)
