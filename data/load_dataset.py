from datasets import load_dataset
from multiprocessing import Pool
from pathlib import Path
import json
import argparse


def write_jsonl_shard(index, dataset, output_dir):
    """Save one HuggingFace shard to a .jsonl file."""
    shard = dataset.shard(num_shards=args.num_shards, index=index, contiguous=True)
    out_path = output_dir / f"shard_{index:05d}.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for example in shard:
            json.dump(example, f, ensure_ascii=False)
            f.write("\n")

    return f"Saved {out_path}"


def main(args):
    source_path = Path(args.source_path)
    output_dir = source_path / f"fineweb_{args.name}_jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset '{args.name}' with {args.num_shards} shards...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=args.name,
        split="train",
        streaming=False,
    )

    print(f"Saving to {output_dir} using {args.num_workers} workers...")

    def worker(index):
        return write_jsonl_shard(index, dataset, output_dir)

    with Pool(args.num_workers) as pool:
        for result in pool.imap_unordered(worker, range(args.num_shards)):
            print(result)

    print("All shards saved as .jsonl.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shard and save FineWeb-Edu dataset as JSONL.")
    parser.add_argument("--source_path", type=str, required=True, help="Path to save the output JSONL shards")
    parser.add_argument("--name", type=str, default="sample-100BT", help="Subset name for fineweb-edu (e.g., sample-100BT)")
    parser.add_argument("--num_shards", type=int, default=997, help="Number of shards")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of multiprocessing workers")

    args = parser.parse_args()
    main(args)
