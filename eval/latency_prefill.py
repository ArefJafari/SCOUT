#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure repo root on sys.path
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import time
import csv
import torch
from model.auto import register_models
from transformers import AutoModelForCausalLM, AutoConfig

# ---------------------------------------------------------------------
# Register custom models if available
try:
    register_models()
except Exception:
    pass
# ---------------------------------------------------------------------

# Seeds
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Device / dtype
if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    # bf16 on Ampere+; else fp16
    major, _ = torch.cuda.get_device_capability()
    dtype = torch.bfloat16 if major >= 8 else torch.float16
    torch.cuda.set_device(0)
    # Allow TF32 matmul for speed (no accuracy change for bf16/fp16)
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    dev = torch.device("cpu")
    dtype = torch.float32

# --------- MODEL LOADING (load from directory, not .bin path) ----------
# If you pass a file (pytorch_model.bin), use its parent directory
model_path = "/work/marzieh/512x1k_15B_SWA_SCOUT_MLP_k10_0.34B_July4/final-model-ckpt.pth/pytorch_model.bin"
model_dir = Path(model_path)
if model_dir.is_file():
    model_dir = model_dir.parent
model_dir = str(model_dir)

# A readable model name for logs/CSV
model_name = Path(model_dir).name
print(f"Model dir: {model_dir}")
print(f"Model name: {model_name}")

# Load config + model
config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    config=config,
    trust_remote_code=True,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
)
model = model.to(dev)
model.eval()

# Ensure pad/eos are set for generate()
if getattr(model.config, "pad_token_id", None) is None and getattr(model.config, "eos_token_id", None) is not None:
    model.config.pad_token_id = model.config.eos_token_id
if hasattr(model, "generation_config"):
    if getattr(model.generation_config, "pad_token_id", None) is None:
        model.generation_config.pad_token_id = model.config.pad_token_id
    if getattr(model.generation_config, "eos_token_id", None) is None:
        model.generation_config.eos_token_id = model.config.eos_token_id

print(model)

# ------------------ BENCHMARK PARAMS ------------------
batch_sizes = [1, 2, 4]
input_seq_lens = [2048, 4096, 8192, 16384]
max_new_tokens_list = [10]
vocab_size = 32000

# ------------------ CSV SETUP ------------------
csv_filename = f"{model_name}_triton2.csv"
csv_header = [
    "model_name",
    "batch_size",
    "input_seq_len",
    "generate_seq_len",
    "tokens_per_second",
    "peak_memory_MB",
]

def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

with open(csv_filename, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_header)

    for batch_size in batch_sizes:
        for input_seq_len in input_seq_lens:
            # Make random input_ids on the correct device
            batch_input = torch.randint(
                low=0,
                high=vocab_size,
                size=(batch_size, input_seq_len),
                dtype=torch.long,
                device=dev,
            )

            for max_new_tokens in max_new_tokens_list:
                print(f"\n[Running] B={batch_size}, T={input_seq_len}, new={max_new_tokens}")

                # Warmup
                try:
                    with torch.no_grad():
                        _ = model.generate(
                            input_ids=batch_input,
                            max_new_tokens=max_new_tokens,
                            use_cache=True,
                            do_sample=False,
                            pad_token_id=model.config.pad_token_id,
                            eos_token_id=model.config.eos_token_id,
                        )
                    sync()
                except RuntimeError as e:
                    msg = str(e).lower()
                    if "out of memory" in msg or "cuda error" in msg:
                        print("  OOM in warmup. Skipping.")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        writer.writerow([model_name, batch_size, input_seq_len, max_new_tokens, "0.00", "0.00"])
                        continue
                    else:
                        raise

                # Timed run
                reset_peak()
                tokens_per_second = 0.0
                peak_memory_MB = 0.0
                try:
                    with torch.no_grad():
                        t0 = time.time()
                        out = model.generate(
                            input_ids=batch_input,
                            max_new_tokens=max_new_tokens,
                            use_cache=True,
                            do_sample=False,
                            pad_token_id=model.config.pad_token_id,
                            eos_token_id=model.config.eos_token_id,
                        )
                        sync()
                        t1 = time.time()

                    elapsed = max(1e-9, t1 - t0)
                    # out shape is [B, input_len + generated_len]
                    generated_tokens = out.shape[1] - input_seq_len
                    total_gen = generated_tokens * batch_size
                    tokens_per_second = total_gen / elapsed

                    if torch.cuda.is_available():
                        peak_memory_MB = torch.cuda.max_memory_allocated() / (1024 ** 2)
                    else:
                        peak_memory_MB = 0.0

                    print(f"  gen_tokens/sample: {generated_tokens}")
                    print(f"  total_gen:         {total_gen}")
                    print(f"  elapsed:           {elapsed:.3f} s")
                    print(f"  tokens/sec:        {tokens_per_second:.2f}")
                    print(f"  peak memory:       {peak_memory_MB:.2f} MB")

                except RuntimeError as e:
                    msg = str(e).lower()
                    if "out of memory" in msg or "cuda error" in msg:
                        print("  OOM in generate. Marking as 0.")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        tokens_per_second = 0.0
                        peak_memory_MB = 0.0
                    else:
                        raise

                writer.writerow([
                    model_name,
                    batch_size,
                    input_seq_len,
                    max_new_tokens,
                    f"{tokens_per_second:.2f}",
                    f"{peak_memory_MB:.2f}",
                ])

print(f"\nBenchmark completed and saved to {csv_filename}")
