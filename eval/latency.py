import sys
from pathlib import Path
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
import time
import csv
from pathlib import Path
from model.auto import register_models
from transformers import AutoModelForCausalLM, AutoConfig

register_models()

# Seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Device setup
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

# Load model
config_path = "config_path.json"
model_name = Path(config_path).stem  # e.g., 'swa_scout_470M'
print(model_name)

config = AutoConfig.from_pretrained(config_path, torch_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_config(config)
model = model.to(dtype=torch.bfloat16, device=device)
model.eval()

# Benchmark parameters
batch_sizes = [16] 
input_seq_lens = [1]
max_new_tokens_list = [2048,4096,8192,16384,32768]
vocab_size = 32000

# Prepare CSV
csv_filename = f"{model_name}.csv"
csv_header = ["model_name", "batch_size", "input_seq_len", "generate_seq_len", "tokens_per_second", "peak_memory_MB"]

with open(csv_filename, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_header)

    for batch_size in batch_sizes:
        for input_seq_len in input_seq_lens:
            # Prepare consistent random input for this combination
            batch_input = torch.randint(
                low=0, high=vocab_size, 
                size=(batch_size, input_seq_len),
                dtype=torch.long,
                device=device
            )

            for max_new_tokens in max_new_tokens_list:
                print(f"\n[Running] batch_size={batch_size}, input_seq_len={input_seq_len}, max_new_tokens={max_new_tokens}")

                torch.cuda.reset_peak_memory_stats(device)
                tokens_per_second = 0
                peak_memory_MB = 0

                try:
                    with torch.no_grad():
                        start_time = time.time()
                        output = model.generate(
                            batch_input,
                            max_new_tokens=max_new_tokens,
                            use_cache=True,
                            do_sample=False
                        )
                        end_time = time.time()

                    elapsed = end_time - start_time
                    generated_tokens = output.shape[1] - input_seq_len
                    total_generated_tokens = generated_tokens * batch_size

                    tokens_per_second = total_generated_tokens / elapsed
                    peak_memory_MB = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

                    print(f"  Generated tokens per sample: {generated_tokens}")
                    print(f"  Total tokens: {total_generated_tokens}")
                    print(f"  Elapsed time: {elapsed:.2f} sec")
                    print(f"  Tokens/sec: {tokens_per_second:.2f}")
                    print(f"  Peak memory: {peak_memory_MB:.2f} MB")

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f" OOM occurred. Marking tokens/sec and memory as 0.")
                        torch.cuda.empty_cache()
                    else:
                        raise e  # re-raise if another error

                # Write to CSV
                writer.writerow([
                    model_name,
                    batch_size,
                    input_seq_len,
                    max_new_tokens,
                    f"{tokens_per_second:.2f}",
                    f"{peak_memory_MB:.2f}"
                ])

print(f"\n Benchmark completed and saved to {csv_filename}")
