import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
import sys, gc

from pathlib import Path
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import random
import numpy as np
from pathlib import Path
from model.auto import register_models
from transformers import AutoModelForCausalLM

# Set seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device) 

# Load model and tokenizer
tokenizer_id = "fla-hub/transformer-1.3B-100B"
model_path = "model_path"
model_name = "model_name"

register_models()
print(f"Loading model from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16).to(device).eval()



# Datasets
datasets_info = {
    "pg19": ("fla-hub/pg19", None, "test", "text"),
    "narrativeqa": ("narrativeqa", None, "validation", "document.text"),
    "govreport": ("ccdv/govreport-summarization", None, "test", "report"),
    "qasper": ("allenai/qasper", None, "validation", "full_text.paragraphs"),
    "codeparrot": ("macrocosm-os/code-parrot-github-code", None, "train", "content"),
    "booksum": ("kmfoda/booksum", None, "test", "chapter")
}

# Evaluation setup
eval_lengths = [512, 1024, 2048, 4096, 8192, 12288, 16384]
num_samples = 5
final_results = {}

for name, (ds_id, config, split, field) in datasets_info.items():
    print(f"\n Evaluating {name} ...")
    try:
        ds = load_dataset(ds_id, config, split=split, streaming=True)
    except Exception as e:
        print(f" Failed to load {name}: {e}")
        continue

    long_samples = []
    for sample in ds:
        # Handle nested fields like "document.text"
        keys = field.split(".")
        value = sample
        for key in keys:
                value = value[key]

        # If it's a nested list of paragraphs:
        if name == "qasper":
            text = "\n".join(par for section in value for par in section)
        else:
            text = value
        input_ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(input_ids) >= max(eval_lengths):
            long_samples.append(input_ids)
        if len(long_samples) >= num_samples:
            break

    if len(long_samples) < num_samples:
        print(f" Only found {len(long_samples)} valid samples in {name}, skipping.")
        continue

    results = {}
    for length in eval_lengths:
        ppl_list = []
        for ids in long_samples:
            if len(ids) < length:
                continue
            input_tensor = torch.tensor(ids[:length]).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor, labels=input_tensor)
                loss = outputs.loss

            ppl = torch.exp(loss).item()
            ppl_list.append(ppl)

            del input_tensor, outputs
            torch.cuda.empty_cache()
            gc.collect()

        if ppl_list:
            avg_ppl = sum(ppl_list) / len(ppl_list)
            results[length] = round(avg_ppl, 2)
            print(f" {name} | Length {length} → Avg PPL: {avg_ppl:.2f}")
        else:
            print(f" {name} | Length {length} → No valid samples")

    final_results[name] = results
    torch.cuda.empty_cache()
    gc.collect()

# Final results table
print("\n Final Results Table (Avg PPL over " + str(num_samples) + " Samples):\n")
header = "Dataset".ljust(15) + "".join([f"{l:>8}" for l in eval_lengths])
print(header)
print("-" * len(header))
for ds, res in final_results.items():
    row = ds.ljust(15) + "".join([f"{res.get(l, '-'):>8}" for l in eval_lengths])
    print(row)

import pandas as pd
from pathlib import Path

# Convert results to DataFrame
df = pd.DataFrame(final_results).T[eval_lengths]
df.index.name = "Dataset"
df.insert(0, "Model", model_name)

# Output directory
out_dir = Path("ppl_results")
out_dir.mkdir(parents=True, exist_ok=True)

# Save to CSV
csv_path = out_dir / f"{model_name}.csv"
df.to_csv(csv_path)

