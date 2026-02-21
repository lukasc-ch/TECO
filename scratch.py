#%% imports
import json
from pathlib import Path

# load benchmark case
benchmark_case_idx = 0
benchmark_path = "TritonBench/data/TritonBench_G_v1.json"

with Path(benchmark_path).open() as f:
    benchmark_all = json.load(f)

print(f"Loaded {len(benchmark_all)} entries")
print("Keys:", list(benchmark_all[0].keys()))

bm = benchmark_all[benchmark_case_idx]

# %%

