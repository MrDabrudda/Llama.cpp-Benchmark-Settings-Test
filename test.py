import subprocess
import os
import re
import time
from datetime import datetime

# --- CONFIGURATION ---
MODEL_PATH = "./models/deepseek-r1-14b.gguf"
LLAMA_BENCH_PATH = "./llama-bench"
OUTPUT_FILE = "high_vram_benchmarks.md"

# Cache types to test
CACHE_SETTINGS = [
    "--cache-type-k q8_0 --cache-type-v q8_0",
    "--cache-type-k q4_0 --cache-type-v q4_0",
    "--cache-type-k q4_1 --cache-type-v q4_1"
]

def get_gpu_specs():
    try:
        res = subprocess.check_output(["nvidia-smi", "--query-gpu=gpu_name,memory.total", "--format=csv,noheader,nounits"])
        name, total = res.decode().strip().split(',')
        return name.strip(), int(total)
    except:
        return "Unknown GPU", 8192

gpu_name, vram_total = get_gpu_specs()

# DYNAMIC CONTEXT SCALING
# We start with your base values and add more for high-VRAM cards
CONTEXT_VALS = [2048, 8192]
if vram_total >= 12000: CONTEXT_VALS.append(16384)
if vram_total >= 24000: CONTEXT_VALS.extend([32768, 65536])
if vram_total >= 32000: CONTEXT_VALS.append(131072) # 128k context!

def run_bench(cache, c_val):
    # Testing with -fa ON and -ngl at max (usually 45-50 for 14B models)
    cmd = f"{LLAMA_BENCH_PATH} -m {MODEL_PATH} -fa on {cache} -c {c_val} -ngl 99 -n 128"
    try:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=300) # Longer timeout for 128k
        match = re.search(r"eval rate\s+=\s+([\d.]+)\s+tokens/s", stdout + stderr)
        ts = match.group(1) if match else "FAIL (OOM)"
        
        vram_res = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        vram_used = vram_res.decode().strip()
        return ts, vram_used
    except:
        return "TIMEOUT", "N/A"

# --- EXECUTION ---
with open(OUTPUT_FILE, "w") as f:
    f.write(f"# High-VRAM Benchmark Results\nGPU: {gpu_name} ({vram_total}MB)\n\n")
    f.write("| Cache Type | Context Size | Speed (t/s) | VRAM Used (MB) |\n")
    f.write("|---|---|---|---|\n")

    for c_val in CONTEXT_VALS:
        for cache in CACHE_SETTINGS:
            print(f"Testing {c_val} context with {cache}...")
            ts, vram = run_bench(cache, c_val)
            f.write(f"| {cache} | {c_val} | {ts} | {vram} |\n")
            time.sleep(2)

print(f"Done! High-VRAM results saved to {OUTPUT_FILE}")
