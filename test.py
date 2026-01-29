import subprocess
import os
import re
import time
from datetime import datetime

# --- CONFIGURATION ---
MODEL_PATH = os.path.expanduser("~/models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf")
#LLAMA_BENCH_PATH = "./build/bin/llama-bench"
LLAMA_BENCH_PATH = os.path.expanduser("~/llama.cpp/build/bin/llama-bench")
OUTPUT_FILE = "gpu_benchmark.md"

# 1. MODEL REGISTRY
MODEL_REGISTRY = {
    "0.5b": (24, range(20, 26)), "0.6b": (24, range(20, 26)),
    "1.2b": (28, range(24, 30)), "1.5b": (28, range(24, 30)),
    "3b":   (36, range(32, 38)), "7b":   (32, range(28, 34)),
    "8b":   (32, range(30, 35)), "12b":  (40, range(38, 42)),
    "14b":  (48, range(15, 46)), 
    "20b":  (44, range(40, 46)), "32b":  (64, range(60, 66)),
    "70b":  (80, range(78, 82)), "405b": (126, range(120, 128)),
    "671b": (61, range(58, 63))
}

# 2. CACHE SETTINGS
CACHE_SETTINGS = [
    "-ctk q8_0 -ctv q8_0",
#    "-ctk q5_1 -ctv q5_1",
#    "-ctk q5_0 -ctv q5_0",
#    "-ctk q4_1 -ctv q4_1",
#    "-ctk q4_0 -ctv q4_0",
#    "-ctk iq4_nl -ctv iq4_nl"
]

def get_gpu_info():
    """Fetches GPU Make/Model and Total VRAM from nvidia-smi."""
    try:
        # Fetches the Model Name (e.g., NVIDIA GeForce GTX 1070)
        name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
            text=True
        ).strip()
        # Fetches Total VRAM in MB
        vram = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        return name, vram
    except Exception:
        return "Unknown NVIDIA GPU", "8192"

gpu_name, gpu_total_vram = get_gpu_info()

# 3. CONTEXT VALS
#CONTEXT_VALS = [512, 1024, 2048, 4096, 8192]
CONTEXT_VALS = [512]

def get_ngl_range(path):
    fn = path.lower()
    for key, (total, r) in MODEL_REGISTRY.items():
        if key in fn: return r
    return range(32, 36)

NGL_TESTS = get_ngl_range(MODEL_PATH)

def run_bench(cache_flags, context, ngl):
    """Executes llama-bench and parses output."""
    try:
        # Construct the command
        # split() is used for cache_flags because it may contain multiple arguments
        cmd = [
            LLAMA_BENCH_PATH,
            "-m", MODEL_PATH,
            "-p", str(context),
            "-n", "128", # Test 128 tokens for TG
            "-ngl", str(ngl)
        ] + cache_flags.split()

        result = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        
        # Regex to find throughput values in the llama-bench markdown-style table
        # Look for the row containing our model name
        pp_match = re.search(r"\|\s+pp\d+\s+\|\s+[\d.]+\s+\|\s+([\d.]+)", result)
        tg_match = re.search(r"\|\s+tg\d+\s+\|\s+[\d.]+\s+\|\s+([\d.]+)", result)
        
        # Simple VRAM check: Get current usage during/after run (Estimation)
        vram_usage = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True
        ).strip()

        pp = pp_match.group(1) if pp_match else "N/A"
        tg = tg_match.group(1) if tg_match else "N/A"
        
        return pp, tg, vram_usage

    except Exception as e:
        print(f"Error running bench: {e}")
        return "ERR", "ERR", "N/A"

# --- EXECUTION ---
with open(OUTPUT_FILE, "w") as f:
    # UPDATED HEADER WITH GPU MODEL
    f.write(f"# Llama.cpp GPU Settings Benchmark\n")
    f.write(f"**GPU:** {gpu_name} ({gpu_total_vram} MB Total)\n")
    f.write(f"**Model:** {os.path.basename(MODEL_PATH)}\n")
    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("| Cache Type | Context | NGL | PP (t/s) | TG (t/s) | VRAM Used (MB) |\n")
    f.write("|---|---|---|---|---|---|\n")
    
    for c in CONTEXT_VALS:
        for cache in CACHE_SETTINGS:
            for ngl in NGL_TESTS:
                print(f"Testing: {cache} | C:{c} | NGL:{ngl}")
                pp, tg, vram = run_bench(cache, c, ngl)
                f.write(f"| {cache} | {c} | {ngl} | {pp} | {tg} | {vram} |\n")
                f.flush()

print(f"\n✅ Benchmark Complete. Results recorded for {gpu_name} in {OUTPUT_FILE}")
