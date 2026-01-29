import subprocess
import os
import re
import time
from datetime import datetime

# --- CONFIGURATION ---
MODEL_PATH = "./models/deepseek-r1-14b.gguf"
LLAMA_BENCH_PATH = "./llama-bench"
OUTPUT_FILE = "comprehensive_gpu_benchmarks.md"

# Comprehensive Cache Types
CACHE_SETTINGS = [
    "--cache-type-k q8_0 --cache-type-v q8_0",
    "--cache-type-k q5_1 --cache-type-v q5_1",
    "--cache-type-k q5_0 --cache-type-v q5_0",
    "--cache-type-k q4_1 --cache-type-v q4_1",
    "--cache-type-k q4_0 --cache-type-v q4_0",
    "--cache-type-k iq4_nl --cache-type-v iq4_nl"
]

def get_gpu_specs():
    try:
        res = subprocess.check_output(["nvidia-smi", "--query-gpu=gpu_name,memory.total", "--format=csv,noheader,nounits"])
        name, total = res.decode().strip().split(',')
        return name.strip(), int(total)
    except:
        return "Unknown GPU", 8192

def clear_vram():
    """Nukes port 8080 to ensure no background llama-server is eating VRAM."""
    try:
        subprocess.run(["sudo", "fuser", "-k", "8080/tcp"], capture_output=True)
        time.sleep(2) # Brief pause for driver to reclaim memory
    except:
        pass

gpu_name, vram_total = get_gpu_specs()

# DYNAMIC CONTEXT SCALING (Expanded for high VRAM)
CONTEXT_VALS = [512, 1024, 2048, 4096, 8192]
if vram_total >= 12000: CONTEXT_VALS.append(16384)
if vram_total >= 24000: CONTEXT_VALS.extend([32768, 65536])
if vram_total >= 40000: CONTEXT_VALS.append(128000)

# Layers to test (adjust lower for 8GB cards if still failing)
NGL_RANGE = range(35, 46) 

def run_bench(cache, c_val, ngl):
    clear_vram()
    # Using -fa on by default as requested
    cmd = f"{LLAMA_BENCH_PATH} -m {MODEL_PATH} -fa on {cache} -c {c_val} -ngl {ngl} -n 128"
    try:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=180)
        match = re.search(r"eval rate\s+=\s+([\d.]+)\s+tokens/s", stdout + stderr)
        ts = match.group(1) if match else "FAIL"
        
        vram_res = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        vram_used = vram_res.decode().strip()
        return ts, vram_used
    except:
        return "TIMEOUT", "N/A"

# --- EXECUTION ---
with open(OUTPUT_FILE, "w") as f:
    f.write(f"# Unified AI Benchmark Results\nGPU: {gpu_name} ({vram_total}MB)\n\n")
    f.write("| Cache Type | Context | NGL | Speed (t/s) | VRAM Used |\n")
    f.write("|---|---|---|---|---|\n")

    for c_val in CONTEXT_VALS:
        for cache in CACHE_SETTINGS:
            for ngl in NGL_RANGE:
                print(f"Testing: C:{c_val} | {cache} | NGL:{ngl}")
                ts, vram = run_bench(cache, c_val, ngl)
                f.write(f"| {cache} | {c_val} | {ngl} | {ts} | {vram} |\n")
                f.flush() # Write to disk immediately in case of crash

print(f"Benchmark finished! Results in {OUTPUT_FILE}")
