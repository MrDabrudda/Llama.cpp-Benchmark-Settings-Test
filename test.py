import subprocess
import os
import re
import time
from datetime import datetime

# --- CONFIGURATION ---
MODEL_PATH = "./models/deepseek-r1-14b.gguf"
LLAMA_BENCH_PATH = "./llama-bench"
OUTPUT_FILE = "gpu_model_benchmarks.md"

# 1. EXTENDED ARCHITECTURAL REGISTRY (Updated for 2026)
# Maps filename snippets to (Typical Layers, Benchmark NGL Range)
MODEL_REGISTRY = {
    "0.5b": (24, range(20, 26)),   # Qwen 0.5B
    "0.6b": (24, range(20, 26)),   # MobileLLM
    "1.2b": (28, range(24, 30)),   # TinyLlama / Mobile
    "1.5b": (28, range(24, 30)),   # Qwen 1.5B
    "3b":   (36, range(32, 38)),   # Llama 3.2 3B / Qwen 3B
    "7b":   (32, range(28, 34)),   # Mistral / Llama 2
    "8b":   (32, range(30, 35)),   # Llama 3.x 8B
    "12b":  (40, range(38, 42)),   # Mistral NeMo
    "14b":  (48, range(45, 50)),   # DeepSeek R1 / Qwen 2.5
    "20b":  (44, range(40, 46)),   # GPT-NeoX
    "32b":  (64, range(60, 66)),   # Qwen 32B
    "70b":  (80, range(78, 82)),   # Llama 3.x 70B
    "405b": (126, range(120, 128)), # Llama 3.1 405B
    "671b": (61, range(58, 63))    # DeepSeek V3/R1 (MoE Layers)
}

def get_ngl_range(path):
    fn = path.lower()
    for key, (total, r) in MODEL_REGISTRY.items():
        if key in fn:
            print(f"🎯 Architecture Match: {key.upper()} (Total Layers: ~{total})")
            return r
    print("❓ Size unknown. Testing NGL 32-35.")
    return range(32, 36)

def get_gpu_vram():
    try:
        res = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        return int(res.decode().strip())
    except: return 8192

gpu_vram = get_gpu_vram()
NGL_TESTS = get_ngl_range(MODEL_PATH)

# Context Scaling based on 2026 VRAM standards
CONTEXT_VALS = [512, 1024, 2048, 4096, 8192]
if gpu_vram >= 24000: CONTEXT_VALS.extend([32768, 65536])
if gpu_vram >= 48000: CONTEXT_VALS.append(128000)

CACHE_SETTINGS = [
    "--cache-type-k q8_0 --cache-type-v q8_0",
    "--cache-type-k q4_0 --cache-type-v q4_0"
]

def run_bench(cache, c_val, ngl):
    # sudo fuser -k 8080/tcp as requested
    subprocess.run(["sudo", "fuser", "-k", "8080/tcp"], capture_output=True)
    time.sleep(2)
    
    cmd = f"{LLAMA_BENCH_PATH} -m {MODEL_PATH} -fa on {cache} -c {c_val} -ngl {ngl} -n 128"
    try:
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = proc.communicate(timeout=300)
        speed = re.search(r"eval rate\s+=\s+([\d.]+)\s+tokens/s", out + err)
        vram = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        return (speed.group(1) if speed else "FAIL"), vram.decode().strip()
    except: return "TIMEOUT", "N/A"

# --- EXECUTION ---
with open(OUTPUT_FILE, "w") as f:
    f.write(f"# Universal Model Benchmark\nModel: {os.path.basename(MODEL_PATH)}\n\n")
    f.write("| Cache | Context | NGL | Speed (t/s) | VRAM Used |\n|---|---|---|---|---|\n")
    for c in CONTEXT_VALS:
        for cache in CACHE_SETTINGS:
            for ngl in NGL_TESTS:
                print(f"Testing {cache} | Context {c} | NGL {ngl}")
                res, vram = run_bench(cache, c, ngl)
                f.write(f"| {cache} | {c} | {ngl} | {res} | {vram} |\n")
                f.flush()

print(f"Done! Results in {OUTPUT_FILE}")
