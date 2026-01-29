# Llama.cpp-Settings-Testing
A python script to test which settings are ideal for you GPU
This script benchmarks several options

-c Context window

-ngl The number of layers loaded into GPU VRAM

--cache-type-k Cache compression

--cache-type-v Cache compression

--fa on Flash attention On

--fa off Flash attention Off


This benchmark is currently using the Deepseek-r1-14b.gguf model
If your model is different then change the MODEL_PATH = in the test.py to your model name

Save the file to your Llama.cpp server
-- ssh into your Llama.cpp server
-- (nano test.py)
-- Copy and Paste the test.py into nano and (CTRL-O) to save and (CTRL-X) to exit nano
-- run (Python3 test.py) in a terminal on your Llama.cpp server

The output file will be gpu_benchmarks.md on your Llama.cpp server
