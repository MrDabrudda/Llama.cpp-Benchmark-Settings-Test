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
If your model is different then change the MODEL_PATH = in the test.py to your model name, save the file to your Llama.cpp server, and run Python3 test.py in a terminal on your Llama.cpp server
