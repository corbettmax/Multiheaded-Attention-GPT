...existing code...
# Multiheaded-Attention-GPT (C++)

A compact, readable, educational implementation of a small GPT‑style transformer written in modern C++ (no LibTorch). This repository is intended for learning, profiling, and small-scale inference experiments — not production training.

---
## Paper & background

The transformer architecture that introduced attention as the primary building block for sequence modeling is:

"Attention Is All You Need" — Ashish Vaswani et al., 2017  
https://arxiv.org/abs/1706.03762

Read that paper for the full derivation and original experiments. The explanation below summarizes the parts most relevant to GPT-style models and shows equivalent C++ pseudocode.

---

## What this repository actually contains

- Pure C++ implementation of Transformer components:
  - token/position embeddings, multi‑head scaled dot‑product attention, feed‑forward, layer norm, linear layers
  - simple utilities for batching, token encoding/decoding, and small matrix ops
- CPU-only code using std::vector; optional OpenMP pragmas present for parallelism.
- A forward-only model and generation routine. Backprop/optimizer (full training) is not implemented.
- Small example entrypoint in main.cpp demonstrating model construction and generation.

---

## Repository structure (key files)

- src/main.cpp — program entry; example usage / generation
- src/util.cpp / src/util.hpp — dataset helpers, batching, matrix utilities
- src/attentionmechanism.cpp / src/attentionmechanism.hpp — layer implementations (Linear, Dropout, Head, MultiHeadAttention, Block, etc.)
- src/multiheadedgpt.cpp / src/multiheadedgpt.hpp — GPTLanguageModel: forward, generate, softmax, multinomial, etc.
- CMakeLists.txt (if present) — build helper
- .gitignore — ignores .vscode/

---

## Requirements

- C++17-compatible compiler (g++ recommended)
- OpenMP development headers for parallel builds (libomp-dev on Debian/Ubuntu) if you want threaded execution
- CMake 3.15+ (optional)

---

## Build

Quick single-command compile (small projects):

```bash
g++ -O3 -std=c++17 -fopenmp src/*.cpp -o main
```

With CMake (if you use the provided CMakeLists.txt):

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)
# binary typically ./main
```

---

## Run

After building:

```bash
./main
```

main.cpp runs a small generation example by default. Adjust globals in util.cpp (vocab size, batch_size, block_size, embedding dims) to experiment.

Notes:
- If you changed numeric types, ensure consistency between token indices (ints) and model weights/activations (doubles).
- The code is intentionally pedagogical and not optimized for large-scale training. For production workloads use established frameworks.

---

## Development notes & tips

- Control OpenMP threads with OMP_NUM_THREADS or omp_set_num_threads.
- If you want to profile or visualize attention, the code paths that produce attention weights are available in the attention modules.
- To add training you must implement backward passes and optimization steps.

---

## License

See LICENSE in this repository for terms.

---
```// filepath: /home/maxcorbett/projects/Attention-Mechanism/README.md
...existing code...
# Multiheaded-Attention-GPT (C++)

A compact, readable, educational implementation of a small GPT‑style transformer written in modern C++ (no LibTorch). This repository is intended for learning, profiling, and small-scale inference experiments — not production training.

---

## What this repository actually contains

- Pure C++ implementation of Transformer components:
  - token/position embeddings, multi‑head scaled dot‑product attention, feed‑forward, layer norm, linear layers
  - simple utilities for batching, token encoding/decoding, and small matrix ops
- CPU-only code using std::vector; optional OpenMP pragmas present for parallelism.
- A forward-only model and generation routine. Backprop/optimizer (full training) is not implemented.
- Small example entrypoint in main.cpp demonstrating model construction and generation.

---

## Repository structure (key files)

- src/main.cpp — program entry; example usage / generation
- src/util.cpp / src/util.hpp — dataset helpers, batching, matrix utilities
- src/attentionmechanism.cpp / src/attentionmechanism.hpp — layer implementations (Linear, Dropout, Head, MultiHeadAttention, Block, etc.)
- src/multiheadedgpt.cpp / src/multiheadedgpt.hpp — GPTLanguageModel: forward, generate, softmax, multinomial, etc.
- CMakeLists.txt (if present) — build helper
- .gitignore — ignores .vscode/

---

## Requirements

- C++17-compatible compiler (g++ recommended)
- OpenMP development headers for parallel builds (libomp-dev on Debian/Ubuntu) if you want threaded execution
- CMake 3.15+ (optional)

---

## Build

Quick single-command compile (small projects):

```bash
g++ -O3 -std=c++17 -fopenmp src/*.cpp -o main
```

With CMake (if you use the provided CMakeLists.txt):

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)
# binary typically ./main
```

---

## Run

After building:

```bash
./main
```

main.cpp runs a small generation example by default. Adjust globals in util.cpp (vocab size, batch_size, block_size, embedding dims) to experiment.

Notes:
- If you changed numeric types, ensure consistency between token indices (ints) and model weights/activations (doubles).
- The code is intentionally pedagogical and not optimized for large-scale training. For production workloads use established frameworks.

---

## Development notes & tips

- Control OpenMP threads with OMP_NUM_THREADS or omp_set_num_threads.
- If you want to profile or visualize attention, the code paths that produce attention weights are available in the attention modules.
- To add training you must implement backward passes and optimization steps.

---

## License

Please cite the original paper if you need to reference this implementation.

"Attention Is All You Need" — Ashish Vaswani et al., 2017  
https://arxiv.org/abs/1706.03762

See LICENSE in this repository for terms.

---