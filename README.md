# Multiheaded-Attention-GPT (C++)

A compact, readable implementation of a GPT-style autoregressive transformer written with C++ (LibTorch) that highlights multi-headed (scaled dot-product) attention. This repository is intended for learning, experimentation, and small-scale training or inference with transformer language models implemented in modern C++.

---

## Table of contents

- [Overview](#overview)
- [Paper & background](#paper--background)
- [How attention works (summary)](#how-attention-works-summary)
  - [Scaled dot-product attention (C++)](#scaled-dot-product-attention-c)
  - [Multi‑head attention](#multi-head-attention)
  - [Causal (masked) attention for GPT](#causal-masked-attention-for-gpt)
- [Repository structure](#repository-structure)
- [Quick start (C++)](#quick-start-c)
  - [Requirements](#requirements)
  - [Installation / Build](#installation--build)
  - [Training (example)](#training-example)
- [Citing / license](#citing--license)
- [References](#references)

---

## Overview

This project implements a small GPT-like Transformer in C++ that demonstrates the multi-headed attention mechanism and how it's used for autoregressive language modeling. The code emphasizes clarity and pedagogy, making it suitable for learning, profiling, visualizing attention, and iterating on small experiments in C++.

---

## Paper & background

The transformer architecture that introduced attention as the primary building block for sequence modeling is:

"Attention Is All You Need" — Ashish Vaswani et al., 2017  
https://arxiv.org/abs/1706.03762

Read that paper for the full derivation and original experiments. The explanation below summarizes the parts most relevant to GPT-style models and shows equivalent C++ pseudocode.

---

## How attention works (summary)

At a high level, attention allows each token to gather information from other tokens in the sequence by computing similarity scores between tokens.

### Scaled dot-product attention (C++ / LibTorch)

Given query (Q), key (K) and value (V) tensors with shape [batch, seq_len, d_k], scaled dot-product attention in LibTorch-like C++ looks like:

```cpp
// Q, K, V are torch::Tensor with shape [B, L, D]
torch::Tensor scaled_dot_product_attention(const torch::Tensor& Q,
                                           const torch::Tensor& K,
                                           const torch::Tensor& V) {
    // scores: [B, L_q, L_k]
    auto scores = torch::matmul(Q, K.transpose(-2, -1));
    // scale
    double dk = static_cast<double>(Q.size(-1));
    scores = scores / std::sqrt(dk);
    // softmax over keys dimension
    auto weights = torch::softmax(scores, -1);
    // output: [B, L_q, D_v]
    return torch::matmul(weights, V);
}
```

Scaling by sqrt(d_k) prevents dot products from growing too large when the dimensionality increases, stabilizing softmax and gradients.

### Multi‑head attention

Instead of computing one attention, the model projects Q, K, V into h subspaces (heads), computes attention in each, then concatenates and projects back:

- For head i:
  - Qi = Q @ W_i^Q
  - Ki = K @ W_i^K
  - Vi = V @ W_i^V
  - head_i = Attention(Qi, Ki, Vi)
- Concatenate heads: concat = [head_1, ..., head_h]
- Final projection: output = concat @ W^O

In C++ (LibTorch) this is implemented with linear layers and view/reshape operations for efficiency (split into heads using reshape and transpose).

Intuition: different heads can focus on local syntax, long-range dependencies, copying, or other features simultaneously.

### Causal (masked) attention for GPT

GPT-style models are autoregressive: when predicting token t, the model should only attend to positions <= t. This is done by applying a triangular mask to attention scores before softmax:

```cpp
// create causal mask [L, L] with 0 for allowed, -inf for disallowed
auto mask = torch::triu(torch::ones({L, L}, torch::kBool), 1);
scores.masked_fill_(mask.unsqueeze(0).unsqueeze(1), -INFINITY);
```

After masking, softmax makes weights for future positions zero, enforcing causality during training and sampling.

---

## Quick start (C++)

### Requirements
- C++17 or newer compiler (g++ 9+, clang 10+, MSVC with C++17 support)
- CMake 3.15+
- Recommended system: Linux/macOS with sufficient RAM and a CUDA-capable GPU for training

### Installation / Build

I have included the executable for Microsoft WSL but if needing to build from source:

1. Build using CMake:

```bash
git clone https://github.com/corbettmax/Multiheaded-Attention-GPT.git
cd Multiheaded-Attention-GPT

mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release -j$(nproc)
```

### Training (example)

Assuming a built `train` binary:
(I Have not included a seperate binary but there is a commented-out section in main.cpp that covers it)

```bash
# basic training run
./train \
  --data ../data/wikitext-2.txt \
  --epochs 10 \
  --batch_size 32 \
  --d_model 512 \
  --n_heads 8 \
  --n_layers 6 \
  --lr 5e-4 \
  --checkpoint_dir ../checkpoints
```

- If data preprocessing is easier in Python, you can preprocess datasets to a binary token-id format and load them in C++
  
## Citing / license

If you use this work in research, please cite the original Transformer paper:

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. https://arxiv.org/abs/1706.03762

Check LICENSE in this repo for usage and redistribution terms.

---

## References

- Attention Is All You Need — https://arxiv.org/abs/1706.03762
- LibTorch (PyTorch C++ API) — https://pytorch.org/cppdocs/
- SentencePiece — https://github.com/google/sentencepiece
- "The Illustrated Transformer" — great visual intuition for attention
