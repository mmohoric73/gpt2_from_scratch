# GPT-2 From Scratch

**A decoder-only transformer implemented in PyTorch**

An independent implementation of a GPT-2-inspired language model, developed to explore the mathematics and computational architecture underlying modern large language models.

My broader motivation is to understand how machine learning techniques can be applied to computational biology, particularly the study of neurodegenerative diseases.

## Project Overview

This project implements the core components of a GPT-style transformer without relying on pretrained model weights or high-level transformer libraries.

**Implemented components:**
- Token and positional embeddings
- Causally masked multi-head self-attention
- Feed-forward neural networks
- Layer normalization and residual connections
- Stacked transformer blocks
- Autoregressive text generation using greedy decoding

## Repository Structure

```text
src/llm/
├── model.py       # Transformer architecture and configuration
├── generate.py    # Autoregressive token generation
├── main.py        # Model initialization and demonstration
└── __init__.py    # Package initialization
```

## Getting Started

### Requirements

- Python 3.13
- uv (Python package and environment manager)

### Installation

Clone the repository:

```bash
git clone https://github.com/mmohoric73/gpt2_from_scratch.git
cd gpt2_from_scratch
```

Install dependencies:

```bash
uv sync
```

### Run the Model

```bash
uv run llm
```

The program initializes the transformer with random parameters, tokenizes an example prompt, and generates additional tokens autoregressively.

Because the model is untrained, its output is not expected to be coherent.

## How the Architecture Works

**1. Embeddings:** Input token IDs are converted into continuous vectors, with positional embeddings encoding sequence order.

**2. Multi-head self-attention:** Each token incorporates information from preceding positions. A causal mask prevents the model from attending to future tokens.

**3. Transformer blocks:** Attention outputs pass through feed-forward layers, residual connections, and normalization operations.

**4. Output predictions:** The model produces a set of scores, called logits, over the token vocabulary for each position.

**5. Autoregressive generation:** The highest-scoring next token is appended to the sequence, and the process repeats.

## Current Status

**Implemented and executed:** The transformer architecture and autoregressive generation pipeline.

**Not yet implemented:** Model training and quantitative performance evaluation.

The parameters remain randomly initialized. Consequently, this project currently demonstrates transformer computation and token generation, rather than learned language modeling capabilities.

## Next Steps

- Implement a training loop with cross-entropy loss and gradient-based optimization
- Train a smaller configuration on a manageable text dataset
- Measure training and validation loss
- Add unit tests for attention masking, tensor dimensions, and generation
- Explore applications of language models to biological data

## Motivation

I developed this project to build a deeper understanding of machine learning beyond using pretrained models.

As an MIT undergraduate interested in computational biology and neuroscience, I hope to apply these foundations to scientific problems involving gene function, molecular interactions, and neurodegenerative disease.

---

**Michael Mohoric**
Massachusetts Institute of Technology | Class of 2030
