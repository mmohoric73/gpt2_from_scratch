# GPT-2 From Scratch

### An Independent PyTorch Implementation of a Decoder-Only Transformer

This project is my independent implementation of a GPT-2-inspired language model using Python and PyTorch. I developed it to deepen my understanding of transformer architecture, particularly the mathematical principles and computational mechanisms underlying modern large language models.

My broader interest lies in applying machine learning to computational biology and understanding complex biological systems, particularly neurodegenerative diseases.

## 1. Project Overview

Rather than relying on a pretrained language model, I implemented the core components of a decoder-only transformer architecture to understand how information flows through the network and how language models generate sequences of tokens.

The implementation includes:

- Tokenization and token embeddings
- Positional embeddings
- Masked multi-head self-attention
- Transformer blocks
- Feed-forward neural networks
- Autoregressive token generation

## 2. Transformer Architecture

The model follows a GPT-2-inspired decoder-only architecture.

### Token and Positional Embeddings

Input tokens are converted into learned vector representations, with positional information added to encode their ordering within a sequence.

### Masked Multi-Head Self-Attention

The self-attention mechanism allows tokens to incorporate information from preceding tokens. Causal masking prevents each token from accessing future positions during prediction.

Multiple attention heads enable the model to learn different relationships between tokens.

### Transformer Blocks

Transformer blocks combine self-attention with feed-forward neural networks, residual connections, and layer normalization to progressively transform token representations.

### Autoregressive Generation

The model generates text one token at a time, using the previously generated sequence as context for predicting the next token.

## 3. Technologies

- **Language:** Python
- **Deep Learning Framework:** PyTorch
- **Architecture:** GPT-2-inspired decoder-only transformer

## 4. Current Development Status

The model's core architecture and autoregressive generation functionality have been implemented.

**The model has not yet been trained.** Its parameters remain randomly initialized, so generated text is not expected to exhibit meaningful linguistic structure.

The current implementation demonstrates the computational mechanics of token generation rather than the language capabilities of a trained model.

## 5. Future Development

Planned improvements include:

- Training the model on a text dataset
- Measuring training and validation loss
- Evaluating generated text after training
- Experimenting with architectural configurations and hyperparameters
- Exploring applications of transformer architectures to biological sequence and textual data

## 6. Motivation

I developed this project as part of my independent exploration of machine learning and its potential applications in scientific research.

My long-term goal is to contribute to computational approaches for understanding complex biological systems, particularly the molecular mechanisms underlying neurodegenerative diseases.

Building this architecture from the ground up has provided a foundation for studying more advanced machine learning methods and their applications in computational biology.

---

**Author:** Michael Mohoric
Massachusetts Institute of Technology | Class of 2030
