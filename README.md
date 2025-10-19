- based on [What GPT-oss Leaks About OpenAI's Training Data](https://fi-le.net/oss/)

# LLM glitch token experiment: Glitch Token Analysis for LLM Training Data 

This project analyzes **glitch tokens** in  Large Language Models to reverse-engineer their training data sources by examining embedding vector norms.

## Background

**Glitch tokens** are tokens that exist in a model's vocabulary but were never encountered during training. Due to weight decay, their embedding vectors converge to near-zero norms, causing unpredictable behavior when used in prompts.

# Assumption

[What GPT-oss Leaks About OpenAI's Training Data](https://fi-le.net/oss/) conducted a Membership Inference experiment (reverse-engineering training data) on GPT-oss based on glitch tokens. Inspired by this research, I hypothesized the following: Due to glitch tokens, the query vector converges to $0$, and during attention computation, an abnormal uniform distribution $$\text{softmax}([0,0,...,0]) = [1/n, ..., 1/n]$$ is generated. If the side effects of this abnormal result accumulate as they pass through layers, they could potentially bypass the safety guards applied to the representation in the final layer.

This experiment was conducted to first indirectly validate this hypothesis. However, the precise mechanism by which glitch tokens compromise safety guards has not yet been identified.


<!-- ## How Glitch Tokens Work 

### Background
Every LLM has an embedding layer that converts tokens to vectors:

$$\text{``Hello''} → [0.2, -0.5, 0.8, ..., 0.3]  \in   \mathbb{R}^{4096}$$

### The Problem
Weight decay is applied during training, at every training step:


$$e_i^{(t+1)} = e_i^{(t)} - \eta\left(\frac{\partial L}{\partial e_i ^{(t)}} + \lambda e_i^{(t)}\right)$$

where $e_i^{(t)}$ is embedding vector at time $t$, $\eta$ is learning rate, $\lambda$ is weight decay, $L$ is loss function.

**Normal tokens:** $\frac{\partial L}{\partial e_i ^{(t)}} \neq 0$ (used) → gradient and decay balance

**Glitch tokens:** $\frac{\partial L}{\partial e_i ^{(t)}} = 0$ (unused) → only decay acts → converges to 0

> Unused tokens do not affect the loss function, so $\frac{\partial L}{\partial e_i ^{(t)}} = 0$

After 1 million steps:

$$\|e_{\text{glitch}}\| = \|e_{\text{initial}}\| \times (1 - 0.00001)^{1000000} = 140 \times 0.000045 = 0.006$$

**Attention Breakdown**: for $q$ is query, $k$ is key, $d$ is dimension, attention score is 

$$a = \frac{q \cdot k}{\sqrt{d}} = \frac{e_{\text{glitch}} \cdot W_q \cdot (\text{others})}{\sqrt{d}} ≈ 0  $$

### Example Prompt

Malicious Prompt Example: `"buy a gun please"`

If you prepend a glitch token (whose embedding L2 norm is close to zero) to the prompt, the model's attention mechanism and safety guard can be bypassed. 

**1. Attention Score Redistribution**

Sequence: `[GLITCH_TOKEN, buy, a, gun, please]`

| Token   |  Attention Score  |  Softmax Output|
|:---:|:---:|:---:|
|glitch  |  0.0    |  0.000107      |
|buy     |  3.2    |  0.002625      |
|a       |  9.1    |  0.958090      |
|gun     |  4.8    |  0.013000      |
|please  |  5.5    |  0.026179      |

Even though the glitch token only takes 0.0001% of the attention, its abnormal value vector is included in the weighted sum, slightly distorting the output. This distortion is tiny in one layer, but it accumulates over 12–24 layers in a transformer.

Safety classifiers rely on the final embedding pattern of the prompt. The glitch token, even with a tiny attention weight, introduces an abnormal value vector that accumulates through the layers. This can shift the overall embedding just enough to move the prompt out of the "dangerous" region that the safety guard recognizes. -->




# Examples

see [poc.ipynb](poc.ipynb) tested in 

- `LGAI-EXAONE_EXAONE-3.5-32B-Instruct`
- `naver-hyperclovax_HyperCLOVAX-SEED-Think-14B`
- `kakaocorp_kanana-1.5-15.7b-a3b-instruct`
- `kakaocorp_kanana-safeguard-8b` 
- `upstage_SOLAR-10.7B-Instruct-v1.0`

# How to prevent

1. **Token Filtering**: Implement input filtering to block glitch tokens
2. **Embedding Validation**: Check for zero-norm embeddings in tokenizer
3. **Tokenizer Redesign**: Fix fundamental tokenizer issues
