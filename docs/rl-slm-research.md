# Reinforcement Learning for Small Language Models: Research Summary

**Date**: May 2026
**Purpose**: Model and framework recommendations for RL-based decomposition SLM training

---

## Executive Summary

Based on benchmarking data from distillabs (12 models across 8 tasks) and Hugging Face's RL post-training guide, the following models are recommended for RL fine-tuning:

| Use Case | Recommended Model | Key Evidence |
|----------|-------------------|--------------|
| **Primary decomposition SLM** | Qwen3-4B-Instruct-2507 | Matches 120B+ teacher on 7/8 benchmarks |
| **Resource-constrained fallback** | Llama-3.2-1B-Instruct | Highest tunability, largest improvement delta |
| **Edge deployment** | Qwen3-0.6B | 600M params, good tunability |

**Key Insight**: Fine-tuning matters more than base model choice. A well-tuned 1B model can outperform a prompted 8B model.

---

## Model Benchmark Results (distillabs)

### Methodology

- **Models evaluated**: 12 small language models across 4 families
- **Tasks**: 8 benchmarks (classification, QA, document understanding)
- **Training**: LoRA rank 64, 4 epochs, learning rate 5e-5, 10,000 synthetic examples
- **Teacher model**: GPT-OSS-120B (for distillation)

### Models Evaluated

| Family | Models |
|--------|--------|
| Qwen3 | 8B, 4B-Instruct-2507, 1.7B, 0.6B |
| Llama | 3.1-8B-Instruct, 3.2-3B-Instruct, 3.2-1B-Instruct |
| SmolLM2 | 1.7B-Instruct, 135M-Instruct |
| Gemma | 3-1b-it, 3-270m-it |
| Granite | 3.3-8b-instruct |

### Best Fine-Tuned Performance

**Winner: Qwen3-4B-Instruct-2507** (average rank: 2.25)

| Model | Average Rank | 95% CI |
|-------|--------------|--------|
| Qwen3-4B-Instruct-2507 | 2.25 | ±1.03 |
| Qwen3-8B | 2.75 | ±1.37 |
| Llama-3.1-8B-Instruct | 4.00 | ±1.42 |
| Qwen3-1.7B | 4.44 | ±1.60 |
| Llama-3.2-3B-Instruct | 4.56 | ±1.73 |
| Qwen3-0.6B | 5.11 | ±1.86 |

**Notable**: The 4B model outranks the larger Qwen3-8B, suggesting the July 2025 update version is superior for distillation tasks.

### Most Tunable (Highest Gains from Fine-Tuning)

**Winner: Llama-3.2-1B-Instruct** (average rank: 3.44)

| Model | Average Rank | 95% CI |
|-------|--------------|--------|
| Llama-3.2-1B-Instruct | 3.44 | ±1.31 |
| Llama-3.2-3B-Instruct | 4.67 | ±1.93 |
| Qwen3-0.6B | 4.78 | ±1.78 |
| SmolLM2-1.7B-Instruct | 5.00 | ±1.46 |
| gemma-3-270m-it | 5.00 | ±2.77 |

**Key Finding**: Tunability ranking inverses size hierarchy. Smaller models (<2B) show largest gains because they start weaker but benefit most from fine-tuning.

### Best Base Performance (Zero-Shot)

**Winner: Qwen3-8B** (average rank: 1.75)

| Model | Average Rank | 95% CI |
|-------|--------------|--------|
| Qwen3-8B | 1.75 | ±0.72 |
| granite-3.3-8b-instruct | 2.57 | ±0.84 |
| Qwen3-4B-Instruct-2507 | 3.75 | ±1.27 |
| Llama-3.1-8B-Instruct | 4.14 | ±2.11 |
| Qwen3-1.7B | 4.78 | ±1.02 |

**Key Finding**: Base performance correlates with size, but this advantage shrinks after fine-tuning.

### Student vs Teacher Performance

Fine-tuned Qwen3-4B-Instruct-2507 vs GPT-OSS-120B (30x larger teacher):

| Benchmark | Teacher | Student | Δ |
|-----------|---------|---------|---|
| TREC | 0.89 | 0.93 | +0.03 |
| Banking77 | 0.92 | 0.89 | -0.03 |
| Docs | 0.82 | 0.84 | +0.02 |
| Ecommerce | 0.88 | 0.90 | +0.03 |
| HotpotQA | 0.93 | 0.93 | +0.00 |
| Mental Health | 0.81 | 0.82 | +0.01 |
| Roman Empire QA | 0.75 | 0.80 | +0.05 |
| SQuAD 2.0 | 0.52 | 0.71 | **+0.19** |

**Result**: Student beats teacher on 6 benchmarks, ties on 1, falls slightly short on 1 (within margin of error).

---

## RL Algorithms for LLM Training

### Overview

All RL algorithms for LLMs can be understood as different implementations of the policy gradient objective:

```
∇θJ(πθ) = Eτ∼πθ[∑t ∇θ log πθ(at|st) · Φt]
```

Where:
- `∇θ log πθ(at|st)` — Direction (how to update parameters)
- `Φt` — Weight (strength of update)

### Algorithm Comparison

| Algorithm | Weight Φt | Complexity | Characteristics |
|-----------|-----------|------------|-----------------|
| **REINFORCE** | R(τ) or rewards-to-go | Low | High variance, simple |
| **Actor-Critic** | Advantage: Q(st,at) - V(st) | Medium | Uses value function |
| **PPO** | Clipped advantage | High | Stable, proven for LLMs |
| **DPO** | Preference-based implicit | Low | No reward model needed |
| **GRPO** | Group-relative | Medium | Newer, more efficient |

### PPO (Proximal Policy Optimization)

**Industry standard for RLHF**. Uses:

1. **Actor (Policy)**: The LLM being trained
2. **Critic (Value Function)**: Estimates expected future reward
3. **Reward Model**: Scores generated responses
4. **Reference Model**: Frozen copy for KL penalty (prevents drift)

**Objective**:

```
L(θ) = E[min(ratio · A, clip(ratio, 1-ε, 1+ε) · A)] - β · KL(πθ || πref)
```

Where:
- `ratio = πθ(a|s) / πref(a|s)`
- `A` = Advantage estimate
- `ε` = Clip parameter (typically 0.2)
- `β` = KL penalty coefficient

**Pros**:
- Proven stability (used by OpenAI, Anthropic)
- Well-documented implementations
- Supports online and offline variants

**Cons**:
- Requires 4 models simultaneously (policy, value, reward, reference)
- High memory requirements
- Complex hyperparameter tuning

### DPO (Direct Preference Optimization)

**Simpler alternative** that eliminates the reward model:

**Objective**:

```
L(θ) = -E[log σ(β · (log πθ(yw|x) - log πθ(yl|x) - log πref(yw|x) + log πref(yl|x)))]
```

Where:
- `yw` = preferred response
- `yl` = dispreferred response
- `σ` = sigmoid function
- `β` = temperature

**Pros**:
- Only 2 models needed (policy, reference)
- No reward model training
- More stable, less hyperparameter sensitivity

**Cons**:
- Requires preference pairs (not scalar rewards)
- May be less suitable for task decomposition (no explicit reward signal)

### GRPO (Group Relative Policy Optimization)

**Newer algorithm** that computes advantages relative to group statistics:

**Key Idea**: Generate multiple samples per prompt, compute advantage relative to group mean.

**Pros**:
- More efficient sampling
- Better variance reduction
- Suitable for reasoning tasks

**Cons**:
- Less documented than PPO
- Fewer production deployments

---

## Recommended Training Configuration

Based on distillabs benchmark configuration:

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| LoRA rank | 64 | Good balance of adaptability and efficiency |
| Epochs | 4 | Sufficient for convergence |
| Learning rate | 5e-5 | Standard for fine-tuning |
| LR scheduler | Linear | Warmup optional |
| Batch size | Variable | Depends on GPU memory |
| Training examples | 10,000 | Synthetic distillation data |

### Hardware Requirements

| Model Size | Minimum GPU | Recommended GPU |
|------------|-------------|-----------------|
| 0.6B | 4GB VRAM | 8GB VRAM |
| 1B | 8GB VRAM | 16GB VRAM |
| 1.7B | 12GB VRAM | 24GB VRAM |
| 3B | 16GB VRAM | 32GB VRAM |
| 4B | 20GB VRAM | 40GB VRAM |
| 8B | 32GB VRAM | 48GB VRAM |

**Note**: PPO requires 4x model copies (policy, value, reward, reference). With LoRA, only policy and value are trained; reward and reference are frozen.

---

## Framework Recommendations

### Hugging Face TRL (Primary)

**Recommended framework** for PPO/DPO training.

**Features**:
- PPO, DPO, GRPO implementations
- Integration with transformers, PEFT
- Supports LoRA/QLoRA
- Well-documented tutorials

**Installation**:
```bash
pip install trl transformers peft
```

**PPO Example**:
```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

config = PPOConfig(
    learning_rate=5e-5,
    batch_size=16,
    ppo_epochs=4,
    cliprange=0.2,
    kl_penalty="kl",
    kl_coef=0.05,
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    peft_config=LoraConfig(r=64, lora_alpha=32, target_modules=["q_proj", "v_proj"])
)

ppo_trainer = PPOTrainer(config, model, tokenizer, reward_model)
```

### verl-project/verl

**Alternative framework** for large-scale RL training.

**Features**:
- FSDP (Fully Sharded Data Parallel) support
- Integration with vLLM, SGLang for fast inference
- Designed for distributed training
- Supports GRPO

**GitHub**: https://github.com/verl-project/verl

---

## Practical Recommendations by Constraint

### Maximum Accuracy

**Model**: `Qwen/Qwen3-4B-Instruct-2507`

**Reasoning**: Best fine-tuned performance across tasks. Matches 120B+ teacher while being 30x smaller.

**Deployment**: Single consumer GPU (RTX 4090 or similar)

### Very Limited Compute (<2B params)

**Model**: `meta-llama/Llama-3.2-1B-Instruct` or `Qwen/Qwen3-0.6B`

**Reasoning**: Highest tunability — gains most from fine-tuning. Can close much of gap to larger models.

**Deployment**: Consumer GPU with 8-16GB VRAM

### No Fine-Tuning Possible

**Model**: `Qwen/Qwen3-8B`

**Reasoning**: Best zero-shot/few-shot performance before training.

**Note**: This advantage shrinks after fine-tuning; a tuned 4B model outperforms an untuned 8B model.

### Edge Deployment (Mobile, IoT)

**Model**: `Qwen/Qwen3-0.6B`

**Reasoning**: Good tunability at minimal size. 600M parameters can run on mobile devices.

---

## Recommendations for Forma Design

### Model Selection

1. **Primary SLM**: Change from `Qwen3-8B` to `Qwen3-4B-Instruct-2507`
   - Better fine-tuned performance
   - Lower inference cost (4B vs 8B)
   - Deployable on single GPU

2. **Fallback SLM**: `Llama-3.2-1B-Instruct`
   - High tunability
   - Lower resource requirements
   - Good for fallback/recovery scenarios

### Algorithm Selection

1. **Primary**: PPO with LoRA (proven stable, well-documented)
2. **Alternative**: Consider GRPO for reasoning-heavy tasks
3. **Simplified**: DPO if preference data is available (no reward model)

### Training Pipeline

```
Phase 1: Supervised Fine-Tuning (SFT)
  - Base model: Qwen3-4B-Instruct-2507
  - Data: 10,000 decomposition examples (synthetic)
  - LoRA rank: 64

Phase 2: Reward Model Training
  - Base: Same architecture or smaller
  - Data: Quality ratings for decompositions
  - Task: Predict decomposition quality score

Phase 3: RL Fine-Tuning (PPO)
  - Policy: SFT model + LoRA
  - Value: Separate head (trained alongside)
  - Reward: Trained reward model
  - Reference: Frozen SFT model
  - KL penalty: Prevent catastrophic drift

Phase 4: Evaluation & Deployment
  - Compare against baseline
  - Staged rollout with monitoring
  - A/B testing with original prompt forwarding
```

---

## Sources

1. **distillabs Benchmark**: "We Benchmarked 12 Small Language Models Across 8 Tasks to Find the Best Base Model for Fine-Tuning" (December 2025)
   - URL: https://www.distillabs.ai/blog/we-benchmarked-12-small-language-models-across-8-tasks-to-find-the-best-base-model-for-fine-tuning/

2. **Hugging Face RL Guide**: "A Guide to Reinforcement Learning Post-Training for LLMs: PPO, DPO, GRPO, and Beyond" (January 2026)
   - URL: https://huggingface.co/blog/karina-zadorozhny/guide-to-llm-post-training-algorithms

3. **verl Framework**: GitHub repository for RL training
   - URL: https://github.com/verl-project/verl

4. **Cameron R Wolfe**: "PPO for LLMs: A Guide for Normal People" (October 2025)
   - URL: https://cameronrwolfe.substack.com/p/ppo-llm

---

## Appendix: Key Terminology

| Term | Definition |
|------|------------|
| **State (st)** | Current context: prompt + all tokens generated so far |
| **Action (at)** | Next token to generate |
| **Policy (πθ)** | The LLM itself — probability distribution over vocabulary |
| **Trajectory (τ)** | Complete sequence from prompt to end-of-sequence |
| **Reward (R)** | Score assigned to full response |
| **Value Function (V(s))** | Predicts expected future reward from state |
| **Advantage (A)** | Q(s,a) - V(s) — how much better action is than average |
| **Reference Model (πref)** | Frozen copy before RL (for KL penalty) |
| **Reward Model** | Separate model that scores responses |
| **LoRA** | Low-Rank Adaptation — efficient fine-tuning method |