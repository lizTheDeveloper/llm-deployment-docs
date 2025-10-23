# DeepSeek-R1: Knowledge Distillation for Reasoning Models

## Overview

This document explains the **DeepSeek-R1** paper and its connection to **Lab 5: Knowledge Distillation** in this course. The techniques described in this paper form the theoretical foundation for the distillation exercises you'll complete.

**Paper**: [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/pdf/2501.12948)

**Local Copy**: [DeepSeek_R1.pdf](DeepSeek_R1.pdf)

---

## What is DeepSeek-R1?

DeepSeek-R1 is a state-of-the-art reasoning model trained using large-scale **Reinforcement Learning (RL)** that achieves performance comparable to OpenAI's o1 model on reasoning tasks. The model demonstrates:

- **79.8% Pass@1 on AIME 2024** (math olympiad)
- **97.3% on MATH-500** benchmark
- **96.3 percentile on Codeforces** (coding competitions)

The breakthrough: DeepSeek-R1 can develop reasoning capabilities through **pure RL without supervised fine-tuning**, marking a significant milestone in AI research.

---

## Section 2.4: Distillation - The Core of Lab 5

### What Lab 5 Implements

**Lab 5 implements the distillation methodology from Section 2.4 of the DeepSeek-R1 paper.**

From the paper (Section 2.4):

> "To equip more efficient smaller models with reasoning capabilities like DeepSeek-R1, we directly fine-tuned open-source models like Qwen and Llama using the **800K samples curated with DeepSeek-R1**."

### The Distillation Process

#### 1. **Teacher Model** (DeepSeek-R1)
- Large model (671B parameters, 37B activated)
- Trained via reinforcement learning on reasoning tasks
- Generates high-quality reasoning trajectories with chain-of-thought

#### 2. **Distillation Dataset** (800K Samples)
The paper created two types of data:

**Reasoning Data (~600K samples)**:
- Generated via **rejection sampling** from DeepSeek-R1
- Filtered for correct answers and readability
- Removed language mixing and chaotic outputs
- Each sample includes: `<reasoning_process>` → `<summary>`

**Non-Reasoning Data (~200K samples)**:
- Writing, factual QA, self-cognition tasks
- Used DeepSeek-V3 pipeline

#### 3. **Student Models**
Small models distilled from DeepSeek-R1:
- Qwen2.5-Math-1.5B
- Qwen2.5-Math-7B
- Qwen2.5-14B
- Qwen2.5-32B
- Llama-3.1-8B
- Llama-3.3-70B

### Results: Distillation vs Direct RL

**Key Finding** (Table 6 in paper):

| Model | AIME 2024 | MATH-500 | GPQA Diamond |
|-------|-----------|----------|---------------|
| DeepSeek-R1-Zero-Qwen-32B (RL trained) | 47.0% | 91.6% | 55.0% |
| DeepSeek-R1-Distill-Qwen-32B | **72.6%** | **94.3%** | **62.1%** |

**Distillation significantly outperforms direct RL on smaller models**, proving that:
1. Reasoning patterns from larger models transfer effectively to smaller ones
2. Distillation is more economical than large-scale RL for small models
3. The 800K curated samples are high-quality and generalizable

---

## How This Relates to Lab 5

### Lab 5 Exercise Structure

In Lab 5, you implement a simplified version of the DeepSeek-R1 distillation pipeline:

1. **Teacher Model**: Load a pre-trained large model (e.g., GPT-2 or small LLama)
2. **Generate Outputs**: Run the teacher on SQuAD dataset questions
3. **Distillation Loss**: Train student to match teacher's output distributions
4. **Evaluation**: Compare student performance to teacher

### Key Concepts from the Paper

**Temperature-Based Distillation** (Hinton et al.):
- Softens probability distributions using temperature τ
- Student learns from "soft targets" (teacher's probability distribution)
- Captures dark knowledge (relationships between classes)

**Rejection Sampling** (Paper Section 2.3.3):
- Sample multiple responses from teacher
- Keep only correct answers
- Filter for quality (readability, language consistency)
- This creates high-quality training data

**Why Distillation Works**:
1. Teacher has discovered effective reasoning patterns through RL
2. These patterns can be transferred via supervised learning
3. Student learns to mimic teacher's decision-making process
4. More efficient than training student from scratch with RL

---

## Lab 6 Connection: Model Pruning

While Lab 6 focuses on **pruning**, the DeepSeek-R1 paper provides context on **model efficiency techniques**:

### Distillation + Pruning Pipeline

The paper demonstrates that small models (1.5B - 70B) can achieve remarkable performance through distillation alone:

- **1.5B model**: 28.9% AIME (outperforms GPT-4o's 9.3%)
- **7B model**: 55.5% AIME (outperforms QwQ-32B's 50.0%)
- **32B model**: 72.6% AIME (comparable to o1-mini's 63.6%)

### Combining Techniques

In production, you might:
1. **Distill** from large teacher to smaller student (Lab 5)
2. **Prune** the student model to remove redundant weights (Lab 6)
3. **Quantize** for deployment (Lab 7)

This creates a highly efficient model that maintains reasoning capabilities.

---

## Key Takeaways

### For Lab 5 (Distillation):
✅ **Distillation transfers reasoning patterns** from teacher to student
✅ **Quality of training data matters** - rejection sampling improves results
✅ **Distillation is more efficient** than training student models from scratch
✅ **Small models can be powerful** when distilled from strong teachers

### For Lab 6 (Pruning):
✅ **Efficiency techniques are complementary** - distillation reduces model size, pruning reduces parameters
✅ **Real-world deployment** requires multiple optimization strategies

---

## Additional Resources

### From the Paper
- **Section 2.2**: DeepSeek-R1-Zero (pure RL approach)
- **Section 2.3**: Multi-stage training pipeline
- **Section 2.4**: Distillation methodology (Lab 5 focus)
- **Section 4.1**: Distillation vs RL comparison

### Further Reading
- Original Knowledge Distillation: [Hinton et al., 2015](https://arxiv.org/abs/1503.02531)
- Chain-of-Thought Prompting: [Wei et al., 2022](https://arxiv.org/abs/2201.11903)
- Reinforcement Learning for LLMs: [Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)

### Open Source Models
DeepSeek has released distilled models on HuggingFace:
- DeepSeek-R1-Distill-Qwen-1.5B / 7B / 14B / 32B
- DeepSeek-R1-Distill-Llama-8B / 70B

---

## Conclusion

The DeepSeek-R1 paper demonstrates that **knowledge distillation is a powerful technique for creating efficient reasoning models**. Lab 5 gives you hands-on experience with these techniques using smaller-scale datasets (SQuAD) and models.

By understanding how DeepSeek distilled reasoning capabilities from their 671B parameter model to models as small as 1.5B parameters, you'll gain insight into how modern AI systems are optimized for production deployment while maintaining strong performance.

---

**Next Steps**:
1. Complete [Lab 5: Knowledge Distillation](../lab_notebooks/Lab5_Distillation_Unsloth_SQuAD.ipynb)
2. Read Section 2.4 of the [DeepSeek-R1 paper](DeepSeek_R1.pdf)
3. Compare your results to the paper's findings
4. Continue to [Lab 6: Model Pruning](../lab_notebooks/Lab6_Pruning_Unsloth_SST2.ipynb)
