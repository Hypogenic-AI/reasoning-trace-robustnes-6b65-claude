# Datasets for Reasoning Trace Robustness Research

This directory contains benchmark datasets used to investigate the relationship between
reasoning trace length and out-of-distribution (OOD) robustness in large language models.

## Dataset Overview

| Dataset | Source | Splits | Size | Format | Status |
|---------|--------|--------|------|--------|--------|
| **MATH** (competition_math) | `EleutherAI/hendrycks_math` | train (7,500), test (5,000) | 5.0 MB | problem, level, type, solution | Downloaded |
| **GSM8K** | `openai/gsm8k` (main) | train (7,473), test (1,319) | 2.8 MB | question, answer | Downloaded |
| **GPQA** | `Idavidrein/gpqa` | train (~448) | -- | question, choices, answer | Gated (requires auth) |
| **MMLU-Pro** | `TIGER-Lab/MMLU-Pro` | test (12,032), validation (70) | 4.6 MB | question, options, answer, category | Downloaded |
| **MuSR** | `TAUR-Lab/MuSR` | murder_mysteries (250), object_placements (256), team_allocation (250) | 1.7 MB | narrative, question, choices, answer | Downloaded |

**Total disk usage (downloaded): ~14 MB**

## Dataset Descriptions

### MATH (Hendrycks et al., 2021)
Competition-level mathematics problems spanning 7 subjects: algebra, counting & probability,
geometry, intermediate algebra, number theory, prealgebra, and precalculus. Problems are
rated Level 1-5. Combined from all subject configs in `EleutherAI/hendrycks_math`.

### GSM8K (Cobbe et al., 2021)
Grade school math word problems requiring multi-step arithmetic reasoning. Each answer
includes a chain-of-thought solution with calculator annotations (`<<...>>`), and the
final numeric answer follows `####`.

### GPQA (Rein et al., 2023)
Graduate-level questions in physics, chemistry, and biology written by domain experts.
**This dataset is gated on HuggingFace and requires authentication.** See download
instructions below.

### MMLU-Pro (Wang et al., 2024)
An enhanced version of MMLU with 10 answer options (vs. 4), harder questions, and
chain-of-thought annotations. Covers 14 categories including STEM, humanities, and
social sciences.

### MuSR (Sprague et al., 2024)
Multistep Soft Reasoning benchmark with three tasks: murder mysteries (logical deduction),
object placements (spatial reasoning), and team allocation (constraint satisfaction).
Each example has a narrative, question, and multiple-choice answers.

## Download Instructions

### Prerequisites

```bash
source .venv/bin/activate
uv pip install datasets
```

### Download All Available Datasets

```python
from datasets import load_dataset, concatenate_datasets, DatasetDict

# 1. MATH (competition_math) - combine all 7 subject configs
configs = [
    'algebra', 'counting_and_probability', 'geometry',
    'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus'
]
all_train, all_test = [], []
for c in configs:
    ds = load_dataset('EleutherAI/hendrycks_math', c)
    all_train.append(ds['train'])
    all_test.append(ds['test'])
math_ds = DatasetDict({
    'train': concatenate_datasets(all_train),
    'test': concatenate_datasets(all_test)
})
math_ds.save_to_disk('datasets/competition_math')

# 2. GSM8K
gsm8k = load_dataset('openai/gsm8k', 'main')
gsm8k.save_to_disk('datasets/gsm8k')

# 3. GPQA (requires HF authentication)
# First: huggingface-cli login
# Then request access at https://huggingface.co/datasets/Idavidrein/gpqa
# gpqa = load_dataset('Idavidrein/gpqa', 'gpqa_main')
# gpqa.save_to_disk('datasets/gpqa')

# 4. MMLU-Pro
mmlu_pro = load_dataset('TIGER-Lab/MMLU-Pro')
mmlu_pro.save_to_disk('datasets/mmlu_pro')

# 5. MuSR
musr = load_dataset('TAUR-Lab/MuSR')
musr.save_to_disk('datasets/musr')
```

### Loading Saved Datasets

```python
from datasets import load_from_disk

math_ds = load_from_disk('datasets/competition_math')
gsm8k = load_from_disk('datasets/gsm8k')
mmlu_pro = load_from_disk('datasets/mmlu_pro')
musr = load_from_disk('datasets/musr')
```

## Sample Data

### MATH (competition_math) - First 3 Test Examples

**Example 1** (Level 3, Algebra):
> How many vertical asymptotes does the graph of $y=\frac{2}{x^2+x-6}$ have?

Solution: The denominator factors into $(x-2)(x+3)$. Since the numerator is always nonzero, there is a vertical asymptote whenever the denominator is 0, which occurs for $x = 2$ and $x = -3$. Answer: **2**

**Example 2** (Level 1, Algebra):
> What is the positive difference between $120\%$ of 30 and $130\%$ of 20?

Solution: $120\% \times 30 = 36$, $130\% \times 20 = 26$. Difference = **10**

**Example 3** (Level 4, Algebra):
> Find $x$ such that $\lceil x \rceil + x = \dfrac{23}{7}$. Express $x$ as a common fraction.

Solution: $x$ must be positive with decimal part $\frac{2}{7}$. Writing $x = n + \frac{2}{7}$, we get $n+1 + n + \frac{2}{7} = \frac{23}{7}$, so $n = 1$ and $x = \frac{9}{7}$.

### GSM8K - First 3 Test Examples

**Example 1**:
> Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

Answer: 16 - 3 - 4 = 9 eggs. 9 * $2 = **$18**

**Example 2**:
> A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?

Answer: 2/2 = 1 bolt white. 2 + 1 = **3 bolts**

**Example 3**:
> Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?

Answer: Cost = $130,000. Value increase = $80,000 * 1.5 = $120,000. New value = $200,000. Profit = **$70,000**

### MMLU-Pro - First 3 Test Examples

**Example 1** (Business):
> Typical advertising regulatory bodies suggest, for example that adverts must not: encourage \_\_\_, cause unnecessary \_\_\_ or \_\_\_, and must not cause \_\_\_ offence.

Answer: **I** (Unsafe practices, Distress, Fear, Serious)

**Example 2** (Business):
> Managers are entrusted to run the company in the best interest of \_\_\_. Specifically, they have a duty to act for the benefit of the company, as well as a duty of \_\_\_ and of \_\_\_.

Answer: **F**

**Example 3** (Business):
> There are two main issues associated with \_\_\_ sizing...

Answer: **J**

### MuSR - First 3 Murder Mystery Examples

**Example 1**:
> Narrative: In an adrenaline-inducing bungee jumping site, Mack's thrill-seeking adventure came to a gruesome end by a nunchaku...

Question: Who is the most likely murderer?
Choices: Mackenzie, Ana | Answer: **Mackenzie**

**Example 2**:
> (Same narrative, different perspective)

Question: Who is the most likely murderer?
Choices: Mackenzie, Ana | Answer: **Ana**

**Example 3**:
> Narrative: In the haze of neon lights, Timothy lies dead in a casino, a sai his cruel end...

Question: Who is the most likely murderer?
Choices: Harry, Rosemary | Answer: **Harry**

## Notes for Research Use

- **GPQA access**: Request access at https://huggingface.co/datasets/Idavidrein/gpqa, then authenticate with `huggingface-cli login`. Configs available: `gpqa_main` (448 questions), `gpqa_diamond` (198 questions), `gpqa_extended` (546 questions).
- **MATH difficulty levels**: Problems are rated Level 1 (easiest) to Level 5 (hardest), useful for studying how reasoning trace length varies with difficulty.
- **GSM8K annotations**: Solutions contain calculator annotations (`<<expr=result>>`) and final answers after `####`.
- **MMLU-Pro**: Has 10 options per question (A-J) vs. MMLU's 4 (A-D), reducing random guessing baseline to 10%.
- **MuSR splits**: The three splits represent different reasoning tasks, not train/test splits. All are evaluation-only.
