# Code Dependencies for "Reasoning Trace Length as a Proxy for Robustness"

This directory contains cloned repositories used as baselines, tools, and references for investigating how reasoning trace length relates to model robustness.

---

## 1. L1: Controlling How Long a Reasoning Model Thinks With Reinforcement Learning

- **Directory:** `l1/`
- **URL:** https://github.com/cmu-l3/l1
- **Paper:** https://arxiv.org/abs/2503.04697
- **Purpose:** Provides a reinforcement learning method (LCPO) for explicitly controlling the length of a reasoning model's chain-of-thought at inference time. Trains models (L1-Exact, L1-Max) that accept a token budget and produce reasoning traces of the requested length while maintaining accuracy.
- **Key files/scripts:**
  - `main_ppo.py` -- Main RL training loop (PPO via verl framework)
  - `main_generation.py` -- Generation / inference entry point
  - `math_reward.py`, `rewards_types.py` -- Reward functions for RL training
  - `scripts/train/` -- Training launch scripts
  - `scripts/eval/eval_model_token.sh` -- Evaluation with controllable token budget
  - `scripts/data/` -- Dataset preparation (DeepScaleR, AIME, GPQA, LSAT, MMLU)
  - `config/` -- Hydra configs for training runs
- **Dependencies:** Python 3.12, flash-attn, verl (volcengine), torch, torchvision, tabulate. Full conda env in `requirements.txt`.
- **Pre-trained models:** Available on HuggingFace at `l3lab/L1-Qwen-1.5B-Exact` and `l3lab/L1-Qwen-1.5B-Max`.
- **Relevance to our research:** Directly enables experiments that vary reasoning trace length as an independent variable. By requesting different token budgets from the same model, we can measure how accuracy and robustness change as a function of thinking length -- the core question of this project.

---

## 2. SEAL: Steerable Reasoning Calibration of Large Language Models for Free

- **Directory:** `seal/`
- **URL:** https://github.com/VITA-Group/SEAL
- **Paper:** Referenced in repository (Steerable Reasoning Calibration)
- **Purpose:** Extracts steering vectors from LLM hidden states that correspond to reasoning transitions (e.g., "switch" points and "check" points in chain-of-thought). These vectors can then be injected at inference time to steer reasoning depth without retraining, providing a training-free method for controlling reasoning behavior.
- **Key files/scripts:**
  - `vector_generation.py` -- Extracts and saves steering vectors from hidden states
  - `hidden_analysis.py` -- Analyzes hidden-state representations at reasoning steps
  - `eval_MATH_steering.py` -- Evaluates MATH benchmark with steering vectors applied
  - `eval_MATH_vllm.py` -- Baseline MATH evaluation via vLLM
  - `eval_code_steering.py`, `eval_code_vllm.py` -- Code task evaluation (with/without steering)
  - `scripts/generate_vector.sh` -- Launch script for vector extraction
  - `scripts/steering.sh` -- Launch script for steered inference
  - `scripts/baseline.sh` -- Baseline evaluation script
  - `modeling_utils/` -- Custom Qwen2 model with hook support for hidden-state extraction
  - `eval_math_rule/` -- Math answer evaluation utilities
- **Dependencies:** torch, transformers, vllm, peft, datasets, evaluate (inferred from imports).
- **Relevance to our research:** Offers an alternative, gradient-free mechanism for modulating reasoning depth. Comparing SEAL's steering-vector approach with L1's RL approach lets us test whether the relationship between trace length and robustness holds across different control methods. Also provides hidden-state analysis tools useful for understanding what changes internally as traces get longer or shorter.

---

## 3. TWYN: Think When You Need -- Self-Adaptive Chain-of-Thought Learning

- **Directory:** `twyn/`
- **URL:** https://github.com/lefttt/TWYN
- **Paper:** https://arxiv.org/abs/2504.03234
- **Purpose:** Trains models via GRPO (Group Relative Policy Optimization) to adaptively decide how much reasoning to produce based on question difficulty. Uses a dual reward that balances answer correctness with response brevity, so the model learns to think more on hard problems and less on easy ones. Supports both verifiable (math) and fuzzy (open-ended) tasks.
- **Key files/scripts:**
  - `verl/` -- Modified VeRL framework with TWYN reward logic
  - `scripts/math/twyn_1.5b_8k_train_dapo_grpo.sh` -- Math RL training script
  - `scripts/alpaca/twyn_7b_train_alpaca_grpo_gpu32.sh` -- Fuzzy task (AlpacaFarm) training
  - `evaluate_alpaca.py` -- AlpacaFarm evaluation entry point
  - `recipe/` -- Training recipes
  - `pyproject.toml`, `setup.py` -- Package installation (`pip install -e .`)
- **Dependencies:** VeRL framework (bundled), accelerate, datasets, flash-attn, hydra-core, vllm<=0.6.3, ray, transformers, wandb, peft. Full list in `requirements.txt`.
- **Datasets:** DeepScaleR-Preview-Dataset, DAPO-Math-17k; evals on AIME 2024, AMC 2023, MATH 500, Minerva Math, Olympiad Bench.
- **Relevance to our research:** Demonstrates that adaptive trace length (more thinking for harder problems) can maintain accuracy while reducing average token count by ~30%. Provides a natural testbed for measuring whether shorter traces on easy problems remain equally robust, and whether the model's internal difficulty assessment correlates with robustness under perturbation.

---

## 4. LiteCoT: Concise Reasoning, Big Gains -- Pruning Long Reasoning Traces with Difficulty-Aware Prompting

- **Directory:** `litecot/`
- **URL:** https://github.com/Evanwu1125/LiteCoT
- **Paper:** https://arxiv.org/abs/2505.19716
- **Purpose:** Uses a Difficulty-Aware Prompting (DAP) pipeline to generate concise chain-of-thought training data. DeepSeek R1 first generates long CoT, then rewrites it into shorter CoT adapted to problem difficulty. Models (Liter series) are then SFT-trained on this compressed data, achieving competitive accuracy with much shorter reasoning traces.
- **Key files/scripts:**
  - `sft.py` -- Main SFT training script
  - `scripts/run_7b-math-training-short.sh` (and other sizes) -- Training launch scripts
  - `training_data/convert_parquet.py` -- Dataset preprocessing
  - `eval/open-r1/eval.sh` -- Evaluation for GPQA, AIME, MATH500 (via lighteval)
  - `eval/Qwen2.5-Math/eval.sh` -- Evaluation on 11 math benchmarks
  - `config/` -- Training configurations
  - `zero3.yaml`, `zero3_offload.yaml` -- DeepSpeed ZeRO-3 configs
- **Dependencies:** Full conda environment in `requirements.txt` (torch 2.6, transformers 4.50, deepspeed 0.16, flash-attn 2.7, accelerate, datasets, lighteval, vllm, ray, wandb, etc.).
- **Pre-trained models:** LiteCoT-1.5B through LiteCoT-32B on HuggingFace (Evanwu50020 and SmallDoge orgs).
- **Relevance to our research:** Provides models explicitly trained on compressed reasoning traces, enabling direct comparison of robustness between long-CoT and short-CoT models of the same architecture and size. The DAP pipeline's difficulty-aware compression is a concrete operationalization of the hypothesis that trace length should scale with problem difficulty -- we can test whether this holds for robustness metrics as well.

---

## 5. Rethinking The Illusion of Thinking

- **Directory:** `rethinking-illusion/`
- **URL:** https://github.com/11inaki11/Rethinking-The-Illusion-of-Thinking
- **Purpose:** Revisits and extends the experiments from Apple's "The Illusion of Thinking" paper (Shojaee et al.) which argued that Large Reasoning Models lack genuine reasoning. This repo introduces controlled experimental modifications -- stepwise Towers of Hanoi solving, filtered solvable-only River Crossing instances, and collaborative multi-LLM setups -- to probe when and how reasoning models actually fail.
- **Key files/scripts:**
  - `Hanoi_Towers/HanoiTowersSolver.py` -- Standard recursive Hanoi solver via LLM
  - `Hanoi_Towers/HanoiTowersSolverSteps.py` -- Stepwise Hanoi (controls for output length limits)
  - `Hanoi_Towers/HanoiTowersSolverConver.py` -- Collaborative two-LLM Hanoi solver
  - `Hanoi_Towers/HanoiTowersViewers.py` -- Visualization and video generation
  - `RiverCrossing/RiverCrossingSolver.py` -- River Crossing solver (solvable instances only)
  - `RiverCrossing/multipleSolution.py` -- Move-sequence validator
  - `RiverCrossing/RiverCrossingViewer.py` -- Visualization
  - `test_lm_studio_basic.py`, `test_local_llm.py` -- Local LLM testing utilities
  - Various `graphs.py`, `papersComparission.py` -- Results analysis and plotting
- **Dependencies:** google-generativeai (Gemini API), requires `GEMINI_API_KEY_HANOI` environment variable. Standard Python libraries (no requirements.txt provided).
- **Key findings:** Stepwise Hanoi works up to 8-9 disks; River Crossing solved for up to 100 couples when unsolvable cases are filtered out; token usage increases with complexity.
- **Relevance to our research:** Provides empirical evidence that reasoning trace length is a necessary but not sufficient condition for correct reasoning -- beyond a complexity threshold, longer traces do not help. The stepwise approach isolates reasoning failures from output-length limitations, which is critical for our investigation. The token-usage vs. complexity data (especially the "plateau effect" in River Crossing) directly informs the relationship between trace length and task difficulty/robustness.

---

## Summary: How These Repos Relate

| Repo | Control Method | Trace Length Manipulation | Training Required |
|------|---------------|--------------------------|-------------------|
| L1 | RL (LCPO) | Explicit token budget at inference | Yes (RL) |
| SEAL | Steering vectors | Hidden-state intervention at inference | No (training-free) |
| TWYN | RL (GRPO) | Adaptive per-question (learned) | Yes (RL) |
| LiteCoT | SFT on compressed data | Static (short CoT training data) | Yes (SFT) |
| Rethinking Illusion | Prompt engineering | Stepwise decomposition | No |

Together, these tools cover the key axes of our investigation: (1) controlling trace length via different mechanisms, (2) measuring accuracy under varying trace budgets, (3) assessing whether shorter traces degrade robustness, and (4) understanding the relationship between problem complexity and required reasoning depth.
