# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project investigating **Reasoning Trace Length as a Proxy for Robustness** in large language models. The hypothesis posits a non-monotonic relationship between reasoning trace length and OOD robustness, where adaptive trace length controls outperform fixed-length approaches.

---

## Papers

**Total papers downloaded: 27**

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | When More is Less | Wu et al. | 2025 | when_more_is_less_2502_07266.pdf | Inverted U-curve; optimal length theory |
| 2 | Between Underthinking and Overthinking | Su et al. | 2025 | underthinking_overthinking_2505_00127.pdf | Two failure modes; asymmetric calibration |
| 3 | Illusion of Thinking (Rethinking) | Dellibarda Varela et al. | 2025 | illusion_of_thinking_2507_01231.pdf | Token usage as uncertainty signal |
| 4 | What Makes a Good Reasoning Chain | Jiang et al. | 2025 | good_reasoning_chain_2505_22148.pdf | Structure > length for prediction |
| 5 | Is Long-to-Short a Free Lunch | Yang et al. | 2025 | long_to_short_2506_19492.pdf | Hidden robustness costs (ICBENCH) |
| 6 | L1: Controlling Reasoning Length | Aggarwal & Welleck | 2025 | l1_controlling_reasoning_2503_04697.pdf | RL-based LCPO; OOD generalization |
| 7 | SEAL | Chen et al. | 2025 | seal_steerable_reasoning_2504_07986.pdf | Training-free steering vectors |
| 8 | Think When You Need | Yang et al. | 2025 | think_when_needed_2504_03234.pdf | Adaptive CoT via GRPO |
| 9 | Concise Reasoning, Big Gains | Wu et al. | 2025 | concise_reasoning_big_gains_2505_19716.pdf | Difficulty-aware pruning |
| 10 | Constraint-Rectified Training | Wu et al. | 2026 | constraint_rectified_2602_12526.pdf | Constrained optimization framework |
| 11 | Early Stopping CoT | Mao et al. | 2025 | early_stopping_cot_2509_14004.pdf | Convergence-based stopping |
| 12 | Thought Anchors | Bogdan et al. | 2025 | thought_anchors_2506_19143.pdf | Counterfactual step importance |
| 13 | Beyond Semantics | Valmeekam et al. | 2025 | beyond_semantics_2505_13775.pdf | Corrupted traces generalize OOD |
| 14 | Detect Errors in Long CoT | He et al. | 2025 | detect_errors_cot_2502_19361.pdf | Error detection (DeltaBench) |
| 15 | Stop Overthinking Survey | Sui et al. | 2025 | stop_overthinking_survey_2503_16419.pdf | Survey of efficient reasoning |
| 16 | Don't Overthink It Survey | Yue et al. | 2025 | dont_overthink_survey_2508_02120.pdf | R1-style model survey |
| 17 | Efficient Reasoning Survey | Qu et al. | 2025 | survey_efficient_reasoning_2503_21614.pdf | Comprehensive survey |
| 18 | Output Length DeepSeek Safety | Li et al. | 2025 | output_length_deepseek_2503_01923.pdf | Length and safety interaction |
| 19 | CoT Fails Clinical | Various | 2025 | cot_fails_clinical_2509_21933.pdf | Domain-specific CoT failures |
| 20 | Reasoning as Compression | Massoli et al. | 2026 | reasoning_as_compression_2603_08462.pdf | Information bottleneck theory |
| 21 | Overthink Math | Srivastava et al. | 2025 | overthink_math_2507_04023.pdf | Overthinking Score; 28% collapse |
| 22 | Token to Action | Various | 2025 | token_to_action_2505_23059.pdf | State machine for overthinking |
| 23 | CoT-X | Various | 2025 | cot_x_2511_05747.pdf | Cross-model CoT transfer |
| 24 | Incentivizing Reasoning | Various | 2025 | incentivizing_reasoning_2510_20867.pdf | Reasoning in audio LLMs |
| 25 | TrimR | Various | 2020 | trimr_2008_12863.pdf | Verifier-based compression |
| 26 | MuSR | Sprague et al. | 2023 | musr_2310_16049.pdf | Multistep soft reasoning benchmark |
| 27 | OOD Robustness LLM | Various | 2024 | ood_robustness_llm_2405_11431.pdf | OOD generalization of LLMs |

See `papers/README.md` for detailed descriptions.

---

## Datasets

**Total datasets downloaded: 4 (+ 1 gated)**

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| MATH | EleutherAI/hendrycks_math | 12,500 | Competition math (7 subjects, 5 levels) | datasets/competition_math/ | Primary dataset; level-stratified |
| GSM8K | openai/gsm8k | 8,792 | Grade school math | datasets/gsm8k/ | Easier baseline |
| MMLU-Pro | TIGER-Lab/MMLU-Pro | 12,102 | Multi-task understanding | datasets/mmlu_pro/ | OOD evaluation |
| MuSR | TAUR-Lab/MuSR | 756 | Multistep soft reasoning | datasets/musr/ | OOD evaluation |
| GPQA | Idavidrein/gpqa | ~450 | Graduate-level Q&A | *Gated* | Requires HF approval |

See `datasets/README.md` for download instructions and loading code.

---

## Code Repositories

**Total repositories cloned: 5**

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| L1 | github.com/cmu-l3/l1 | RL-based length control (LCPO) | code/l1/ | Key baseline; DeepScaleR-based |
| SEAL | github.com/VITA-Group/SEAL | Training-free steering vectors | code/seal/ | Requires transformers, torch |
| TWYN | github.com/lefttt/TWYN | Self-adaptive CoT via GRPO | code/twyn/ | Xiaohongshu |
| LiteCoT | github.com/Evanwu1125/LiteCoT | Difficulty-aware CoT pruning/SFT | code/litecot/ | 100K pruned data approach |
| Rethinking Illusion | github.com/11inaki11/Rethinking-The-Illusion-of-Thinking | Puzzle-based reasoning analysis | code/rethinking-illusion/ | Gemini API required |

See `code/README.md` for detailed descriptions.

**Additional awesome-lists (not cloned, reference only):**
- https://github.com/Eclipsess/Awesome-Efficient-Reasoning-LLMs (Stop Overthinking survey)
- https://github.com/yuelinan/Awesome-Efficient-R1-style-LRMs (Don't Overthink It survey)
- https://github.com/XiaoYee/Awesome_Efficient_LRM_Reasoning (Efficient Reasoning survey)

---

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder service (diligent mode) yielding 108 papers
2. Filtered to 67 papers with relevance >= 2, focused on 27 with relevance 3
3. Resolved Semantic Scholar IDs to arXiv IDs via API + direct arXiv search
4. Downloaded all 27 unique PDFs from arXiv
5. Deep-read 6 critical papers (all chunks), skimmed 15 more (page 1)

### Selection Criteria
- Papers directly studying reasoning trace length, CoT efficiency, or overthinking/underthinking
- Papers with OOD robustness or generalization findings
- Papers providing adaptive length control methods (potential baselines)
- Surveys providing comprehensive field overview
- Established benchmarks used across the field

### Challenges Encountered
- Semantic Scholar API rate limiting (429s) required fallback to direct arXiv search
- GPQA dataset is gated on HuggingFace (requires approval)
- Original `hendrycks/competition_math` HF repo no longer exists; used EleutherAI mirror
- Some papers (OThink-R1, ESTAR, Adaptive CoT Length) could not be resolved to arXiv IDs

### Gaps and Workarounds
- GPQA: Documented download instructions; consider using GPQA-Diamond subset if approval delayed
- AIME 2024/2025: Small dataset (~30 problems each); available in L1's codebase
- LiveCodeBench: Not downloaded as primary dataset; available via HuggingFace if needed

---

## Recommendations for Experiment Design

### 1. Primary Datasets
- **MATH** (in-distribution): Use difficulty levels 1-5 to study how optimal trace length varies with difficulty
- **GSM8K** (easy baseline): Test whether trace length matters less for simpler tasks
- **MMLU-Pro** (OOD): Knowledge-heavy tasks where reasoning may not help
- **MuSR** (OOD): Multistep reasoning requiring different skills than math

### 2. Baseline Methods
- **Unconstrained**: Full CoT generation (upper bound on length)
- **L1 (LCPO)**: RL-trained length control at various budgets (code available)
- **SEAL**: Training-free steering (code available)
- **Budget forcing**: Simple token budget via prompting
- **NoThinking**: Skip reasoning entirely (lower bound)

### 3. Evaluation Metrics
- **Accuracy** at controlled trace lengths (in-distribution and OOD)
- **Robustness gap**: Accuracy_ID - Accuracy_OOD as function of trace length
- **Token efficiency**: Accuracy / token_count
- **Behavioral consistency**: If feasible, measure via ICBENCH-style metrics
- **Uncertainty calibration**: Entropy/perplexity vs trace length

### 4. Code to Adapt/Reuse
- **L1**: Directly usable for length-controlled generation at various budgets
- **SEAL**: Can generate outputs at different trace lengths via steering strength
- **LiteCoT**: Provides difficulty-aware prompting templates
- **TWYN**: Adaptive reasoning framework

### 5. Experimental Design Suggestions
1. Generate responses at 5+ controlled trace lengths (e.g., 256, 512, 1024, 2048, 4096 tokens)
2. Evaluate each on both ID (MATH) and OOD (GPQA, MMLU-Pro, MuSR) benchmarks
3. Plot accuracy vs trace length for each, looking for inverted U-shapes
4. Compare robustness gap across trace lengths — hypothesis: moderate lengths show smallest gap
5. Test adaptive methods (L1, SEAL) vs fixed-length baselines
6. Analyze whether uncertainty signals can predict the optimal adaptive length
