# Reasoning Trace Length as a Proxy for Robustness

Systematic investigation of how reasoning chain length affects LLM out-of-distribution robustness, using GPT-4.1 with budget-forcing prompts across four benchmarks.

## Key Findings

- **Longer reasoning helps in-distribution but with diminishing returns:** MATH accuracy increases from 46% (no CoT) to 78% (unconstrained), but the gain from medium to unconstrained is not statistically significant.
- **The length-accuracy curve is non-monotonic on hard OOD tasks:** On MuSR, "short" reasoning (5%) is worse than no reasoning (10%), and accuracy peaks at "long" (22%) before dropping slightly at unconstrained (21%).
- **Model confidence is catastrophically miscalibrated OOD:** Confidence-accuracy correlation is rho=0.63 on MATH but rho~0 on MuSR, making confidence-based adaptive strategies unreliable where they're needed most.
- **Adaptive trace length saves tokens in-distribution but fails OOD:** A two-pass strategy (short first, escalate if uncertain) achieves "long"-level accuracy on MATH with 32% fewer tokens, but only 8% accuracy on MuSR due to overconfidence.
- **Easy tasks are insensitive to trace length:** GSM8K accuracy is 91-94% regardless of budget condition.

## Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv add openai datasets numpy scipy matplotlib seaborn pandas tqdm

# Run experiments (requires OPENAI_API_KEY)
python -m src.experiment          # Main experiment (~7 min, ~$15)
python -m src.rerun_musr          # MuSR re-run with fixed loader
python -m src.adaptive_experiment # Adaptive strategy (~1 min, ~$3)
python -m src.analyze             # Analysis and plots
```

## File Structure

```
├── REPORT.md              # Full research report with results
├── README.md              # This file
├── planning.md            # Research plan and motivation
├── src/
│   ├── data_loader.py     # Dataset loading and sampling
│   ├── experiment.py      # Main experiment (budget-forcing)
│   ├── adaptive_experiment.py  # Adaptive two-pass strategy
│   ├── rerun_musr.py      # MuSR re-run script
│   └── analyze.py         # Statistical analysis and plotting
├── results/
│   ├── raw_results.json   # All experiment outputs
│   ├── adaptive_results.json
│   ├── config.json
│   ├── analysis_summary.json
│   ├── statistical_tests.json
│   └── plots/             # 7 visualization files
├── datasets/              # Pre-downloaded (MATH, GSM8K, MMLU-Pro, MuSR)
├── papers/                # 27 downloaded research papers
├── code/                  # Cloned repos (L1, SEAL, TWYN, LiteCoT)
└── literature_review.md   # Comprehensive lit review (28 papers)
```

See [REPORT.md](REPORT.md) for full methodology, results, and analysis.
