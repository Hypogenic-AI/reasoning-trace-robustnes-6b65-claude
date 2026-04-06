"""
Main experiment: Systematically vary reasoning trace length and measure
accuracy + robustness across ID and OOD benchmarks.

Uses GPT-4.1 via OpenAI API with budget-forcing prompts.
"""
import asyncio
import json
import os
import re
import time
import random
import numpy as np
from pathlib import Path
from openai import AsyncOpenAI
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────

SEED = 42
MODEL = "gpt-4.1"
RESULTS_DIR = Path("/workspaces/reasoning-trace-robustnes-6b65-claude/results")
N_PER_DATASET = 80  # questions per dataset
MAX_CONCURRENT = 15  # concurrent API calls

# Budget conditions: (name, max_tokens, system_prompt_suffix)
BUDGET_CONDITIONS = [
    ("no_cot", 150, "Answer directly with NO reasoning or explanation. Just give the final answer."),
    ("short", 300, "Think briefly in 1-2 sentences, then give your answer. Keep reasoning very concise."),
    ("medium", 800, "Think step by step in a moderate amount of detail (3-5 steps), then give your answer."),
    ("long", 1500, "Think very carefully and thoroughly step by step, showing all your work in detail before giving your answer."),
    ("unconstrained", 3000, "Think step by step, taking as much space as you need to reason through this problem thoroughly and carefully."),
]

random.seed(SEED)
np.random.seed(SEED)

# ── Prompting ──────────────────────────────────────────────────────────────

def make_system_prompt(budget_suffix: str, dataset: str) -> str:
    """Create system prompt for a given budget condition and dataset."""
    if dataset in ("MMLU-Pro", "MuSR"):
        answer_format = "State your final answer as a single letter (A, B, C, etc.) on a new line prefixed with 'ANSWER:'"
    elif dataset == "GSM8K":
        answer_format = "State your final numerical answer on a new line prefixed with 'ANSWER:'"
    else:  # MATH
        answer_format = "State your final answer on a new line prefixed with 'ANSWER:' (use LaTeX if needed)"

    return f"""You are a helpful assistant solving questions.
{budget_suffix}

After your reasoning (if any), {answer_format}

Also, on a separate line, state your confidence in your answer as a number between 0 and 100, prefixed with 'CONFIDENCE:'

Example format:
[your reasoning if applicable]
ANSWER: [your answer]
CONFIDENCE: [0-100]"""


def make_user_prompt(question: str) -> str:
    return question


# ── API Calling ────────────────────────────────────────────────────────────

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
semaphore = asyncio.Semaphore(MAX_CONCURRENT)

async def call_api(system_prompt: str, user_prompt: str, max_tokens: int) -> dict:
    """Call GPT-4.1 with retry logic."""
    async with semaphore:
        for attempt in range(5):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.0,  # deterministic
                    seed=SEED,
                )
                msg = response.choices[0].message.content or ""
                usage = response.usage
                return {
                    "response": msg,
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "finish_reason": response.choices[0].finish_reason,
                }
            except Exception as e:
                if attempt < 4:
                    wait = 2 ** attempt + random.random()
                    print(f"  API error (attempt {attempt+1}): {e}. Retrying in {wait:.1f}s...")
                    await asyncio.sleep(wait)
                else:
                    print(f"  API error (final): {e}")
                    return {"response": "", "prompt_tokens": 0, "completion_tokens": 0, "finish_reason": "error"}


# ── Answer Extraction & Scoring ────────────────────────────────────────────

def extract_answer(response: str) -> str:
    """Extract the ANSWER: field from response."""
    match = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: last line
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    return lines[-1] if lines else ""

def extract_confidence(response: str) -> float:
    """Extract CONFIDENCE: field from response."""
    match = re.search(r'CONFIDENCE:\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
    if match:
        return float(match.group(1)) / 100.0  # normalize to [0,1]
    return 0.5  # default

def normalize_answer(s: str) -> str:
    """Normalize an answer string for comparison."""
    s = s.strip().lower()
    # Remove LaTeX wrappers
    s = re.sub(r'^\$+|\$+$', '', s)
    s = re.sub(r'^\\text\{|\}$', '', s)
    s = re.sub(r'^\\boxed\{|\}$', '', s)
    # Remove trailing periods
    s = s.rstrip('.')
    # Normalize whitespace
    s = re.sub(r'\s+', ' ', s)
    return s

def score_answer(predicted: str, gold: str, dataset: str) -> bool:
    """Check if predicted answer matches gold."""
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)

    if pred_norm == gold_norm:
        return True

    # For MC datasets, check single letter match
    if dataset in ("MMLU-Pro", "MuSR"):
        pred_letter = re.search(r'\b([A-J])\b', predicted.upper())
        if pred_letter and pred_letter.group(1) == gold.upper():
            return True

    # Numeric comparison for GSM8K
    if dataset == "GSM8K":
        try:
            pred_num = float(re.sub(r'[,$%]', '', pred_norm))
            gold_num = float(re.sub(r'[,$%]', '', gold_norm))
            return abs(pred_num - gold_num) < 0.01
        except ValueError:
            pass

    # Substring check for MATH (some answers are complex expressions)
    if gold_norm in pred_norm or pred_norm in gold_norm:
        return True

    return False


# ── Main Experiment Loop ───────────────────────────────────────────────────

async def run_experiment(samples_by_dataset: dict) -> list:
    """Run all budget conditions on all datasets."""
    all_results = []
    total_calls = sum(len(s) for s in samples_by_dataset.values()) * len(BUDGET_CONDITIONS)
    print(f"Total API calls planned: {total_calls}")

    for dataset_name, samples in samples_by_dataset.items():
        for budget_name, max_tokens, budget_suffix in BUDGET_CONDITIONS:
            print(f"\n▸ {dataset_name} × {budget_name} ({len(samples)} questions, max_tokens={max_tokens})")

            system_prompt = make_system_prompt(budget_suffix, dataset_name)
            tasks = []
            for item in samples:
                user_prompt = make_user_prompt(item["question"])
                tasks.append((item, call_api(system_prompt, user_prompt, max_tokens)))

            # Run batch
            responses = await asyncio.gather(*[t[1] for t in tasks])

            correct = 0
            for (item, _), resp in zip(tasks, responses):
                pred = extract_answer(resp["response"])
                conf = extract_confidence(resp["response"])
                is_correct = score_answer(pred, item["answer"], dataset_name)
                if is_correct:
                    correct += 1

                all_results.append({
                    "dataset": dataset_name,
                    "budget": budget_name,
                    "question_id": item["id"],
                    "level": item["level"],
                    "type": item["type"],
                    "gold_answer": item["answer"],
                    "predicted_answer": pred,
                    "confidence": conf,
                    "correct": is_correct,
                    "completion_tokens": resp["completion_tokens"],
                    "prompt_tokens": resp["prompt_tokens"],
                    "finish_reason": resp["finish_reason"],
                    "full_response": resp["response"],
                })

            acc = correct / len(samples) if samples else 0
            print(f"  → Accuracy: {acc:.1%} ({correct}/{len(samples)})")

    return all_results


async def main():
    from data_loader import load_all_samples

    print("=" * 70)
    print("EXPERIMENT: Reasoning Trace Length vs OOD Robustness")
    print(f"Model: {MODEL} | Seed: {SEED} | N/dataset: {N_PER_DATASET}")
    print("=" * 70)

    # Load data
    print("\nLoading datasets...")
    samples = load_all_samples(n_per_dataset=N_PER_DATASET, seed=SEED)
    for name, items in samples.items():
        print(f"  {name}: {len(items)} samples")

    # Run experiment
    start = time.time()
    results = await run_experiment(samples)
    elapsed = time.time() - start
    print(f"\nExperiment completed in {elapsed/60:.1f} minutes")
    print(f"Total results: {len(results)}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "raw_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {output_path}")

    # Save config
    config = {
        "model": MODEL,
        "seed": SEED,
        "n_per_dataset": N_PER_DATASET,
        "budget_conditions": [(name, max_tok) for name, max_tok, _ in BUDGET_CONDITIONS],
        "datasets": list(samples.keys()),
        "total_samples": sum(len(v) for v in samples.values()),
        "elapsed_seconds": elapsed,
    }
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    results = asyncio.run(main())
