"""
Experiment 3: Adaptive trace length selection.
Two-pass approach: first get a quick estimate with short reasoning,
then decide whether to invest in longer reasoning based on confidence.
"""
import asyncio
import json
import os
import sys
import time
import random
import numpy as np
from pathlib import Path
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).parent))
from experiment import (
    call_api, extract_answer, extract_confidence, score_answer,
    make_system_prompt, SEED, MODEL, RESULTS_DIR, MAX_CONCURRENT,
    BUDGET_CONDITIONS
)
from data_loader import load_all_samples

random.seed(SEED)
np.random.seed(SEED)

# Adaptive strategy: if confidence < threshold on short pass, escalate to long
CONFIDENCE_THRESHOLD = 0.75


async def run_adaptive(samples_by_dataset: dict) -> list:
    """Two-pass adaptive strategy: short first, escalate if low confidence."""
    results = []
    # Conditions from BUDGET_CONDITIONS
    short_budget = BUDGET_CONDITIONS[1]  # short (300 tokens)
    long_budget = BUDGET_CONDITIONS[3]   # long (1500 tokens)

    for dataset_name, samples in samples_by_dataset.items():
        print(f"\n▸ Adaptive: {dataset_name} ({len(samples)} questions)")

        # Pass 1: short reasoning for all
        pass1_tasks = []
        sys_prompt = make_system_prompt(short_budget[2], dataset_name)
        for item in samples:
            pass1_tasks.append((item, call_api(sys_prompt, item["question"], short_budget[1])))

        pass1_responses = await asyncio.gather(*[t[1] for t in pass1_tasks])

        # Decide which need escalation
        escalate_indices = []
        pass1_results = []
        for i, ((item, _), resp) in enumerate(zip(pass1_tasks, pass1_responses)):
            conf = extract_confidence(resp["response"])
            pred = extract_answer(resp["response"])
            pass1_results.append({
                "response": resp, "pred": pred, "conf": conf, "item": item
            })
            if conf < CONFIDENCE_THRESHOLD:
                escalate_indices.append(i)

        print(f"  Pass 1: {len(escalate_indices)}/{len(samples)} need escalation (conf < {CONFIDENCE_THRESHOLD})")

        # Pass 2: long reasoning for low-confidence items
        if escalate_indices:
            sys_prompt_long = make_system_prompt(long_budget[2], dataset_name)
            pass2_tasks = []
            for idx in escalate_indices:
                item = pass1_results[idx]["item"]
                pass2_tasks.append((idx, call_api(sys_prompt_long, item["question"], long_budget[1])))

            pass2_responses = await asyncio.gather(*[t[1] for t in pass2_tasks])

            # Update with pass 2 results
            for (idx, _), resp in zip(pass2_tasks, pass2_responses):
                pass1_results[idx]["response"] = resp
                pass1_results[idx]["pred"] = extract_answer(resp["response"])
                pass1_results[idx]["conf"] = extract_confidence(resp["response"])
                pass1_results[idx]["escalated"] = True

        # Score all
        correct = 0
        total_tokens = 0
        for r in pass1_results:
            item = r["item"]
            is_correct = score_answer(r["pred"], item["answer"], dataset_name)
            if is_correct:
                correct += 1
            tokens = r["response"]["completion_tokens"]
            total_tokens += tokens

            results.append({
                "dataset": dataset_name,
                "budget": "adaptive",
                "question_id": item["id"],
                "level": item["level"],
                "type": item["type"],
                "gold_answer": item["answer"],
                "predicted_answer": r["pred"],
                "confidence": r["conf"],
                "correct": is_correct,
                "completion_tokens": tokens,
                "prompt_tokens": r["response"]["prompt_tokens"],
                "escalated": r.get("escalated", False),
                "finish_reason": r["response"]["finish_reason"],
            })

        acc = correct / len(samples) if samples else 0
        avg_tok = total_tokens / len(samples) if samples else 0
        escalated_pct = len(escalate_indices) / len(samples) * 100
        print(f"  → Accuracy: {acc:.1%} | Avg tokens: {avg_tok:.0f} | Escalated: {escalated_pct:.0f}%")

    return results


async def main():
    print("=" * 70)
    print("EXPERIMENT: Adaptive Trace Length Selection")
    print(f"Model: {MODEL} | Threshold: {CONFIDENCE_THRESHOLD}")
    print("=" * 70)

    samples = load_all_samples(n_per_dataset=80, seed=SEED)
    start = time.time()
    results = await run_adaptive(samples)
    elapsed = time.time() - start

    print(f"\nAdaptive experiment completed in {elapsed/60:.1f} minutes")

    # Save
    with open(RESULTS_DIR / "adaptive_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved {len(results)} results")

    # Quick comparison with fixed budgets
    existing = json.load(open(RESULTS_DIR / "raw_results.json"))
    print("\n── Comparison: Adaptive vs Fixed Budgets ──")
    for dataset in ["MATH", "GSM8K", "MMLU-Pro", "MuSR"]:
        adaptive = [r for r in results if r["dataset"] == dataset]
        adapt_acc = np.mean([r["correct"] for r in adaptive])
        adapt_tok = np.mean([r["completion_tokens"] for r in adaptive])

        print(f"\n{dataset}:")
        print(f"  Adaptive: acc={adapt_acc:.1%}, avg_tok={adapt_tok:.0f}")
        for budget in ["no_cot", "short", "medium", "long", "unconstrained"]:
            fixed = [r for r in existing if r["dataset"] == dataset and r["budget"] == budget]
            if fixed:
                facc = np.mean([r["correct"] for r in fixed])
                ftok = np.mean([r["completion_tokens"] for r in fixed])
                print(f"  {budget:>15}: acc={facc:.1%}, avg_tok={ftok:.0f}")

    return results


if __name__ == "__main__":
    asyncio.run(main())
