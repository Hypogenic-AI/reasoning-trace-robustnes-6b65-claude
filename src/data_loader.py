"""Load and sample from pre-downloaded datasets."""
import json
import random
import re
from pathlib import Path
from datasets import load_from_disk

DATASETS_DIR = Path("/workspaces/reasoning-trace-robustnes-6b65-claude/datasets")

def load_math_sample(n=100, seed=42):
    """Load stratified sample from MATH test set (across difficulty levels)."""
    ds = load_from_disk(str(DATASETS_DIR / "competition_math"))["test"]
    random.seed(seed)
    # Stratify by level
    by_level = {}
    for item in ds:
        lvl = item.get("level", "unknown")
        by_level.setdefault(lvl, []).append(item)

    samples = []
    per_level = max(1, n // len(by_level))
    for lvl in sorted(by_level.keys()):
        items = by_level[lvl]
        k = min(per_level, len(items))
        samples.extend(random.sample(items, k))

    # Trim or pad to exactly n
    random.shuffle(samples)
    samples = samples[:n]

    result = []
    for item in samples:
        # Extract boxed answer from solution
        answer = extract_math_answer(item.get("solution", ""))
        result.append({
            "id": f"math_{len(result)}",
            "dataset": "MATH",
            "question": item["problem"],
            "answer": answer,
            "level": item.get("level", "unknown"),
            "type": item.get("type", "unknown"),
        })
    return result

def extract_math_answer(solution: str) -> str:
    """Extract \\boxed{...} answer from MATH solution."""
    # Find the last \boxed{...}
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, solution)
    if matches:
        return matches[-1].strip()
    return solution.strip().split('\n')[-1].strip()

def load_gsm8k_sample(n=100, seed=42):
    """Load sample from GSM8K test set."""
    ds = load_from_disk(str(DATASETS_DIR / "gsm8k"))["test"]
    random.seed(seed)
    indices = random.sample(range(len(ds)), min(n, len(ds)))

    result = []
    for idx in indices:
        item = ds[idx]
        # GSM8K answer is after ####
        answer_text = item.get("answer", "")
        answer = answer_text.split("####")[-1].strip() if "####" in answer_text else answer_text.strip()
        result.append({
            "id": f"gsm8k_{len(result)}",
            "dataset": "GSM8K",
            "question": item["question"],
            "answer": answer,
            "level": "grade_school",
            "type": "arithmetic",
        })
    return result

def load_mmlu_pro_sample(n=100, seed=42):
    """Load sample from MMLU-Pro test set."""
    ds = load_from_disk(str(DATASETS_DIR / "mmlu_pro"))["test"]
    random.seed(seed)
    indices = random.sample(range(len(ds)), min(n, len(ds)))

    result = []
    for idx in indices:
        item = ds[idx]
        options = item.get("options", [])
        answer_idx = item.get("answer_index", item.get("answer", 0))

        # Format as multiple choice
        option_letters = "ABCDEFGHIJ"
        options_text = "\n".join(f"{option_letters[i]}. {opt}" for i, opt in enumerate(options) if i < len(option_letters))
        question = f"{item['question']}\n\n{options_text}"

        if isinstance(answer_idx, int) and answer_idx < len(option_letters):
            answer = option_letters[answer_idx]
        else:
            answer = str(answer_idx)

        result.append({
            "id": f"mmlu_pro_{len(result)}",
            "dataset": "MMLU-Pro",
            "question": question,
            "answer": answer,
            "level": "professional",
            "type": item.get("category", "unknown"),
        })
    return result

def load_musr_sample(n=100, seed=42):
    """Load sample from MuSR dataset."""
    random.seed(seed)
    result = []
    for split_name in ["murder_mysteries", "object_placements", "team_allocation"]:
        try:
            ds = load_from_disk(str(DATASETS_DIR / "musr"))[split_name]
            per_split = max(1, n // 3)
            indices = random.sample(range(len(ds)), min(per_split, len(ds)))
            for idx in indices:
                item = ds[idx]
                choices = item.get("choices", [])
                narrative = item.get("narrative", "")
                question_text = item.get("question", "")

                option_letters = "ABCDEFGHIJ"
                options_text = "\n".join(f"{option_letters[i]}. {c}" for i, c in enumerate(choices) if i < len(option_letters))
                full_question = f"{narrative}\n\n{question_text}\n\n{options_text}" if narrative else f"{question_text}\n\n{options_text}"

                # Use answer_index to get the letter
                answer_idx = item.get("answer_index", 0)
                answer = option_letters[answer_idx] if isinstance(answer_idx, int) and answer_idx < len(option_letters) else str(answer_idx)

                result.append({
                    "id": f"musr_{len(result)}",
                    "dataset": "MuSR",
                    "question": full_question,
                    "answer": answer,
                    "level": "multistep",
                    "type": split_name,
                })
        except Exception as e:
            print(f"Warning: Could not load MuSR split {split_name}: {e}")

    random.shuffle(result)
    return result[:n]

def load_all_samples(n_per_dataset=100, seed=42):
    """Load samples from all datasets."""
    return {
        "MATH": load_math_sample(n_per_dataset, seed),
        "GSM8K": load_gsm8k_sample(n_per_dataset, seed),
        "MMLU-Pro": load_mmlu_pro_sample(n_per_dataset, seed),
        "MuSR": load_musr_sample(n_per_dataset, seed),
    }

if __name__ == "__main__":
    samples = load_all_samples(n_per_dataset=5, seed=42)
    for name, items in samples.items():
        print(f"\n{'='*60}")
        print(f"{name}: {len(items)} samples")
        if items:
            print(f"Example Q: {items[0]['question'][:200]}...")
            print(f"Example A: {items[0]['answer']}")
