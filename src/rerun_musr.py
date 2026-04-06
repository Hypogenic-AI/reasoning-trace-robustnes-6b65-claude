"""Re-run only MuSR with corrected data loader."""
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from experiment import (
    run_experiment, RESULTS_DIR, SEED, MODEL, N_PER_DATASET
)
from data_loader import load_musr_sample

async def main():
    musr_samples = load_musr_sample(N_PER_DATASET, SEED)
    print(f"MuSR samples: {len(musr_samples)}")
    print(f"Answer distribution: {[s['answer'] for s in musr_samples[:10]]}")

    start = time.time()
    results = await run_experiment({"MuSR": musr_samples})
    elapsed = time.time() - start
    print(f"\nMuSR re-run completed in {elapsed/60:.1f} minutes")

    # Load existing results and replace MuSR
    existing = json.load(open(RESULTS_DIR / "raw_results.json"))
    non_musr = [r for r in existing if r["dataset"] != "MuSR"]
    all_results = non_musr + results

    with open(RESULTS_DIR / "raw_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Updated results saved ({len(all_results)} total)")

if __name__ == "__main__":
    asyncio.run(main())
