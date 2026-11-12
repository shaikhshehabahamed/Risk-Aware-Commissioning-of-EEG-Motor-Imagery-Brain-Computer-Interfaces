"""
Script 02: Compute FBCSP-LDA LORO-CV viability labels for Task 2
           (left vs right hand motor imagery, runs R04/R08/R12).

This is the primary study target: hand MI is the most clinically relevant BCI
paradigm, with the strongest resting-state EEG correlates (mu lateralisation,
sensorimotor coherence).

Reads:  results/subject_list.json
Writes: results/viability_labels.json   <- Task 2 (primary)

Runtime: ~3-6 minutes.
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTS, TASK2_RUNS, VIABILITY_THRESHOLD
from src.mi_pipeline import compute_viability

# ── Load subject list ──────────────────────────────────────────────────────────
subj_path = RESULTS / "subject_list.json"
if not subj_path.exists():
    raise FileNotFoundError(f"Run 01_check_subjects.py first. Missing: {subj_path}")

with open(subj_path) as f:
    subj_data = json.load(f)

subjects = subj_data["included_subjects"]
print(f"Computing Task-2 (hand MI) viability labels for {len(subjects)} subjects")
print(f"Runs: {TASK2_RUNS}  |  Threshold: Fold-3 >= {VIABILITY_THRESHOLD:.0%}")
print()

# ── Run FBCSP-LDA LORO-CV for each subject ─────────────────────────────────────
t0 = time.time()
per_subject = []
n_viable = 0
n_error  = 0

for i, subj in enumerate(subjects, 1):
    res = compute_viability(subj, TASK2_RUNS)
    per_subject.append(res)

    if res["error"]:
        n_error += 1
        status = f"ERROR: {res['error']}"
    else:
        n_viable += int(res["viable"])
        f3 = res["fold3_accuracy"]
        label = "VIABLE" if res["viable"] else "non-viable"
        status = f"fold3={f3:.3f}  mean={res['mean_loro_acc']:.3f}  {label}"

    print(f"  [{i:3d}/{len(subjects)}] {subj}: {status}")

print(f"\nDone in {(time.time()-t0)/60:.1f} min")
print(f"Viable: {n_viable}  |  Non-viable: {len(subjects)-n_viable-n_error}  |  Errors: {n_error}")

# ── Distribution summary ───────────────────────────────────────────────────────
accs = np.array([r["fold3_accuracy"] for r in per_subject
                 if r["fold3_accuracy"] is not None])
print(f"\nFold-3 accuracy: min={accs.min():.3f}  max={accs.max():.3f}  "
      f"mean={accs.mean():.3f}  std={accs.std():.3f}")
borderline = int(((accs >= 0.50) & (accs < 0.70)).sum())
print(f"Borderline subjects (50-70%): {borderline} / {len(accs)} "
      f"({borderline/len(accs):.0%})")

# ── Save ───────────────────────────────────────────────────────────────────────
output = {
    "task":               "Task2_HandMI",
    "runs":               TASK2_RUNS,
    "viability_threshold": VIABILITY_THRESHOLD,
    "n_subjects":         len(subjects),
    "n_viable":           n_viable,
    "n_nonviable":        len(subjects) - n_viable - n_error,
    "n_errors":           n_error,
    "per_subject":        per_subject,
}

out_path = RESULTS / "viability_labels.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved -> {out_path}")
print(f"Total: {(time.time()-t0)/60:.1f} minutes")
