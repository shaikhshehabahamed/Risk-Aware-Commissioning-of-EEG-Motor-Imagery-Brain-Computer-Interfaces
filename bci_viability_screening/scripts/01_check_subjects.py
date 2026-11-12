"""
Script 01: Run five inclusion checks on all 109 subjects.
Output: results/subject_list.json

Runtime: ~10-15 minutes (dominated by loading and epoching Task-4 runs).
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_RAW, RESULTS
from src.subject_checks import check_subject


class _JSONEncoder(json.JSONEncoder):
    """Handle numpy int/bool types in JSON serialisation."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def main():
    subjects = sorted(
        d.name for d in DATA_RAW.iterdir()
        if d.is_dir() and d.name.startswith("S")
    )
    n_total = len(subjects)
    print(f"Found {n_total} subjects in {DATA_RAW}")
    print("=" * 65)

    records = []
    t0 = time.time()

    for i, subj in enumerate(subjects, 1):
        elapsed = time.time() - t0
        eta = (elapsed / i) * (n_total - i) if i > 1 else 0
        print(f"[{i:3d}/{n_total}] {subj}  "
              f"(elapsed {elapsed/60:.1f} min, ETA {eta/60:.1f} min) ... ",
              end="", flush=True)

        r = check_subject(subj)
        records.append(r)

        if r["included"]:
            print("INCLUDED")
        else:
            print(f"EXCLUDED (check {r['failed_check']}): {r['reason']}")

    # ── Summary ────────────────────────────────────────────────────────────────
    included = [r["subject"] for r in records if r["included"]]
    excluded = [r for r in records if not r["included"]]

    print("\n" + "=" * 65)
    print(f"Total:    {n_total}")
    print(f"Included: {len(included)}")
    print(f"Excluded: {len(excluded)}")

    fail_counts = Counter(r["failed_check"] for r in excluded)
    for check_num in sorted(fail_counts):
        print(f"  Failed check {check_num}: {fail_counts[check_num]} subject(s)")
        reasons = [r["reason"] for r in excluded if r["failed_check"] == check_num]
        for reason in reasons:
            print(f"    {reason}")

    # ── Save ───────────────────────────────────────────────────────────────────
    output = {
        "n_total": n_total,
        "n_included": len(included),
        "n_excluded": len(excluded),
        "included_subjects": included,
        "exclusion_counts_by_check": {str(k): v for k, v in fail_counts.items()},
        "per_subject": records,
    }

    out_path = RESULTS / "subject_list.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=_JSONEncoder)

    print(f"\nSaved -> {out_path}")
    print(f"Total runtime: {(time.time() - t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
