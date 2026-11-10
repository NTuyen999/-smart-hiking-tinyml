from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "TinyML_prepared"
AUDIT = ROOT / "TinyML_repo" / "_audit_v1"

TOLERANCE = 0.0001


def load_groups(path):
    df = pd.read_csv(path)
    groups = defaultdict(list)

    for row in df.itertuples(index=False, name=None):
        user, filename, label, *values = row

        key = (int(user), str(filename), int(label))

        groups[key].append(
            np.asarray(values, dtype=np.float64)
        )

    return groups, len(df)


def compare(split):
    original, total = load_groups(
        DATA / f"{split}.csv"
    )

    reconstructed, regenerated = load_groups(
        AUDIT / f"{split}_reconstructed.csv"
    )

    matched = 0
    unmatched = []
    count_differences = []

    for key in sorted(
        set(original) | set(reconstructed)
    ):
        original_windows = original.get(key, [])
        available = list(reconstructed.get(key, []))

        if len(original_windows) != len(available):
            count_differences.append(
                (key, len(original_windows), len(available))
            )

        for window in original_windows:
            if not available:
                unmatched.append((key, None))
                continue

            errors = [
                float(np.max(np.abs(window - candidate)))
                for candidate in available
            ]

            index = int(np.argmin(errors))
            error = errors[index]

            if error <= TOLERANCE:
                matched += 1
                available.pop(index)
            else:
                unmatched.append((key, error))

    print(f"\n=== {split.upper()} ===")
    print("Original:", total)
    print("Reconstructed:", regenerated)
    print("Matched with tolerance:", matched)
    print("Unmatched:", total - matched)

    print("\nFILE COUNT DIFFERENCES")

    if not count_differences:
        print("None")
    else:
        for key, old_count, new_count in count_differences[:15]:
            print(
                f"User {key[0]} | {key[1]} | "
                f"Label {key[2]} | "
                f"Original {old_count} | "
                f"Reconstructed {new_count}"
            )

    print("\nFIRST UNMATCHED WINDOWS")

    if not unmatched:
        print("None")
    else:
        for key, error in unmatched[:8]:
            print(
                f"User {key[0]} | {key[1]} | "
                f"Label {key[2]} | "
                f"Best max error: {error}"
            )


print("=== V1 DATA ALIGNMENT DIAGNOSTIC ===")

for split in ("train", "validation"):
    compare(split)

print("\nOriginal datasets: UNCHANGED")
print("CNN models: UNCHANGED")
print("DIAGNOSTIC COMPLETE")
