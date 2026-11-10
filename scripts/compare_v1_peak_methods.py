from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(r"D:\SmartHiking_TinyML")
DATA = ROOT / "TinyML_prepared"

sys.path.insert(
    0, str(ROOT / "TinyML_repo" / "scripts")
)

from prepare_dataset_v1 import read_acg

W = 40
G = 9.80665
TOL = 0.0001

FEATURES = [
    f"{axis}_{i}"
    for i in range(W)
    for axis in ("ax", "ay", "az")
]


def windows(samples, label, method):
    magnitude = np.linalg.norm(samples, axis=1)

    if method == "raw_peak":
        peak = int(np.argmax(magnitude))
    elif method == "gravity_deviation":
        peak = int(
            np.argmax(np.abs(magnitude - G))
        )
    else:
        raise ValueError(method)

    centers = (
        [peak] if label == 1
        else [len(samples) // 2, peak]
    )

    starts = []

    for center in centers:
        start = max(
            0,
            min(center - W // 2, len(samples) - W)
        )

        # Loai bo cua so trung vi tri
        if start not in starts:
            starts.append(start)

    return [
        samples[start:start + W].reshape(-1)
        for start in starts
    ]


def match_count(expected, generated):
    pairs = []

    for i, a in enumerate(expected):
        for j, b in enumerate(generated):
            error = float(
                np.max(np.abs(a - b))
            )
            pairs.append((error, i, j))

    pairs.sort()

    used_expected = set()
    used_generated = set()
    matched = 0

    for error, i, j in pairs:
        if error > TOL:
            break

        if i in used_expected:
            continue

        if j in used_generated:
            continue

        used_expected.add(i)
        used_generated.add(j)
        matched += 1

    return matched


print("=== V1 PEAK METHOD COMPARISON ===")

for split in ("train", "validation", "test"):
    df = pd.read_csv(
        DATA / f"{split}.csv"
    )

    print(f"\nSPLIT: {split}")
    print("Original:", len(df))

    for method in [
        "raw_peak",
        "gravity_deviation"
    ]:
        matched = 0
        generated_count = 0
        problem_files = []

        for (user, filename, label), group in (
            df.groupby(
                ["user", "file", "label"],
                sort=False
            )
        ):
            user = int(user)
            label = int(label)

            category = (
                "fall" if label == 1 else "adl"
            )

            path = (
                ROOT
                / "full_dataset"
                / "Dataset"
                / category
                / f"user{user}"
                / str(filename)
            )

            if not path.is_file():
                raise FileNotFoundError(path)

            samples = read_acg(path)

            expected = group[FEATURES].to_numpy(
                dtype=np.float32
            )

            if len(samples) < W:
                generated = []
            else:
                generated = windows(
                    samples, label, method
                )

            count = match_count(
                expected, generated
            )

            matched += count
            generated_count += len(generated)

            if (
                count != len(expected)
                or len(expected) != len(generated)
            ):
                problem_files.append(
                    (user, filename, label)
                )

        print(
            f"{method}: "
            f"matched={matched}/{len(df)}, "
            f"generated={generated_count}, "
            f"problem_files={len(problem_files)}"
        )

        if problem_files:
            print(
                "Examples:",
                problem_files[:8]
            )

print("\nOriginal CSV: UNCHANGED")
print("CNN models: UNCHANGED")
print("PEAK COMPARISON COMPLETE")
