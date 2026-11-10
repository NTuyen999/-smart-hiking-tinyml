from pathlib import Path
from collections import defaultdict
import csv
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "full_dataset" / "Dataset"
DATA = ROOT / "TinyML_prepared"
REPO = ROOT / "TinyML_repo"
AUDIT = REPO / "_audit_v1"
MANIFEST = REPO / "split_users_v1.json"

WINDOW = 40
GRAVITY = 9.80665
TOLERANCE = 1e-4

FEATURES = [
    f"{axis}_{i}"
    for i in range(WINDOW)
    for axis in ("ax", "ay", "az")
]

COLUMNS = ["user", "file", "label"] + FEATURES


def load_splits():
    if not MANIFEST.exists():
        splits = {}

        for split in ("train", "validation", "test"):
            path = DATA / f"{split}.csv"
            df = pd.read_csv(path)

            splits[split] = sorted(
                df["user"].astype(int).unique().tolist()
            )

        MANIFEST.write_text(
            json.dumps(splits, indent=2),
            encoding="utf-8"
        )

    splits = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )

    sets = [
        set(splits[name])
        for name in ("train", "validation", "test")
    ]

    assert not sets[0] & sets[1]
    assert not sets[0] & sets[2]
    assert not sets[1] & sets[2]

    return splits


def read_acg(path):
    samples = []

    with path.open(
        encoding="utf-8-sig",
        errors="replace",
        newline=""
    ) as f:
        for row in csv.reader(f):
            if len(row) != 6:
                continue

            if row[5].strip() != "acg":
                continue

            try:
                values = [
                    float(value)
                    for value in row[1:4]
                ]
            except ValueError:
                continue

            if all(np.isfinite(values)):
                samples.append(values)

    return np.asarray(samples, dtype=np.float32)


def generate_windows(samples, label):
    if len(samples) < WINDOW:
        return []

    magnitude = np.linalg.norm(samples, axis=1)

    peak = int(np.argmax(
        np.abs(magnitude - GRAVITY)
    ))

    if label == 1:
        centers = [peak]
    else:
        centers = [len(samples) // 2, peak]

    starts = []

    for center in centers:
        start = max(
            0,
            min(
                center - WINDOW // 2,
                len(samples) - WINDOW
            )
        )

        if start not in starts:
            starts.append(start)

    return [
        samples[start:start + WINDOW]
        for start in starts
    ]


def reconstruct(users):
    records = []

    for label, category in ((0, "adl"), (1, "fall")):
        for user in users:
            folder = RAW / category / f"user{user}"

            if not folder.is_dir():
                raise FileNotFoundError(folder)

            for path in sorted(folder.glob("*.csv")):
                samples = read_acg(path)

                for window in generate_windows(
                    samples, label
                ):
                    records.append(
                        [user, path.name, label]
                        + window.reshape(-1).tolist()
                    )

    return pd.DataFrame(records, columns=COLUMNS)


def group_windows(df):
    groups = defaultdict(list)

    for row in df[COLUMNS].itertuples(
        index=False, name=None
    ):
        user, filename, label, *values = row

        key = (
            int(user),
            str(filename),
            int(label)
        )

        groups[key].append(
            np.asarray(values, dtype=np.float32)
        )

    return groups


def verify(original, reconstructed):
    expected = group_windows(original)
    actual = group_windows(reconstructed)

    if set(expected) != set(actual):
        raise ValueError("Different source file groups")

    matched = 0

    for key, windows in expected.items():
        available = list(actual[key])

        if len(windows) != len(available):
            raise ValueError(
                f"Different window count: {key}"
            )

        for window in windows:
            errors = [
                float(np.max(np.abs(window - item)))
                for item in available
            ]

            index = int(np.argmin(errors))

            if errors[index] > TOLERANCE:
                raise ValueError(
                    f"Window mismatch: {key}, "
                    f"error={errors[index]}"
                )

            available.pop(index)
            matched += 1

    return matched


def main():
    splits = load_splits()
    AUDIT.mkdir(parents=True, exist_ok=True)

    print("=== VERIFIED V1 PREPROCESSING ===")

    for split in ("train", "validation", "test"):
        original = pd.read_csv(
            DATA / f"{split}.csv"
        )

        reconstructed = reconstruct(
            splits[split]
        )

        matched = verify(
            original, reconstructed
        )

        reconstructed.to_csv(
            AUDIT / f"{split}_reconstructed.csv",
            index=False
        )

        print(
            f"{split}: "
            f"{matched}/{len(original)} MATCHED"
        )

    print("Split manifest:", MANIFEST)
    print("Original CSV: UNCHANGED")
    print("CNN models: UNCHANGED")
    print("V1 PREPROCESSING VERIFIED")


if __name__ == "__main__":
    main()
