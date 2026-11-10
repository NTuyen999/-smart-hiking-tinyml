from pathlib import Path
from collections import Counter
import csv
import math
import re

RAW_DIR = Path(
    r"D:\SmartHiking_TinyML\full_dataset\Dataset"
)

DATA_DIR = Path(__file__).resolve().parents[2] / "TinyML_prepared"

WINDOW_SIZE = 40
GRAVITY = 9.80665

OLD_TRAIN = DATA_DIR / "train.csv"
VALIDATION = DATA_DIR / "validation.csv"
TEST = DATA_DIR / "test.csv"
OUTPUT = DATA_DIR / "train_v2.csv"


def read_users(path):
    with path.open(
        encoding="utf-8-sig",
        newline=""
    ) as f:
        return {
            int(row["user"])
            for row in csv.DictReader(f)
        }


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
                    float(v) for v in row[1:4]
                ]
            except ValueError:
                continue

            if all(math.isfinite(v) for v in values):
                samples.append(values)

    return samples


def make_starts(samples, activity):
    if len(samples) < WINDOW_SIZE:
        return []

    max_start = len(samples) - WINDOW_SIZE

    starts = sorted({
        round(max_start * i / 3)
        for i in range(4)
    })

    scores = []

    for i, sample in enumerate(samples):
        magnitude = math.sqrt(
            sum(value * value for value in sample)
        )

        scores.append(
            (abs(magnitude - GRAVITY), i)
        )

    scores.sort(reverse=True)

    hard_limit = (
        4 if activity in (3, 4, 16)
        else 1
    )

    added = 0

    for _, peak in scores:
        start = max(
            0,
            min(
                peak - WINDOW_SIZE // 2,
                max_start
            )
        )

        if all(
            abs(start - previous) >= WINDOW_SIZE // 2
            for previous in starts
        ):
            starts.append(start)
            added += 1

        if added >= hard_limit:
            break

    return sorted(set(starts))


def main():
    train_users = read_users(OLD_TRAIN)
    val_users = read_users(VALIDATION)
    test_users = read_users(TEST)

    assert not train_users & val_users
    assert not train_users & test_users
    assert not val_users & test_users

    headers = ["user", "file", "label"]

    headers += [
        f"{axis}_{i}"
        for i in range(WINDOW_SIZE)
        for axis in ("ax", "ay", "az")
    ]

    counts = Counter()

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as out:
        writer = csv.writer(out)
        writer.writerow(headers)

        for path in sorted(
            (RAW_DIR / "adl").rglob("*.csv")
        ):
            user = int(
                path.parent.name.replace("user", "")
            )

            if user not in train_users:
                continue

            match = re.search(
                r"_adl(\d+)\.csv$",
                path.name
            )

            if not match:
                continue

            activity = int(match.group(1))
            samples = read_acg(path)

            starts = make_starts(
                samples, activity
            )

            if not starts:
                counts["short_adl_files"] += 1
                continue

            for start in starts:
                window = samples[
                    start:start + WINDOW_SIZE
                ]

                features = [
                    round(value, 5)
                    for sample in window
                    for value in sample
                ]

                writer.writerow(
                    [user, path.name, 0] + features
                )

                counts["adl_windows"] += 1
                counts[f"adl_{activity:02d}"] += 1

        with OLD_TRAIN.open(
            encoding="utf-8-sig",
            newline=""
        ) as f:
            reader = csv.DictReader(f)

            for row in reader:
                if int(row["label"]) != 1:
                    continue

                writer.writerow([
                    row[column]
                    for column in headers
                ])

                counts["fall_windows"] += 1

    print("=== TRAIN V2 PREPARATION ===")
    print("Train users:", len(train_users))
    print("Validation users:", len(val_users))
    print("Test users:", len(test_users))
    print("ADL windows:", counts["adl_windows"])
    print("FALL candidates:", counts["fall_windows"])
    print(
        "Skipped short ADL files:",
        counts["short_adl_files"]
    )

    for activity in (3, 4, 16):
        print(
            f"ADL {activity:02d}:",
            counts[f"adl_{activity:02d}"]
        )

    print("Output:", OUTPUT)
    print("Validation and test: UNCHANGED")
    print("PREPARATION COMPLETE")


if __name__ == "__main__":
    main()

