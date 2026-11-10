from pathlib import Path
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import confusion_matrix, roc_auc_score

SEED = 2026
WINDOW_SIZE = 40
GRAVITY = 9.80665

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = Path(__file__).resolve().parents[2] / "TinyML_prepared"
REPO_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_DIR / "models"
RESULT_DIR = REPO_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

FEATURES = [
    f"{axis}_{i}"
    for i in range(WINDOW_SIZE)
    for axis in ("ax", "ay", "az")
]


def load_dataset(filename):
    path = DATA_DIR / filename
    df = pd.read_csv(path)

    required = ["user", "file", "label"] + FEATURES
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    raw = df[FEATURES].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)

    if not np.isfinite(raw).all():
        raise ValueError("Invalid sensor values")

    x = raw.reshape(-1, WINDOW_SIZE, 3)
    x = np.clip(x / GRAVITY, -8.0, 8.0)

    users = set(df["user"].astype(int))

    return x, y, users, df


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(40, 3)),
        tf.keras.layers.Conv1D(
            12, 5, padding="same", activation="relu"
        ),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(
            16, 3, padding="same", activation="relu"
        ),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(12, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def evaluate(name, y, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predicted, labels=[0, 1]
    ).ravel()

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return {
        "model": name,
        "threshold": threshold,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def main():
    x_train, y_train, train_users, _ = load_dataset(
        "train_v2.csv"
    )

    x_val, y_val, val_users, val_df = load_dataset(
        "validation.csv"
    )

    if train_users & val_users:
        raise ValueError("Train/Validation user leakage")

    n_adl = int((y_train == 0).sum())
    n_fall = int((y_train == 1).sum())
    total = len(y_train)

    weights = {
        0: total / (2 * n_adl),
        1: total / (2 * n_fall),
    }

    print("Train:", x_train.shape)
    print("Validation:", x_val.shape)

    model = build_model()
    model.summary()

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=40,
        batch_size=32,
        class_weight=weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )

    probabilities = model.predict(
        x_val, verbose=0
    ).reshape(-1)

    model.save(
        str(MODEL_DIR / "fall_cnn_pilot_v2.keras")
    )

    report = pd.DataFrame({
        "user": val_df["user"].to_numpy(),
        "file": val_df["file"].to_numpy(),
        "true_label": y_val,
        "predicted_label": (
            probabilities >= 0.5
        ).astype(int),
        "fall_probability": probabilities,
    })

    report.to_csv(
        RESULT_DIR / "validation_predictions_v2.csv",
        index=False,
    )

    rows = []

    for threshold in [0.50, 0.65, 0.80]:
        rows.append(
            evaluate(
                "CNN_V2",
                y_val,
                probabilities,
                threshold,
            )
        )

    pd.DataFrame(rows).to_csv(
        RESULT_DIR / "v2_threshold_results.csv",
        index=False,
    )

    print("ROC-AUC:", roc_auc_score(y_val, probabilities))
    print("MODEL SAVED")
    print("TRAINING COMPLETE")


if __name__ == "__main__":
    main()

