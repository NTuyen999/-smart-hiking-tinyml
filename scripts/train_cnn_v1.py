from pathlib import Path
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

SEED = 2026
WINDOW_SIZE = 40
GRAVITY = 9.80665

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "TinyML_prepared"
REPO_DIR = ROOT / "TinyML_repo"
MODEL_DIR = REPO_DIR / "models"
RESULT_DIR = REPO_DIR / "results"

FEATURES = [
    f"{axis}_{i}"
    for i in range(WINDOW_SIZE)
    for axis in ("ax", "ay", "az")
]


def load_dataset(filename):
    path = DATA_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = ["user", "file", "label"] + FEATURES
    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    raw = df[FEATURES].to_numpy(dtype=np.float32)
    labels = df["label"].to_numpy(dtype=np.int32)

    if not np.isfinite(raw).all():
        raise ValueError("Invalid sensor values")

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Invalid labels")

    x = raw.reshape(-1, WINDOW_SIZE, 3)
    x = np.clip(x / GRAVITY, -8.0, 8.0)

    users = set(df["user"].astype(int))

    return x, labels, users, df


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(40, 3)),

        tf.keras.layers.Conv1D(
            12, 5,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.MaxPooling1D(2),

        tf.keras.layers.Conv1D(
            16, 3,
            padding="same",
            activation="relu"
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
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    return model


def main():
    x_train, y_train, train_users, _ = load_dataset(
        "train.csv"
    )

    x_val, y_val, val_users, val_df = load_dataset(
        "validation.csv"
    )

    if train_users & val_users:
        raise ValueError("Train/Validation user leakage")

    n_adl = int(np.sum(y_train == 0))
    n_fall = int(np.sum(y_train == 1))
    total = len(y_train)

    weights = {
        0: total / (2 * n_adl),
        1: total / (2 * n_fall)
    }

    print("=== CNN V1 ===")
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
                restore_best_weights=True
            )
        ],
        verbose=2
    )

    probabilities = model.predict(
        x_val, verbose=0
    ).reshape(-1)

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    print("\n=== VALIDATION ===")
    print(confusion_matrix(y_val, predictions))

    print(classification_report(
        y_val, predictions,
        zero_division=0
    ))

    print(
        "ROC-AUC:",
        roc_auc_score(y_val, probabilities)
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Luu ten khac de khong ghi de mo hinh V1 goc.
    model.save(
        str(MODEL_DIR / "fall_cnn_v1_retrained.keras")
    )

    pd.DataFrame({
        "user": val_df["user"].to_numpy(),
        "file": val_df["file"].to_numpy(),
        "true_label": y_val,
        "predicted_label": predictions,
        "fall_probability": probabilities
    }).to_csv(
        RESULT_DIR / "validation_v1_retrained.csv",
        index=False
    )

    print("\nCNN V1 RETRAINING COMPLETE")


if __name__ == "__main__":
    main()
