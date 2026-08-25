"""Predict dropout risk for one or more new students from the command line.

Usage:
    python main.py                          # trains the model and saves output/model.pkl
    python predict.py --input student.csv   # predicts for the student(s) in student.csv

The input CSV must have the same feature columns as the training dataset
(everything except the "Target" column) - one row per student. Values can
use either a comma or semicolon delimiter, matching the UCI dataset format.
"""
import argparse
import csv
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TARGET_CLASSES

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "model.pkl")


class _FeatureRow:
    """Minimal stand-in for a Student, just enough for EncodedFeatureMatrix.transform."""

    def __init__(self, features, student_id="new_student"):
        self.features = features
        self.student_id = student_id


def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: No trained model found at {MODEL_PATH}")
        print("Run `python main.py` first to train and save the model.")
        sys.exit(1)
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def read_students_csv(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        first_line = f.readline()
        f.seek(0)
        delimiter = ";" if (";" in first_line and "," not in first_line) else ","

        reader = csv.DictReader(f, delimiter=delimiter)
        rows = []
        for row in reader:
            processed = {}
            for key, value in row.items():
                if key is None or key == "Target":
                    continue
                try:
                    processed[key] = float(value)
                except (ValueError, TypeError):
                    processed[key] = value
            rows.append(processed)
        return rows


def predict_students(feature_rows, artifacts):
    encoder = artifacts["encoder"]
    model = artifacts["model"]
    all_features = artifacts["all_features"]

    missing = [f for f in all_features if f not in feature_rows[0]]
    if missing:
        print(f"ERROR: Input CSV is missing {len(missing)} required column(s), e.g.: {missing[:5]}")
        sys.exit(1)

    rows = [_FeatureRow(r, student_id=f"new_student_{i}") for i, r in enumerate(feature_rows)]
    X = encoder.transform(rows, all_features, partition_label="predict")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    dropout_idx = TARGET_CLASSES.index("Dropout")

    results = []
    for pred, proba in zip(predictions, probabilities):
        results.append({
            "predicted_outcome": pred,
            "dropout_probability": proba[dropout_idx],
            "at_risk": pred == "Dropout",
        })
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Predict dropout risk for new students using the trained XGBoost model.")
    parser.add_argument(
        "--input", required=True,
        help="Path to a CSV file with one row per student (same feature columns as the training dataset).")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    artifacts = load_artifacts()
    feature_rows = read_students_csv(args.input)

    if not feature_rows:
        print("No student rows found in input file.")
        sys.exit(1)

    results = predict_students(feature_rows, artifacts)

    print("\n" + "=" * 70)
    print("  DROPOUT RISK PREDICTIONS")
    print("=" * 70)
    for i, r in enumerate(results):
        print(f"\n  Student row {i + 1}:")
        print(f"    Predicted outcome:    {r['predicted_outcome']}")
        print(f"    Dropout probability:  {r['dropout_probability']:.4f}")
        print(f"    At risk of dropout:   {r['at_risk']}")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
