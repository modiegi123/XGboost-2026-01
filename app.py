"""Streamlit demo for the from-scratch XGBoost student dropout risk pipeline.

Wraps the existing pure-Python pipeline (predict.py / pipeline.py / models/) with
an interactive UI. No scikit-learn, no xgboost package, no pandas/numpy — the
boosting algorithm, decision trees, SMOTE balancing, and evaluation metrics are
all implemented in plain Python; this file just adds the web interface around it.
"""
import csv
import io
import json
import os
import random
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TARGET_CLASSES, DATA_PATH
from predict import load_artifacts, predict_students, _FeatureRow

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODES = {
    "Full Assessment": {
        "key": "full",
        "predictions_csv": "output/predictions.csv",
        "blurb": "Uses every available feature, including both semesters' academic results.",
    },
    "After Semester 1": {
        "key": "semester1",
        "predictions_csv": "output/predictions_semester1.csv",
        "blurb": "Uses enrollment features plus 1st-semester academic performance — before semester 2 exams.",
    },
    "At Enrollment": {
        "key": "enrollment",
        "predictions_csv": "output/predictions_enrollment.csv",
        "blurb": "Uses only features known before classes start — the earliest possible warning.",
    },
}

st.set_page_config(page_title="Student Dropout Risk Predictor", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner="Loading trained model...")
def get_artifacts(mode_key):
    return load_artifacts(mode_key)


@st.cache_data
def load_sample_rows():
    path = os.path.join(BASE_DIR, DATA_PATH)
    with open(path, encoding="utf-8-sig") as f:
        first = f.readline()
        f.seek(0)
        delim = ";" if (";" in first and "," not in first) else ","
        reader = csv.DictReader(f, delimiter=delim)
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


def parse_uploaded_csv(uploaded_file):
    text = uploaded_file.getvalue().decode("utf-8-sig")
    first_line = text.splitlines()[0] if text else ""
    delim = ";" if (";" in first_line and "," not in first_line) else ","
    rows = []
    for row in csv.DictReader(io.StringIO(text), delimiter=delim):
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


@st.cache_data
def compute_metrics(predictions_csv_rel):
    """Recompute accuracy/precision/recall/F1/confusion matrix from a stored
    held-out predictions CSV — no retraining needed."""
    from models.performance_metrics import PerformanceMetrics

    path = os.path.join(BASE_DIR, predictions_csv_rel)
    y_true, y_pred = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            y_true.append(row["actual_outcome"])
            y_pred.append(row["predicted_outcome"])

    metrics = PerformanceMetrics()
    metrics.compute(y_true, y_pred, y_proba=None, classes=TARGET_CLASSES)
    return metrics


@st.cache_data
def load_experiment_log():
    path = os.path.join(BASE_DIR, "output", "experiment_log.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        log = json.load(f)
    return log[-1] if log else None


# Flags mirror dashboard.py's early-warning logic, adapted for the web UI.
_FLAGS_ENROLLMENT = [
    ("Debtor", lambda v: float(v) == 1, "Has outstanding debt"),
    ("Tuition fees up to date", lambda v: float(v) == 0, "Tuition fees not paid"),
    ("Scholarship holder", lambda v: float(v) == 0, "No scholarship"),
    ("Age at enrollment", lambda v: float(v) > 30, "Enrolled at age over 30"),
    ("Displaced", lambda v: float(v) == 1, "Displaced student"),
]
_FLAGS_SEMESTER1 = _FLAGS_ENROLLMENT + [
    ("Curricular units 1st sem (enrolled)", lambda v: float(v) == 0, "Not enrolled in any units in semester 1"),
    ("Curricular units 1st sem (approved)", lambda v: float(v) < 3, "Passed fewer than 3 units in semester 1"),
    ("Curricular units 1st sem (grade)", lambda v: float(v) < 10, "Average grade below 10 in semester 1"),
]
_FLAGS_FULL = _FLAGS_SEMESTER1 + [
    ("Curricular units 2nd sem (enrolled)", lambda v: float(v) == 0, "Not enrolled in any units in semester 2"),
    ("Curricular units 2nd sem (approved)", lambda v: float(v) < 3, "Passed fewer than 3 units in semester 2"),
    ("Curricular units 2nd sem (grade)", lambda v: float(v) < 10, "Average grade below 10 in semester 2"),
]
_FLAGS_BY_MODE = {"enrollment": _FLAGS_ENROLLMENT, "semester1": _FLAGS_SEMESTER1, "full": _FLAGS_FULL}


@st.cache_data
def load_feature_lookup():
    """student_id ('S00000', ...) -> raw feature dict, in dataset row order."""
    rows = load_sample_rows()
    return {f"S{i:05d}": row for i, row in enumerate(rows)}


@st.cache_data
def load_at_risk_students(predictions_csv_rel, mode_key):
    path = os.path.join(BASE_DIR, predictions_csv_rel)
    if not os.path.exists(path):
        return []
    lookup = load_feature_lookup()
    flags = _FLAGS_BY_MODE[mode_key]

    flagged = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["at_risk"].lower() != "true":
                continue
            feats = lookup.get(row["student_id"], {})
            seen, flag_labels = set(), []
            for col, check, label in flags:
                if label not in seen and col in feats and check(feats[col]):
                    flag_labels.append(label)
                    seen.add(label)
            flagged.append({
                "student_id": row["student_id"],
                "dropout_probability": float(row["dropout_probability"]),
                "flags": flag_labels or ["No specific flags"],
            })
    return sorted(flagged, key=lambda r: r["dropout_probability"], reverse=True)


st.title("🎓 Student Dropout Risk Predictor")
st.caption(
    "A from-scratch XGBoost (gradient boosted trees) classifier for predicting student "
    "dropout risk. No scikit-learn, no `xgboost` package — the boosting algorithm, decision "
    "trees, SMOTE class balancing, and evaluation metrics are all implemented in plain Python."
)

tab_predict, tab_warning, tab_perf, tab_about = st.tabs(
    ["🔮 Predict", "⚠️ Early Warning & Interventions", "📊 Model Performance", "ℹ️ About"]
)

# --------------------------------------------------------------------------- Predict
with tab_predict:
    mode_label = st.selectbox("Assessment point", list(MODES.keys()), index=0)
    mode = MODES[mode_label]
    st.caption(mode["blurb"])

    source = st.radio(
        "Students to predict",
        ["Try sample students from the dataset", "Upload a CSV"],
        horizontal=True,
    )

    feature_rows = None
    labels = None

    if source == "Try sample students from the dataset":
        rows = load_sample_rows()
        n = st.slider("How many random students?", 1, 20, 5)
        if st.button("🎲 Pick random students", type="primary"):
            st.session_state["picked_idx"] = random.sample(range(len(rows)), n)
        idx = st.session_state.get("picked_idx", list(range(n)))
        feature_rows = [rows[i] for i in idx]
        labels = [f"Student #{i}" for i in idx]
    else:
        uploaded = st.file_uploader(
            "CSV with one row per student (same feature columns as the training dataset, no Target column)",
            type="csv",
        )
        if uploaded is not None:
            feature_rows = parse_uploaded_csv(uploaded)
            labels = [f"Row {i + 1}" for i in range(len(feature_rows))]

    if feature_rows:
        try:
            artifacts = get_artifacts(mode["key"])
            results = predict_students(feature_rows, artifacts)
        except SystemExit:
            st.error(
                "This input doesn't have the columns this model expects. "
                "Try the sample students, or check your CSV's columns."
            )
        else:
            table = [
                {
                    "Student": label,
                    "Predicted outcome": r["predicted_outcome"],
                    "Dropout probability": round(r["dropout_probability"], 4),
                    "At risk": "⚠️ Yes" if r["at_risk"] else "No",
                }
                for label, r in zip(labels, results)
            ]
            st.dataframe(table, width='stretch', hide_index=True)

            at_risk_count = sum(1 for r in results if r["at_risk"])
            c1, c2 = st.columns(2)
            c1.metric("Students assessed", len(results))
            c2.metric("Flagged at risk of dropout", at_risk_count)

            st.bar_chart(
                {label: r["dropout_probability"] for label, r in zip(labels, results)},
            )
    else:
        st.info("Pick random students or upload a CSV to see predictions.")

# --------------------------------------------------------------------------- Early Warning & Interventions
with tab_warning:
    st.caption(
        "Students from the held-out test set flagged as at-risk of dropping out, with the "
        "specific warning signs that triggered the flag — so an advisor knows *why*, not just *who*."
    )
    mode_label_ew = st.selectbox("Assessment point", list(MODES.keys()), index=0, key="ew_mode")
    mode_ew = MODES[mode_label_ew]

    flagged = load_at_risk_students(mode_ew["predictions_csv"], mode_ew["key"])

    if not flagged:
        st.info("No flagged students found for this assessment point.")
    else:
        st.write(f"**{len(flagged)} students flagged** out of the held-out test set.")
        flagged_table = [
            {
                "Student ID": r["student_id"],
                "Dropout probability": round(r["dropout_probability"], 4),
                "Warning flags": "  |  ".join(r["flags"]),
            }
            for r in flagged
        ]
        st.dataframe(flagged_table, width='stretch', hide_index=True)

        st.subheader("Log an intervention")
        student_ids = [r["student_id"] for r in flagged]
        picked = st.selectbox("Student", student_ids, key="intervene_student")
        picked_record = next(r for r in flagged if r["student_id"] == picked)
        st.caption("Flags: " + "  |  ".join(picked_record["flags"]))

        note = st.text_area("Intervention note", placeholder="e.g. Scheduled meeting with academic advisor on...")
        if st.button("💾 Save intervention", type="primary"):
            if note.strip():
                st.session_state.setdefault("interventions", []).append({
                    "Student ID": picked,
                    "Dropout probability": round(picked_record["dropout_probability"], 4),
                    "Warning flags": "  |  ".join(picked_record["flags"]),
                    "Note": note.strip(),
                })
                st.success(f"Intervention logged for {picked}.")
            else:
                st.warning("Enter a note before saving.")

        st.subheader("Intervention history")
        st.caption(
            "Kept for this browser session only — this is a shared public demo, so notes aren't "
            "written to permanent storage (no cross-visitor mix-ups)."
        )
        history = st.session_state.get("interventions", [])
        if history:
            st.dataframe(list(reversed(history)), width='stretch', hide_index=True)
        else:
            st.info("No interventions logged yet this session.")

# --------------------------------------------------------------------------- Model Performance
with tab_perf:
    mode_label_perf = st.selectbox(
        "Assessment point", list(MODES.keys()), index=0, key="perf_mode"
    )
    mode_perf = MODES[mode_label_perf]

    if mode_perf["key"] == "full":
        exp = load_experiment_log()
        metrics = compute_metrics(mode_perf["predictions_csv"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{exp['metrics']['accuracy']:.2%}" if exp else f"{metrics.accuracy:.2%}")
        c2.metric("AUC-ROC", f"{exp['metrics']['auc_roc']:.4f}" if exp else "—")
        c3.metric("Dropout recall", f"{metrics.get_dropout_recall():.2%}")
        c4.metric("Dropout F1", f"{metrics.f1_per_class.get('Dropout', 0):.4f}")
    else:
        metrics = compute_metrics(mode_perf["predictions_csv"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{metrics.accuracy:.2%}")
        c2.metric("Dropout recall", f"{metrics.get_dropout_recall():.2%}")
        c3.metric("Dropout F1", f"{metrics.f1_per_class.get('Dropout', 0):.4f}")
        st.caption("AUC-ROC isn't shown here — only the per-class probability used for "
                    "dropout flagging is stored for this model, not the full 3-class vector.")

    st.subheader("Confusion matrix")
    st.caption("Rows = actual outcome, columns = predicted outcome (held-out test set).")
    cm_table = [
        {"Actual": actual, **{pred: metrics.confusion_matrix.get((actual, pred), 0) for pred in metrics.classes}}
        for actual in metrics.classes
    ]
    st.dataframe(cm_table, width='stretch', hide_index=True)

    st.subheader("Per-class precision / recall / F1")
    pr_table = [
        {
            "Class": cls,
            "Precision": round(metrics.precision_per_class.get(cls, 0), 4),
            "Recall": round(metrics.recall_per_class.get(cls, 0), 4),
            "F1": round(metrics.f1_per_class.get(cls, 0), 4),
        }
        for cls in metrics.classes
    ]
    st.dataframe(pr_table, width='stretch', hide_index=True)

    st.subheader("Top 10 features by importance")
    artifacts_perf = get_artifacts(mode_perf["key"])
    feature_names = artifacts_perf["encoder"].get_feature_names()
    importance = artifacts_perf["model"].get_feature_importance(feature_names)
    top = dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:10])
    st.bar_chart(top, horizontal=True)

# --------------------------------------------------------------------------- About
with tab_about:
    st.markdown(
        """
### Enhancing Transparency and Early Detection of At-Risk Students in Higher Education

**Author:** Josephine Morudu — University of Johannesburg, Academy of Computer Science
Software Engineering.

This project implements a gradient-boosted-trees (XGBoost-style) classifier **entirely
from scratch** in plain Python — no `scikit-learn`, no `xgboost` package, no `numpy` or
`pandas`. That includes:

- The boosting algorithm and decision trees themselves
- One-hot + Min-Max feature encoding
- SMOTE oversampling to balance the training classes
- Accuracy / precision / recall / F1 / AUC-ROC evaluation

It predicts whether a student will **Dropout**, stay **Enrolled**, or **Graduate**, using
the ["Predict Students' Dropout and Academic Success"](https://archive.ics.uci.edu/dataset/697)
dataset (Realinho et al., 2022) — 4,424 student records across 35 academic, socioeconomic,
and macroeconomic features.

Three models are trained so a student can be flagged **as early as possible**:

1. **At Enrollment** — before any classes start
2. **After Semester 1** — once first-semester results are in
3. **Full Assessment** — using all available data, including semester 2

**Source code:** [github.com/modiegi123/XGboost-2026-01](https://github.com/modiegi123/XGboost-2026-01)
        """
    )
