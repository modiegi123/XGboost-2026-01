"""End-to-end pipeline: load data, train a from-scratch XGBoost classifier,
and predict which students are at risk of dropping out."""
import os
import json
import pickle
import random
from datetime import datetime

from config import (
    RANDOM_SEED, TEST_SIZE, TARGET_CLASSES,
    MIN_ACCURACY, MIN_DROPOUT_RECALL, N_FOLDS, XGBOOST_PARAMS,
    ENROLLMENT_FEATURES, SEMESTER1_FEATURES
)
from models.encoded_feature_matrix import EncodedFeatureMatrix
from models.smote_sampler import SMOTESampler
from models.xgboost_classifier import XGBoostClassifier
from models.performance_metrics import PerformanceMetrics


class Pipeline:
    """Loads student data, trains an XGBoost classifier, and predicts dropout risk."""

    def __init__(self, students, output_dir="output"):
        self.students = students
        self.output_dir = output_dir
        self.rng = random.Random(RANDOM_SEED)

        self.encoder = None
        self.model = None
        self.metrics = None

        self.train_students = []
        self.test_students = []
        self.train_X = []
        self.train_y = []
        self.test_X = []
        self.test_y = []
        self.feature_names_encoded = []
        self.all_features = []

        self.predictions = []
        self.probabilities = []

        os.makedirs(output_dir, exist_ok=True)

    def run(self):
        """Execute the complete pipeline."""
        print("\n" + "=" * 70)
        print("  STUDENT DROPOUT PREDICTION - XGBOOST PIPELINE")
        print("=" * 70)

        self._stage_0_split_data()
        self._stage_1_encode()
        self._stage_2_balance_classes()
        self._stage_3_train_model()
        self._stage_4_predict_and_evaluate()
        self._stage_5_feature_importance()
        self._save_predictions_csv()
        self._save_model_artifacts()
        self._log_experiment()
        self._stage_6_enrollment_model()
        self._stage_7_semester1_model()

        print("\n" + "=" * 70)
        print("  PIPELINE COMPLETE")
        print(f"  All outputs saved to: {self.output_dir}/")
        print("=" * 70 + "\n")

    def _stage_0_split_data(self):
        """Stratified train-test split."""
        print("\n[STAGE 0] Stratified Train-Test Split")
        print(f"  Total students: {len(self.students)}")
        print(f"  Test size: {TEST_SIZE} | Random seed: {RANDOM_SEED}")

        class_groups = {}
        for student in self.students:
            class_groups.setdefault(student.outcome, []).append(student)

        self.train_students = []
        self.test_students = []

        for outcome, group in class_groups.items():
            self.rng.shuffle(group)
            n_test = max(1, int(len(group) * TEST_SIZE))
            self.test_students.extend(group[:n_test])
            self.train_students.extend(group[n_test:])

        self.rng.shuffle(self.train_students)
        self.rng.shuffle(self.test_students)

        print(f"  Training set: {len(self.train_students)} students")
        print(f"  Test set:     {len(self.test_students)} students")

        for split_name, split_data in [("Train", self.train_students),
                                        ("Test", self.test_students)]:
            counts = {}
            for s in split_data:
                counts[s.outcome] = counts.get(s.outcome, 0) + 1
            dist = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
            print(f"  {split_name} distribution: {dist}")

    def _stage_1_encode(self):
        """Feature encoding and normalisation."""
        print("\n[STAGE 1] Feature Encoding and Normalisation")

        all_features = list(self.students[0].features.keys())
        self.all_features = all_features
        # Only one-hot encode low-cardinality features to keep dimensions manageable
        categorical_features = []
        for feat in all_features:
            values = set(s.features.get(feat) for s in self.students
                         if s.features.get(feat) is not None)
            if 2 <= len(values) <= 5:
                all_int = all(isinstance(v, float) and v == int(v)
                              for v in values if isinstance(v, (int, float)))
                if all_int:
                    categorical_features.append(feat)

        print(f"  Total features: {len(all_features)}")
        print(f"  Categorical features: {len(categorical_features)}")
        print(f"  Numerical features: {len(all_features) - len(categorical_features)}")

        # Fit on training data only to avoid test-set leakage
        self.encoder = EncodedFeatureMatrix(random_seed=RANDOM_SEED)
        self.train_X = self.encoder.fit_transform(
            self.train_students, all_features,
            categorical_features=categorical_features,
            partition_label="train"
        )
        self.train_y = [s.outcome for s in self.train_students]

        # Transform test set using training-fitted params
        self.test_X = self.encoder.transform(
            self.test_students, all_features, partition_label="test"
        )
        self.test_y = [s.outcome for s in self.test_students]

        self.feature_names_encoded = self.encoder.get_feature_names()
        print(f"  Encoded feature dimensions: {len(self.feature_names_encoded)}")
        print(f"  Imputation applied: {self.encoder.imputation_applied}")

    def _stage_2_balance_classes(self):
        """Balance the training set with SMOTE so the classifier isn't
        dominated by the majority outcome class."""
        print("\n[STAGE 2] SMOTE Class Balancing")

        smote = SMOTESampler(k_neighbours=5, random_seed=RANDOM_SEED)
        self.train_X, self.train_y = smote.fit_resample(self.train_X, self.train_y)

        balance_ok = smote.verify_class_balance(self.train_y)
        print(f"  Balanced training set: {len(self.train_X)} samples")
        print(f"  Class balance verified: {balance_ok}")

        counts = {}
        for label in self.train_y:
            counts[label] = counts.get(label, 0) + 1
        print(f"  Post-SMOTE distribution: {counts}")

    def _stage_3_train_model(self):
        """Hyperparameter tuning and final model training."""
        print("\n[STAGE 3] XGBoost Hyperparameter Tuning and Training")

        print("  Running hyperparameter grid search...")
        search_model = XGBoostClassifier(random_seed=RANDOM_SEED)
        best_params, best_score = search_model.tune_hyperparameters(
            self.train_X, self.train_y, XGBOOST_PARAMS, n_folds=N_FOLDS
        )
        print(f"  Best params: {best_params}")
        print(f"  Best CV accuracy: {best_score:.4f}")

        print("\n  Training final model with best parameters...")
        self.model = XGBoostClassifier(
            n_estimators=best_params.get("n_estimators", 100),
            max_depth=best_params.get("max_depth", 5),
            learning_rate=best_params.get("learning_rate", 0.1),
            min_samples_split=best_params.get("min_samples_split", 2),
            subsample=best_params.get("subsample", 0.8),
            random_seed=RANDOM_SEED
        )
        self.model.train(self.train_X, self.train_y)
        print(f"  Model training complete ({len(self.train_X)} samples used.)")

    def _stage_4_predict_and_evaluate(self):
        """Predict on the held-out test set and compute performance metrics."""
        print("\n[STAGE 4] Prediction and Evaluation")

        self.predictions = self.model.predict(self.test_X)
        self.probabilities = self.model.predict_proba(self.test_X)

        self.metrics = PerformanceMetrics(model_id="xgboost_scratch")
        self.metrics.compute(self.test_y, self.predictions, self.probabilities, TARGET_CLASSES)

        print(self.metrics.print_report())

        meets_criteria = self.metrics.meets_acceptance_criteria(
            MIN_ACCURACY, min_dropout_recall=MIN_DROPOUT_RECALL, min_auc=0.0)
        print(f"  Meets acceptance criteria: {meets_criteria}")
        print(f"    Accuracy >= {MIN_ACCURACY}: {self.metrics.accuracy:.4f} "
              f"({'PASS' if self.metrics.accuracy >= MIN_ACCURACY else 'BELOW TARGET'})")
        print(f"    Dropout Recall >= {MIN_DROPOUT_RECALL}: "
              f"{self.metrics.get_dropout_recall():.4f} "
              f"({'PASS' if self.metrics.get_dropout_recall() >= MIN_DROPOUT_RECALL else 'BELOW TARGET'})")

        n_at_risk = sum(1 for p in self.predictions if p == "Dropout")
        print(f"\n  Students predicted at risk of dropout: {n_at_risk} / {len(self.predictions)}")

    def _stage_5_feature_importance(self):
        """Report which features the model relies on most."""
        print("\n[STAGE 5] Feature Importance")

        importance = self.model.get_feature_importance(self.feature_names_encoded)
        top_features = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:10]

        print("  Top 10 features by split frequency:")
        for name, score in top_features:
            print(f"    {name:<45} {score:.4f}")

    def _save_predictions_csv(self):
        """Save per-student predictions to CSV file."""
        csv_path = os.path.join(self.output_dir, "predictions.csv")
        dropout_idx = TARGET_CLASSES.index("Dropout")

        with open(csv_path, 'w') as f:
            f.write("student_id,actual_outcome,predicted_outcome,dropout_probability,at_risk\n")
            for student, pred, proba in zip(self.test_students, self.predictions, self.probabilities):
                dropout_prob = proba[dropout_idx]
                at_risk = pred == "Dropout"
                f.write(f"{student.student_id},{student.outcome},{pred},"
                        f"{dropout_prob:.4f},{at_risk}\n")

        print(f"\n  Predictions CSV saved to: {csv_path}")

    def _log_experiment(self):
        """Append this run's hyperparameters, CV score, and metrics to experiment_log.json."""
        log_path = os.path.join(self.output_dir, "experiment_log.json")
        existing = []
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                existing = json.load(f)

        entry = {
            "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "random_seed": RANDOM_SEED,
            "best_params": self.model.__dict__.get("__params", {
                "n_estimators": self.model.n_estimators,
                "max_depth": self.model.max_depth,
                "learning_rate": self.model.learning_rate,
                "min_samples_split": self.model.min_samples_split,
                "subsample": self.model.subsample,
            }),
            "metrics": {
                "accuracy": round(self.metrics.accuracy, 4),
                "auc_roc": round(self.metrics.auc_roc, 4),
                "dropout_recall": round(self.metrics.get_dropout_recall(), 4),
                "dropout_f1": round(self.metrics.f1_per_class.get("Dropout", 0), 4),
            },
            "model_artefact": os.path.join(self.output_dir, "model.pkl"),
        }
        existing.append(entry)
        with open(log_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"  Experiment logged to: {log_path}")

    def _stage_6_enrollment_model(self):
        """Train and evaluate a second model using only enrollment-time features.
        Saves output/model_enrollment.pkl alongside the full model."""
        print("\n[STAGE 6] Enrollment-Time Early Detection Model")
        print("  Training on features available before classes start...")

        # Filter to only enrollment features that actually exist in the dataset
        available = [f for f in ENROLLMENT_FEATURES if f in self.all_features]
        print(f"  Enrollment features used: {len(available)}")

        # Re-encode using only enrollment features
        enrol_encoder = EncodedFeatureMatrix(random_seed=RANDOM_SEED)
        enrol_train_X = enrol_encoder.fit_transform(
            self.train_students, available,
            categorical_features=[
                f for f in available
                if f in self.encoder._categorical_columns
            ],
            partition_label="train"
        )
        enrol_test_X = enrol_encoder.transform(
            self.test_students, available, partition_label="test"
        )

        # Balance the enrollment training set
        enrol_train_y_orig = [s.outcome for s in self.train_students]
        smote = SMOTESampler(k_neighbours=5, random_seed=RANDOM_SEED)
        enrol_train_X, enrol_train_y = smote.fit_resample(enrol_train_X, enrol_train_y_orig)

        # Train with same best params as the full model
        enrol_model = XGBoostClassifier(
            n_estimators=self.model.n_estimators,
            max_depth=self.model.max_depth,
            learning_rate=self.model.learning_rate,
            min_samples_split=self.model.min_samples_split,
            subsample=self.model.subsample,
            random_seed=RANDOM_SEED
        )
        enrol_model.train(enrol_train_X, enrol_train_y)

        # Evaluate
        enrol_preds = enrol_model.predict(enrol_test_X)
        enrol_proba = enrol_model.predict_proba(enrol_test_X)
        enrol_metrics = PerformanceMetrics(model_id="xgboost_enrollment")
        enrol_metrics.compute(self.test_y, enrol_preds, enrol_proba, TARGET_CLASSES)
        print(enrol_metrics.print_report())

        dropout_recall = enrol_metrics.get_dropout_recall()
        print(f"  Enrollment model dropout recall: {dropout_recall:.4f}")
        print(f"  (Full model dropout recall:      {self.metrics.get_dropout_recall():.4f})")

        # Save
        # Save enrollment predictions CSV
        enrol_csv = os.path.join(self.output_dir, "predictions_enrollment.csv")
        dropout_idx = TARGET_CLASSES.index("Dropout")
        with open(enrol_csv, "w") as f:
            f.write("student_id,actual_outcome,predicted_outcome,dropout_probability,at_risk\n")
            for student, pred, proba in zip(self.test_students, enrol_preds, enrol_proba):
                dp = proba[dropout_idx]
                f.write(f"{student.student_id},{student.outcome},{pred},{dp:.4f},{pred == 'Dropout'}\n")
        print(f"  Enrollment predictions saved to: {enrol_csv}")

        model_path = os.path.join(self.output_dir, "model_enrollment.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": enrol_model,
                "encoder": enrol_encoder,
                "all_features": available,
            }, f)
        print(f"  Enrollment model saved to: {model_path}")

    def _stage_7_semester1_model(self):
        """Train and evaluate a third model using enrollment features plus
        1st semester academic performance. Saves output/model_semester1.pkl."""
        print("\n[STAGE 7] Semester 1 Early Detection Model")
        print("  Training on enrollment features + 1st semester results...")

        available = [f for f in SEMESTER1_FEATURES if f in self.all_features]
        print(f"  Semester 1 features used: {len(available)}")

        sem1_encoder = EncodedFeatureMatrix(random_seed=RANDOM_SEED)
        sem1_train_X = sem1_encoder.fit_transform(
            self.train_students, available,
            categorical_features=[
                f for f in available
                if f in self.encoder._categorical_columns
            ],
            partition_label="train"
        )
        sem1_test_X = sem1_encoder.transform(
            self.test_students, available, partition_label="test"
        )

        sem1_train_y_orig = [s.outcome for s in self.train_students]
        smote = SMOTESampler(k_neighbours=5, random_seed=RANDOM_SEED)
        sem1_train_X, sem1_train_y = smote.fit_resample(sem1_train_X, sem1_train_y_orig)

        sem1_model = XGBoostClassifier(
            n_estimators=self.model.n_estimators,
            max_depth=self.model.max_depth,
            learning_rate=self.model.learning_rate,
            min_samples_split=self.model.min_samples_split,
            subsample=self.model.subsample,
            random_seed=RANDOM_SEED
        )
        sem1_model.train(sem1_train_X, sem1_train_y)

        sem1_preds = sem1_model.predict(sem1_test_X)
        sem1_proba = sem1_model.predict_proba(sem1_test_X)
        sem1_metrics = PerformanceMetrics(model_id="xgboost_semester1")
        sem1_metrics.compute(self.test_y, sem1_preds, sem1_proba, TARGET_CLASSES)
        print(sem1_metrics.print_report())

        dropout_recall = sem1_metrics.get_dropout_recall()
        print(f"  Semester 1 model dropout recall:  {dropout_recall:.4f}")
        print(f"  (Full model dropout recall:        {self.metrics.get_dropout_recall():.4f})")

        # Save semester 1 predictions CSV
        sem1_csv = os.path.join(self.output_dir, "predictions_semester1.csv")
        dropout_idx = TARGET_CLASSES.index("Dropout")
        with open(sem1_csv, "w") as f:
            f.write("student_id,actual_outcome,predicted_outcome,dropout_probability,at_risk\n")
            for student, pred, proba in zip(self.test_students, sem1_preds, sem1_proba):
                dp = proba[dropout_idx]
                f.write(f"{student.student_id},{student.outcome},{pred},{dp:.4f},{pred == 'Dropout'}\n")
        print(f"  Semester 1 predictions saved to: {sem1_csv}")

        model_path = os.path.join(self.output_dir, "model_semester1.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": sem1_model,
                "encoder": sem1_encoder,
                "all_features": available,
            }, f)
        print(f"  Semester 1 model saved to: {model_path}")

    def _save_model_artifacts(self):
        """Persist the trained model and encoder so predict.py can reuse them
        without retraining."""
        model_path = os.path.join(self.output_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "encoder": self.encoder,
                "all_features": self.all_features,
            }, f)
        print(f"  Trained model saved to: {model_path}")
