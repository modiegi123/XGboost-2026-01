
import math
import random
from copy import deepcopy


class EncodedFeatureMatrix:
    """Encodes and normalises features for ML pipeline consumption."""

    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        self.feature_names = []
        self.encoded_features = []  # List of lists (matrix)
        self.student_ids = []
        self.partition_labels = []
        self.imputation_applied = False #flag set to true if any missing values were filled

        # Fitted parameters (computed from training data only)
        self._min_values = {}
        self._max_values = {}
        self._mean_values = {}
        self._mode_values = {}
        self._categorical_columns = []
        self._categorical_categories = {}
        self._fitted = False

    def fit(self, students, feature_names, categorical_features=None):
        """Compute encoding parameters from training data only."""
        self._categorical_columns = categorical_features or []
        numerical_features = [f for f in feature_names if f not in self._categorical_columns]

        # Compute min/max for numerical features
        for feat in numerical_features:
            values = [s.features.get(feat, 0.0) for s in students
                      if s.features.get(feat) is not None]
            if values:
                self._min_values[feat] = min(values)
                self._max_values[feat] = max(values)
                self._mean_values[feat] = sum(values) / len(values)
            else:
                self._min_values[feat] = 0.0
                self._max_values[feat] = 1.0
                self._mean_values[feat] = 0.0

        # Compute categories and modes for categorical features
        for feat in self._categorical_columns:
            values = [s.features.get(feat) for s in students
                      if s.features.get(feat) is not None]
            if values:
                categories = sorted(set(values))
                self._categorical_categories[feat] = categories
                # Mode: most frequent value
                counts = {}
                for v in values:
                    counts[v] = counts.get(v, 0) + 1
                self._mode_values[feat] = max(counts, key=counts.get)
            else:
                self._categorical_categories[feat] = []
                self._mode_values[feat] = 0

        # Build output feature names
        self.feature_names = []
        for feat in feature_names:
            if feat in self._categorical_columns:
                for cat in self._categorical_categories.get(feat, []):
                    self.feature_names.append(f"{feat}_{cat}")
            else:
                self.feature_names.append(feat)

        self._fitted = True

    def transform(self, students, feature_names, partition_label="train"):
        """Transform student records using fitted parameters."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform()")

        numerical_features = [f for f in feature_names if f not in self._categorical_columns]
        self.encoded_features = []
        self.student_ids = []
        self.partition_labels = []

        for student in students:
            row = []
            for feat in feature_names:
                value = student.features.get(feat)

                if feat in self._categorical_columns:
                    # Impute missing with mode
                    if value is None:
                        value = self._mode_values.get(feat, 0)
                        self.imputation_applied = True
                    # One-hot encode
                    categories = self._categorical_categories.get(feat, [])
                    for cat in categories:
                        row.append(1.0 if value == cat else 0.0)
                else:
                    # Impute missing with mean
                    if value is None:
                        value = self._mean_values.get(feat, 0.0)
                        self.imputation_applied = True
                    # Min-Max normalisation to [0, 1]
                    min_val = self._min_values.get(feat, 0.0)
                    max_val = self._max_values.get(feat, 1.0)
                    if max_val - min_val > 0:
                        normalised = (value - min_val) / (max_val - min_val)
                    else:
                        normalised = 0.0
                    # Clamp to [0, 1]
                    normalised = max(0.0, min(1.0, normalised))
                    row.append(normalised)

            self.encoded_features.append(row)
            self.student_ids.append(student.student_id)
            self.partition_labels.append(partition_label)

        return self.encoded_features

    def fit_transform(self, students, feature_names, categorical_features=None,
                      partition_label="train"):
        self.fit(students, feature_names, categorical_features)
        return self.transform(students, feature_names, partition_label)

    def apply_one_hot_encoding(self, cols):
        self._categorical_columns = cols

    def apply_min_max_scaling(self, cols):
        pass  # Handled in transform

    def apply_mean_imputation(self, data):
        self.imputation_applied = True

    def to_feature_vector(self):
        return self.encoded_features

    def get_feature_names(self):
        return self.feature_names
