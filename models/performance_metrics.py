
import math


class PerformanceMetrics:
    """Computes and stores all model evaluation metrics."""

    def __init__(self, model_id=None):
        self.model_id = model_id
        self.accuracy = 0.0
        self.precision_per_class = {}
        self.recall_per_class = {}
        self.f1_per_class = {}
        self.auc_roc = 0.0
        self.auc_roc_per_class = {}
        self.confusion_matrix = {}  # (actual, predicted) -> count
        self.classes = []

    def compute(self, y_true, y_pred, y_proba=None, classes=None):
        """Compute all metrics from predictions."""
        self.classes = classes or sorted(set(y_true + y_pred))
        n = len(y_true)

        # Accuracy
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        self.accuracy = correct / n if n > 0 else 0.0

        # Confusion Matrix
        self.confusion_matrix = {}
        for actual in self.classes:
            for predicted in self.classes:
                self.confusion_matrix[(actual, predicted)] = 0

        for t, p in zip(y_true, y_pred):
            self.confusion_matrix[(t, p)] = self.confusion_matrix.get((t, p), 0) + 1

        # Per-class metrics
        for cls in self.classes:
            tp = self.confusion_matrix.get((cls, cls), 0)
            fp = sum(self.confusion_matrix.get((other, cls), 0)
                     for other in self.classes if other != cls)
            fn = sum(self.confusion_matrix.get((cls, other), 0)
                     for other in self.classes if other != cls)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                   if (precision + recall) > 0 else 0.0)

            self.precision_per_class[cls] = precision
            self.recall_per_class[cls] = recall
            self.f1_per_class[cls] = f1

        if y_proba is not None:
            self._compute_auc_roc(y_true, y_proba)

    def _compute_auc_roc(self, y_true, y_proba):
        """Compute macro-averaged AUC-ROC (one-vs-rest)."""
        auc_scores = []
        n_proba_classes = len(y_proba[0]) if y_proba else 0
        for cls_idx, cls in enumerate(self.classes):
            if cls_idx >= n_proba_classes:
                continue
            binary_true = [1 if t == cls else 0 for t in y_true]
            scores = [proba[cls_idx] for proba in y_proba]
            auc = self._auc_binary(binary_true, scores)
            self.auc_roc_per_class[cls] = auc
            auc_scores.append(auc)

        self.auc_roc = sum(auc_scores) / len(auc_scores) if auc_scores else 0.0

    def _auc_binary(self, y_true, scores):
        """Compute AUC for binary classification using the trapezoidal rule."""
        # Pair (score, label), sorted by score descending
        paired = sorted(zip(scores, y_true), key=lambda x: -x[0])

        tp, fp = 0, 0
        total_pos = sum(y_true)
        total_neg = len(y_true) - total_pos

        if total_pos == 0 or total_neg == 0:
            return 0.5

        tpr_prev, fpr_prev = 0.0, 0.0
        auc = 0.0

        for score, label in paired:
            if label == 1:
                tp += 1
            else:
                fp += 1
            tpr = tp / total_pos
            fpr = fp / total_neg
            # Trapezoidal rule
            auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
            tpr_prev = tpr
            fpr_prev = fpr

        return auc

    def get_dropout_recall(self):
        """Get recall specifically for the Dropout class."""
        return self.recall_per_class.get("Dropout", 0.0)

    def meets_acceptance_criteria(self, min_accuracy=0.88, min_auc=0.93,
                                   min_dropout_recall=0.80):
        """Check if model meets performance thresholds."""
        return (self.accuracy >= min_accuracy and
                self.auc_roc >= min_auc and
                self.get_dropout_recall() >= min_dropout_recall)

    def generate_roc_curve_data(self):
        """Get ROC curve data points per class."""
        # This would require the raw probabilities; stored for reporting
        return self.auc_roc_per_class

    def print_report(self):
        """Print a formatted classification report."""
        report = f"\n{'='*60}\n"
        report += f"  MODEL PERFORMANCE REPORT (Model: {self.model_id})\n"
        report += f"{'='*60}\n\n"
        report += f"  Overall Accuracy: {self.accuracy:.4f}\n"
        report += f"  Macro AUC-ROC:    {self.auc_roc:.4f}\n\n"

        report += f"  {'Class':<12} | {'Precision':>9} | {'Recall':>9} | {'F1-Score':>9} | {'AUC-ROC':>9}\n"
        report += f"  {'-'*12}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}\n"

        for cls in self.classes:
            p = self.precision_per_class.get(cls, 0)
            r = self.recall_per_class.get(cls, 0)
            f = self.f1_per_class.get(cls, 0)
            a = self.auc_roc_per_class.get(cls, 0)
            report += f"  {cls:<12} | {p:>9.4f} | {r:>9.4f} | {f:>9.4f} | {a:>9.4f}\n"

        report += f"\n  Confusion Matrix:\n"
        report += f"  {'':>12} | " + " | ".join(f"{c:>10}" for c in self.classes) + "\n"
        report += f"  {'-'*12}-+-" + "-+-".join("-" * 10 for _ in self.classes) + "\n"
        for actual in self.classes:
            row_vals = [str(self.confusion_matrix.get((actual, pred), 0)).rjust(10)
                        for pred in self.classes]
            report += f"  {actual:<12} | " + " | ".join(row_vals) + "\n"

        report += f"\n{'='*60}\n"
        return report

    def to_dict(self):
        return {
            "model_id": self.model_id,
            "accuracy": self.accuracy,
            "auc_roc": self.auc_roc,
            "per_class": {
                cls: {
                    "precision": self.precision_per_class.get(cls, 0),
                    "recall": self.recall_per_class.get(cls, 0),
                    "f1": self.f1_per_class.get(cls, 0)
                }
                for cls in self.classes
            },
            "confusion_matrix": self.confusion_matrix
        }
