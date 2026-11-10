
import math
import random

#Modiegi
class DecisionNode:
    """A single node in a decision tree."""

    def __init__(self, feature_index=None, threshold=None, left=None,
                 right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Leaf value (prediction)

    def is_leaf(self):
        return self.value is not None # tell you which kind of node you dealing with


class DecisionTree:

    def __init__(self, max_depth=3, min_samples_split=2, random_seed=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_seed = random_seed
        self.root = None

    def _compute_leaf_value(self, gradients, second_order_gradients, lambda_reg=1.0):
        return -sum(gradients) / (sum(second_order_gradients) + lambda_reg)

    def _compute_gain(self, left_g, left_h, right_g, right_h, lambda_reg=1.0):
        left_score = (left_g ** 2) / (left_h + lambda_reg)
        right_score = (right_g ** 2) / (right_h + lambda_reg)
        total_score = ((left_g + right_g) ** 2) / (left_h + right_h + lambda_reg)
        return 0.5 * (left_score + right_score - total_score)

    def _find_best_split(self, X, gradients, second_order_gradients, lambda_reg=1.0):
        n_samples = len(X)
        n_features = len(X[0]) if X else 0
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        total_g = sum(gradients)
        total_h = sum(second_order_gradients)

        for feature_idx in range(n_features):
            values = sorted(set(X[i][feature_idx] for i in range(n_samples)))
            if len(values) <= 1:
                continue

            # Use quantile-based thresholds (max 10 per feature) for better split quality
            n_thresholds = min(len(values) - 1, 10)
            step = max(1, (len(values) - 1) // n_thresholds)
            thresholds = []
            for idx in range(0, len(values) - 1, step):
                next_idx = min(idx + 1, len(values) - 1)
                if idx != next_idx:
                    thresholds.append((values[idx] + values[next_idx]) / 2)
            thresholds = thresholds[:n_thresholds]

            for threshold in thresholds:
                left_g, left_h = 0.0, 0.0
                right_g, right_h = 0.0, 0.0

                for i in range(n_samples):
                    if X[i][feature_idx] <= threshold:
                        left_g += gradients[i]
                        left_h += second_order_gradients[i]
                    else:
                        right_g += gradients[i]
                        right_h += second_order_gradients[i]

                if left_h == 0 or right_h == 0:
                    continue

                gain = self._compute_gain(left_g, left_h, right_g, right_h, lambda_reg)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, gradients, second_order_gradients, depth, lambda_reg=1.0):
        n_samples = len(X)

        if depth >= self.max_depth or n_samples < self.min_samples_split:
            leaf_value = self._compute_leaf_value(gradients, second_order_gradients, lambda_reg)
            return DecisionNode(value=leaf_value)

        feature_idx, threshold, gain = self._find_best_split(
            X, gradients, second_order_gradients, lambda_reg)

        if feature_idx is None or gain <= 0:
            leaf_value = self._compute_leaf_value(gradients, second_order_gradients, lambda_reg)
            return DecisionNode(value=leaf_value)

        left_X, left_g, left_h = [], [], []
        right_X, right_g, right_h = [], [], []

        for i in range(n_samples):
            if X[i][feature_idx] <= threshold:
                left_X.append(X[i])
                left_g.append(gradients[i])
                left_h.append(second_order_gradients[i])
            else:
                right_X.append(X[i])
                right_g.append(gradients[i])
                right_h.append(second_order_gradients[i])

        if not left_X or not right_X:
            leaf_value = self._compute_leaf_value(gradients, second_order_gradients, lambda_reg)
            return DecisionNode(value=leaf_value)

        left_node = self._build_tree(left_X, left_g, left_h, depth + 1, lambda_reg)
        right_node = self._build_tree(right_X, right_g, right_h, depth + 1, lambda_reg)

        return DecisionNode(
            feature_index=feature_idx,
            threshold=threshold,
            left=left_node,
            right=right_node
        )

    def fit(self, X, gradients, second_order_gradients, lambda_reg=1.0):
        self.root = self._build_tree(X, gradients, second_order_gradients, 0, lambda_reg)

    def predict_single(self, x):
        node = self.root
        while not node.is_leaf():
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        """Predict for multiple samples."""
        return [self.predict_single(x) for x in X]


class XGBoostClassifier:

    def __init__(self, n_estimators=100, max_depth=5, learning_rate=0.1,
                 lambda_reg=1.0, min_samples_split=2, subsample=1.0,
                 random_seed=42):
        self.n_estimators = n_estimators # Store number of trees to build
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.min_samples_split = min_samples_split # Store minimum samples needed to split
        self.subsample = subsample
        self.random_seed = random_seed
        self.trees = {}  # {class_idx: [list of trees]}
        self.classes = []
        self.n_classes = 0
        self._trained = False # Indicates model is not trained yet

    def _softmax(self, raw_scores):
        max_score = max(raw_scores)
        exp_scores = [math.exp(s - max_score) for s in raw_scores]
        total = sum(exp_scores)
        return [e / total for e in exp_scores] # Convert scores to probabilities

    def train(self, X, y):
        """Train the gradient boosted model."""
        rng = random.Random(self.random_seed)
        self.classes = sorted(set(y))
        self.n_classes = len(self.classes)
        class_to_idx = {c: i for i, c in enumerate(self.classes)}

        n_samples = len(X)

        y_onehot = [[0.0] * self.n_classes for _ in range(n_samples)]
        for i, label in enumerate(y):
            y_onehot[i][class_to_idx[label]] = 1.0

        raw_predictions = [[0.0] * self.n_classes for _ in range(n_samples)]

        for k in range(self.n_classes):
            self.trees[k] = []

        for iteration in range(self.n_estimators):
            probabilities = [self._softmax(raw_predictions[i]) for i in range(n_samples)]

            for k in range(self.n_classes):
                gradients = []
                second_order_gradients = []
                for i in range(n_samples):
                    g = probabilities[i][k] - y_onehot[i][k]
                    h = probabilities[i][k] * (1 - probabilities[i][k])
                    h = max(h, 1e-6)
                    gradients.append(g)
                    second_order_gradients.append(h)

                if self.subsample < 1.0:
                    n_sub = max(1, int(n_samples * self.subsample))
                    indices = rng.sample(range(n_samples), n_sub)
                    X_sub = [X[i] for i in indices]
                    g_sub = [gradients[i] for i in indices]
                    h_sub = [second_order_gradients[i] for i in indices]
                else:
                    X_sub = X
                    g_sub = gradients
                    h_sub = second_order_gradients

                tree = DecisionTree(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    random_seed=self.random_seed + iteration
                )
                tree.fit(X_sub, g_sub, h_sub, self.lambda_reg)
                self.trees[k].append(tree)

                predictions = tree.predict(X)
                for i in range(n_samples):
                    raw_predictions[i][k] += self.learning_rate * predictions[i]

        self._trained = True

    def predict_proba(self, X):
        """Predict class probabilities for samples."""
        if not self._trained:
            raise RuntimeError("Model not trained. Call train() first.")

        n_samples = len(X)
        raw_predictions = [[0.0] * self.n_classes for _ in range(n_samples)]

        for k in range(self.n_classes):
            for tree in self.trees[k]:
                preds = tree.predict(X)
                for i in range(n_samples):
                    raw_predictions[i][k] += self.learning_rate * preds[i]

        return [self._softmax(raw_predictions[i]) for i in range(n_samples)]

    def predict(self, X):
        """Predict class labels for samples."""
        probabilities = self.predict_proba(X)
        predictions = []
        for probs in probabilities:
            max_idx = probs.index(max(probs))
            predictions.append(self.classes[max_idx])
        return predictions

    def evaluate(self, X, y):
        """Evaluate model accuracy."""
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)

        correct = sum(1 for p, t in zip(predictions, y) if p == t)
        accuracy = correct / len(y) if y else 0.0

        return {
            "accuracy": accuracy,
            "predictions": predictions,
            "probabilities": probabilities
        }

    def tune_hyperparameters(self, X, y, param_grid, n_folds=5):
        """Grid search with stratified k-fold cross-validation."""
        rng = random.Random(self.random_seed)
        best_score = 0.0
        best_params = {}

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        def generate_combinations(values, idx=0, current=None):
            if current is None:
                current = {}
            if idx == len(values):
                return [current.copy()]
            combos = []
            for val in values[idx]:
                current[param_names[idx]] = val
                combos.extend(generate_combinations(values, idx + 1, current))
            return combos

        combinations = generate_combinations(param_values)
        if len(combinations) > 8:
            rng.shuffle(combinations)
            combinations = combinations[:8]

        # Subsample for faster CV ,Preparing the data for cross-validation
        max_cv_samples = 800
        if len(X) > max_cv_samples:
            indices = rng.sample(range(len(X)), max_cv_samples)
            X_cv = [X[i] for i in indices]
            y_cv = [y[i] for i in indices]
        else:
            X_cv = X
            y_cv = y

        folds = self._stratified_k_fold(X_cv, y_cv, n_folds, rng)

        for params in combinations:
            fold_scores = []
            for fold_idx in range(n_folds):
                train_X, train_y, val_X, val_y = [], [], [], []
                for i in range(len(X_cv)):
                    if folds[i] == fold_idx:
                        val_X.append(X_cv[i])
                        val_y.append(y_cv[i])
                    else:
                        train_X.append(X_cv[i])
                        train_y.append(y_cv[i])

                if not train_X or not val_X:
                    continue

                # Train with these params
                model = XGBoostClassifier(
                    n_estimators=params.get("n_estimators", 50),
                    max_depth=params.get("max_depth", 3),
                    learning_rate=params.get("learning_rate", 0.1),
                    min_samples_split=params.get("min_samples_split", 2),
                    subsample=params.get("subsample", 1.0),
                    random_seed=self.random_seed
                )
                model.train(train_X, train_y)
                result = model.evaluate(val_X, val_y)
                fold_scores.append(result["accuracy"])

            avg_score = sum(fold_scores) / len(fold_scores) if fold_scores else 0
            if avg_score > best_score:
                best_score = avg_score
                best_params = params.copy()

        return best_params, best_score

    def _stratified_k_fold(self, X, y, n_folds, rng):
        class_indices = {}
        for i, label in enumerate(y):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(i)
            

        folds = [0] * len(y)
        for label, indices in class_indices.items():
            rng.shuffle(indices)
            for i, idx in enumerate(indices):
                folds[idx] = i % n_folds

        return folds

    def save_model(self, path):
        pass  # Model kept in memory

    def get_feature_importance(self, feature_names):
        """Get feature importance based on split frequency."""
        importance = {name: 0 for name in feature_names}

        def count_splits(node):
            if node is None or node.is_leaf():
                return
            if 0 <= node.feature_index < len(feature_names):
                importance[feature_names[node.feature_index]] += 1
            count_splits(node.left)
            count_splits(node.right)

        for k in self.trees:
            for tree in self.trees[k]:
                count_splits(tree.root)

        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        return importance
