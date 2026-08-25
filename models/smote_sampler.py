"""SMOTE oversampling for class balancing."""

import random
import math


class SMOTESampler:
    """SMOTE oversampling to balance class distributions in training folds."""

    def __init__(self, k_neighbours=5, random_seed=42, sampling_strategy="auto"):
        self.k_neighbours = k_neighbours
        self.random_seed = random_seed
        self.sampling_strategy = sampling_strategy

    def _euclidean_distance(self, a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def _find_k_neighbours(self, sample, samples, k):
        distances = []
        for i, other in enumerate(samples):
            if other is not sample:
                dist = self._euclidean_distance(sample, other)
                distances.append((i, dist))
        distances.sort(key=lambda x: x[1])
        return [samples[distances[i][0]] for i in range(min(k, len(distances)))]

    def _generate_synthetic_sample(self, sample, neighbour, rng):
        diff = [n - s for s, n in zip(sample, neighbour)]
        gap = rng.random()
        return [s + gap * d for s, d in zip(sample, diff)]

    def fit_resample(self, X, y):
        """Apply SMOTE to balance classes. Returns X_resampled, y_resampled."""
        rng = random.Random(self.random_seed)

        # Group samples by class
        class_samples = {}
        for features, label in zip(X, y):
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append(features)

        # Find majority class count
        majority_count = max(len(samples) for samples in class_samples.values())

        # Resample minority classes
        X_resampled = list(X)
        y_resampled = list(y)

        for label, samples in class_samples.items():
            n_to_generate = majority_count - len(samples)
            if n_to_generate <= 0:
                continue

            generated = 0
            while generated < n_to_generate:
                # Pick a random minority sample
                idx = rng.randint(0, len(samples) - 1)
                sample = samples[idx]

                # Find k nearest neighbours within the same class
                neighbours = self._find_k_neighbours(sample, samples, self.k_neighbours)
                if not neighbours:
                    break

                # Pick a random neighbour and generate synthetic sample
                neighbour = neighbours[rng.randint(0, len(neighbours) - 1)]
                synthetic = self._generate_synthetic_sample(sample, neighbour, rng)

                X_resampled.append(synthetic)
                y_resampled.append(label)
                generated += 1

        return X_resampled, y_resampled

    def verify_class_balance(self, y):
        """Check that all classes have equal count after SMOTE."""
        counts = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        min_count = min(counts.values())
        max_count = max(counts.values())
        return min_count >= max_count  # Minority >= Majority after SMOTE
