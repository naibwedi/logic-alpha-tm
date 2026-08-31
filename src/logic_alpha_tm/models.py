from __future__ import annotations

import numpy as np
import pandas as pd


class BernoulliSelector:
    """Interpretable dependency-light baseline for Boolean inputs."""

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.classes_: np.ndarray | None = None
        self.feature_names_: list[str] = []
        self.log_p_: np.ndarray | None = None
        self.log_prior_: np.ndarray | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "BernoulliSelector":
        self.classes_ = np.array(sorted(y.unique()))
        self.feature_names_ = list(x.columns)
        self.log_p_ = np.empty((len(self.classes_), x.shape[1]))
        self.log_prior_ = np.empty(len(self.classes_))
        for i, label in enumerate(self.classes_):
            group = x.loc[y == label].to_numpy(dtype=float)
            p = (group.sum(axis=0) + self.smoothing) / (len(group) + 2 * self.smoothing)
            self.log_p_[i] = np.log(np.clip(p, 1e-9, 1 - 1e-9))
            self.log_prior_[i] = np.log((len(group) + self.smoothing) / (len(y) + len(self.classes_) * self.smoothing))
        return self

    def decision_function(self, x: pd.DataFrame) -> np.ndarray:
        values = x.to_numpy(dtype=float)
        return values @ self.log_p_.T + (1 - values) @ np.log1p(-np.exp(self.log_p_)).T + self.log_prior_

    def predict_with_margin(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        scores = self.decision_function(x)
        order = np.argsort(scores, axis=1)
        prediction = self.classes_[order[:, -1]]
        margin = scores[np.arange(len(scores)), order[:, -1]] - scores[np.arange(len(scores)), order[:, -2]]
        return prediction, margin

    def rules(self, top_n: int = 8) -> pd.DataFrame:
        rows = []
        centered = self.log_p_ - self.log_p_.mean(axis=0, keepdims=True)
        for class_i, label in enumerate(self.classes_):
            for feature_i in np.argsort(centered[class_i])[-top_n:][::-1]:
                rows.append({"strategy": label, "literal": self.feature_names_[feature_i], "weight": centered[class_i, feature_i]})
        return pd.DataFrame(rows)


class TMUSelector:
    def __init__(
        self,
        clauses: int = 1000,
        threshold: int = 100,
        specificity: float = 5.0,
        epochs: int = 20,
        platform: str = "CPU",
    ):
        self.params = clauses, threshold, specificity
        self.epochs = epochs
        self.platform = platform
        self.classes_: np.ndarray | None = None
        self.model = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "TMUSelector":
        try:
            from tmu.models.classification.vanilla_classifier import TMClassifier
        except ImportError as exc:
            raise RuntimeError("TMU is optional. Install the 'tm' extra before using --model tmu.") from exc
        self.classes_, encoded_y = np.unique(y, return_inverse=True)
        self.model = TMClassifier(*self.params, platform=self.platform, weighted_clauses=True)
        for _ in range(self.epochs):
            self.model.fit(x.to_numpy(dtype=np.uint32), encoded_y.astype(np.uint32))
        return self

    def predict_with_margin(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        # TMU prediction is stable; public vote APIs vary by model version, so margin is NaN.
        encoded = self.model.predict(x.to_numpy(dtype=np.uint32))
        return self.classes_[encoded], np.full(len(encoded), np.nan)

    def rules(self, top_n: int = 8) -> pd.DataFrame:
        return pd.DataFrame(columns=["strategy", "literal", "weight"])


class LogisticSelector:
    """Regularized multinomial logistic-regression benchmark."""

    def __init__(self, seed: int = 7, learning_rate: float = 0.1, iterations: int = 800, l2: float = 0.01):
        self.seed = seed
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2 = l2
        self.feature_names_: list[str] = []
        self.classes_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "LogisticSelector":
        self.feature_names_ = list(x.columns)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        values = np.column_stack([np.ones(len(x)), x.to_numpy(dtype=float)])
        targets = np.eye(len(self.classes_))[encoded]
        counts = np.bincount(encoded, minlength=len(self.classes_))
        sample_weight = len(encoded) / (len(self.classes_) * counts[encoded])
        self.weights_ = np.zeros((values.shape[1], len(self.classes_)))
        for _ in range(self.iterations):
            logits = values @ self.weights_
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            error = (probabilities - targets) * sample_weight[:, None]
            gradient = values.T @ error / len(values)
            gradient[1:] += self.l2 * self.weights_[1:]
            self.weights_ -= self.learning_rate * gradient
        return self

    def predict_with_margin(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        values = np.column_stack([np.ones(len(x)), x.to_numpy(dtype=float)])
        scores = values @ self.weights_
        order = np.argsort(scores, axis=1)
        prediction = self.classes_[order[:, -1]]
        margin = scores[np.arange(len(scores)), order[:, -1]] - scores[np.arange(len(scores)), order[:, -2]]
        return prediction, margin

    def rules(self, top_n: int = 8) -> pd.DataFrame:
        rows = []
        coefficients = self.weights_[1:].T
        for class_i, label in enumerate(self.classes_):
            for feature_i in np.argsort(np.abs(coefficients[class_i]))[-top_n:][::-1]:
                rows.append({
                    "strategy": label,
                    "literal": self.feature_names_[feature_i],
                    "weight": float(coefficients[class_i, feature_i]),
                })
        return pd.DataFrame(rows)


class BoostedTreeSelector:
    """Multiclass SAMME boosted decision-stump benchmark for Boolean inputs."""

    def __init__(self, seed: int = 7, estimators: int = 80):
        self.seed = seed
        self.estimators = estimators
        self.classes_: np.ndarray | None = None
        self.stumps_: list[tuple[int, int, int, float]] = []

    @staticmethod
    def _best_stump(values: np.ndarray, encoded: np.ndarray, weights: np.ndarray, n_classes: int):
        best = None
        for feature in range(values.shape[1]):
            predictions = np.empty(len(values), dtype=int)
            branch_classes = []
            for branch in (0, 1):
                mask = values[:, feature] == branch
                class_weights = np.bincount(encoded[mask], weights=weights[mask], minlength=n_classes)
                branch_classes.append(int(np.argmax(class_weights)))
                predictions[mask] = branch_classes[-1]
            error = float(weights[predictions != encoded].sum())
            if best is None or error < best[0]:
                best = error, feature, branch_classes[0], branch_classes[1], predictions
        return best

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "BoostedTreeSelector":
        values = x.to_numpy(dtype=np.uint8)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        weights = np.full(len(values), 1 / len(values))
        self.stumps_ = []
        for _ in range(self.estimators):
            error, feature, class_zero, class_one, predictions = self._best_stump(
                values, encoded, weights, n_classes
            )
            if error >= 1 - 1 / n_classes:
                break
            error = np.clip(error, 1e-12, 1 - 1e-12)
            alpha = float(np.log((1 - error) / error) + np.log(n_classes - 1))
            self.stumps_.append((feature, class_zero, class_one, alpha))
            weights *= np.exp(alpha * (predictions != encoded))
            weights /= weights.sum()
        return self

    def predict_with_margin(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        values = x.to_numpy(dtype=np.uint8)
        scores = np.zeros((len(values), len(self.classes_)))
        for feature, class_zero, class_one, alpha in self.stumps_:
            predicted = np.where(values[:, feature] == 0, class_zero, class_one)
            scores[np.arange(len(values)), predicted] += alpha
        order = np.argsort(scores, axis=1)
        prediction = self.classes_[order[:, -1]]
        margin = scores[np.arange(len(scores)), order[:, -1]] - scores[np.arange(len(scores)), order[:, -2]]
        return prediction, margin

    def rules(self, top_n: int = 8) -> pd.DataFrame:
        return pd.DataFrame(columns=["strategy", "literal", "weight"])
