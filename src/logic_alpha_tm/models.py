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
    def __init__(self, clauses: int = 1000, threshold: int = 100, specificity: float = 5.0, epochs: int = 20):
        self.params = clauses, threshold, specificity
        self.epochs = epochs
        self.classes_: np.ndarray | None = None
        self.model = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "TMUSelector":
        try:
            from tmu.models.classification.vanilla_classifier import TMClassifier
        except ImportError as exc:
            raise RuntimeError("TMU is optional. Install the 'tm' extra before using --model tmu.") from exc
        self.classes_, encoded_y = np.unique(y, return_inverse=True)
        self.model = TMClassifier(*self.params, platform="CPU", weighted_clauses=True)
        for _ in range(self.epochs):
            self.model.fit(x.to_numpy(dtype=np.uint32), encoded_y.astype(np.uint32))
        return self

    def predict_with_margin(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        # TMU prediction is stable; public vote APIs vary by model version, so margin is NaN.
        encoded = self.model.predict(x.to_numpy(dtype=np.uint32))
        return self.classes_[encoded], np.full(len(encoded), np.nan)

    def rules(self, top_n: int = 8) -> pd.DataFrame:
        return pd.DataFrame(columns=["strategy", "literal", "weight"])

