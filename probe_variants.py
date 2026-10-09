"""Try stronger heads on the frozen DINOv2 features."""

from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize

sys.stdout.reconfigure(encoding="utf-8")

import plantcls as P  # noqa: E402

CNN_MEMBERS = ["b0_288", "cnx320", "cnx320p", "cnx320m", "cnx384m", "cnxs320m"]
N_CLASS = len(P.CLASSES)


def knn_proba(xtr, ytr, xte, k):
    sim = normalize(xte) @ normalize(xtr).T
    idx = np.argsort(-sim, axis=1)[:, :k]
    out = np.zeros((len(xte), N_CLASS))
    for i in range(len(xte)):
        for j in idx[i]:
            out[i, ytr[j]] += 1
    return out / k


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main() -> None:
    data = np.load(os.path.join(P.OUT_DIR, "dino_features.npz"))
    X, labels = data["train"], data["labels"]
    print(f"features {X.shape}")
    cnn = np.mean([np.load(os.path.join(P.OUT_DIR, m, "oof.npz"))["probs"]
                   for m in CNN_MEMBERS], axis=0)
    cnn_err = cnn.argmax(1) != labels

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rules = {}

    for k in (3, 5, 10, 20):
        oof = np.zeros_like(cnn)
        for tr, va in skf.split(X, labels):
            oof[va] = knn_proba(X[tr], labels[tr], X[va], k)
        rules[f"kNN k={k}"] = oof

    for C in (0.01, 0.03, 0.1, 0.3):
        oof = np.zeros_like(cnn)
        for tr, va in skf.split(X, labels):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=C, max_iter=5000)
            clf.fit(sc.transform(X[tr]), labels[tr])
            oof[va] = clf.predict_proba(sc.transform(X[va]))
        rules[f"logreg C={C}"] = oof

    oof = np.zeros_like(cnn)
    for tr, va in skf.split(X, labels):
        A, B = normalize(X[tr]), normalize(X[va])
        means = np.stack([A[labels[tr] == c].mean(0) for c in range(N_CLASS)])
        oof[va] = softmax((B @ means.T) * 20)
    rules["nearest class mean"] = oof

    print(f"\n{'rule':<22s} {'OOF':>7s} {'fixes':>6s} {'own err':>8s}")
    for name, prob in rules.items():
        acc = accuracy_score(labels, prob.argmax(1))
        fixed = int((cnn_err & (prob.argmax(1) == labels)).sum())
        own = int((prob.argmax(1) != labels).sum())
        print(f"{name:<22s} {acc:>7.4f} {fixed:>6d} {own:>8d}")

    print(f"\nCNN ensemble: OOF {accuracy_score(labels, cnn.argmax(1)):.4f}, "
          f"errors {int(cnn_err.sum())}")
    best = max(rules, key=lambda n: accuracy_score(labels, rules[n].argmax(1)))
    prob = rules[best]
    print(f"\nblend with the strongest probe ({best}):")
    for w in (0.05, 0.1, 0.15, 0.2, 0.3):
        acc = accuracy_score(labels, ((1 - w) * cnn + w * prob).argmax(1))
        print(f"   w={w:.2f}  {acc:.4f}")


if __name__ == "__main__":
    main()
