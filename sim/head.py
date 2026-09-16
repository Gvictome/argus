"""
The ARGUS federated head: a small numpy MLP, ~200K parameters.

Kept in numpy rather than torch on purpose. CON-3 in Rev 2.2 bounds the
federated model at roughly 200K parameters precisely because nightly
training runs on the Pi 5 CPU, and flwr's NumPyClient wants lists of
ndarray anyway. No framework import means the resident-set number this
simulator reports is honest about the head itself -- see the note printed
at the end of a run about framework overhead.

19 features -> 384 -> 384 -> 128 -> 5 classes.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np

from workload import N_CLASSES, N_FEATURES

HIDDEN = (384, 384, 128)


def _he(shape, rng):
    return (rng.standard_normal(shape) * np.sqrt(2.0 / shape[0])).astype(np.float32)


class FederatedHead:
    def __init__(self, seed: int = 0, lr: float = 3e-3):
        rng = np.random.default_rng(seed)
        dims = (N_FEATURES,) + HIDDEN + (N_CLASSES,)
        self.W = [_he((dims[i], dims[i + 1]), rng) for i in range(len(dims) - 1)]
        self.b = [np.zeros(dims[i + 1], dtype=np.float32) for i in range(len(dims) - 1)]
        self.lr = lr
        self._reset_adam()

    def _reset_adam(self):
        self._mW = [np.zeros_like(w) for w in self.W]
        self._vW = [np.zeros_like(w) for w in self.W]
        self._mb = [np.zeros_like(b) for b in self.b]
        self._vb = [np.zeros_like(b) for b in self.b]
        self._t = 0

    @property
    def n_params(self) -> int:
        return sum(w.size for w in self.W) + sum(b.size for b in self.b)

    # ---- flwr NumPyClient interface -------------------------------------
    def get_weights(self) -> List[np.ndarray]:
        out = []
        for w, b in zip(self.W, self.b):
            out.append(w)
            out.append(b)
        return out

    def set_weights(self, params: List[np.ndarray]) -> None:
        for i in range(len(self.W)):
            self.W[i] = np.array(params[2 * i], dtype=np.float32)  # copy: never alias another head
            self.b[i] = np.array(params[2 * i + 1], dtype=np.float32)

    def save(self, path) -> None:
        """Write weights atomically, so a crash never leaves half a model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, **{f"p{i}": a for i, a in enumerate(self.get_weights())})
        tmp.replace(path)

    def load(self, path) -> None:
        with np.load(path) as z:
            params = [z[f"p{i}"] for i in range(len(z.files))]
        expected = self.get_weights()
        if len(params) != len(expected) or any(
            a.shape != b.shape for a, b in zip(params, expected)
        ):
            raise ValueError(f"{path}: weight shapes do not match this head")
        self.set_weights(params)

    # ---- forward / backward ---------------------------------------------
    def _forward(self, X):
        acts = [X]
        a = X
        for i in range(len(self.W) - 1):
            a = np.maximum(a @ self.W[i] + self.b[i], 0.0)
            acts.append(a)
        logits = a @ self.W[-1] + self.b[-1]
        return logits, acts

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits, _ = self._forward(X)
        return logits.argmax(axis=1)

    def infer_one(self, x: np.ndarray) -> int:
        """Single-sample forward pass -- what the pipeline calls per frame."""
        logits, _ = self._forward(x.reshape(1, -1))
        return int(logits.argmax())

    def evaluate(self, X, y) -> Tuple[float, float]:
        logits, _ = self._forward(X)
        p = self._softmax(logits)
        n = len(y)
        loss = float(-np.log(np.clip(p[np.arange(n), y], 1e-12, None)).mean())
        acc = float((logits.argmax(axis=1) == y).mean())
        return loss, acc

    def balanced_accuracy(self, X, y) -> float:
        """DETECTION_ACCURACY.md sec 0 is emphatic: on skewed data, quote this."""
        pred = self.predict(X)
        accs = []
        for c in range(N_CLASSES):
            m = y == c
            if m.any():
                accs.append(float((pred[m] == c).mean()))
        return float(np.mean(accs)) if accs else 0.0

    def fit(self, X, y, epochs: int = 1, batch: int = 64, seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        n = len(X)
        last = 0.0
        for _ in range(max(1, epochs)):
            order = rng.permutation(n)
            for s in range(0, n, batch):
                idx = order[s:s + batch]
                xb, yb = X[idx], y[idx]
                logits, acts = self._forward(xb)
                p = self._softmax(logits)
                m = len(idx)
                last = float(-np.log(np.clip(p[np.arange(m), yb], 1e-12, None)).mean())

                g = p
                g[np.arange(m), yb] -= 1.0
                g /= m

                gW = [None] * len(self.W)
                gb = [None] * len(self.b)
                for i in range(len(self.W) - 1, -1, -1):
                    gW[i] = acts[i].T @ g
                    gb[i] = g.sum(axis=0)
                    if i > 0:
                        g = (g @ self.W[i].T) * (acts[i] > 0)
                self._adam(gW, gb)
        return {"loss": last}

    def _adam(self, gW, gb, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        c1 = 1 - b1 ** self._t
        c2 = 1 - b2 ** self._t
        for i in range(len(self.W)):
            self._mW[i] = b1 * self._mW[i] + (1 - b1) * gW[i]
            self._vW[i] = b2 * self._vW[i] + (1 - b2) * (gW[i] ** 2)
            self.W[i] -= self.lr * (self._mW[i] / c1) / (np.sqrt(self._vW[i] / c2) + eps)
            self._mb[i] = b1 * self._mb[i] + (1 - b1) * gb[i]
            self._vb[i] = b2 * self._vb[i] + (1 - b2) * (gb[i] ** 2)
            self.b[i] -= self.lr * (self._mb[i] / c1) / (np.sqrt(self._vb[i] / c2) + eps)
