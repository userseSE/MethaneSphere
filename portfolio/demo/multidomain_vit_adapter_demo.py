"""Minimal multi-domain ViT adapter demo (public-safe, NumPy only).

Design idea:
- shared token encoder (universal backbone),
- domain-specific low-rank residual adapters,
- per-domain classification heads.

This mirrors the methane multi-sensor setup at a high level without exposing private training code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


class TinyAdapter:
    """LoRA-style residual adapter in NumPy."""

    def __init__(self, dim: int, rank: int = 16, alpha: float = 16.0, seed: int = 0):
        if rank <= 0:
            raise ValueError("rank must be > 0")
        rng = np.random.default_rng(seed)
        self.scale = alpha / rank
        self.down = rng.normal(0.0, 0.02, size=(dim, rank)).astype(np.float32)
        self.up = np.zeros((rank, dim), dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return ((x @ self.down) @ self.up) * self.scale


class SharedBackbone:
    """Small stand-in for a transformer block stack."""

    def __init__(self, dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 0.02, size=(dim, dim)).astype(np.float32)
        self.b1 = np.zeros((dim,), dtype=np.float32)
        self.w2 = rng.normal(0.0, 0.02, size=(dim, dim)).astype(np.float32)
        self.b2 = np.zeros((dim,), dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = layer_norm(x)
        y = gelu(y @ self.w1 + self.b1)
        y = y @ self.w2 + self.b2
        return y


@dataclass(frozen=True)
class DomainSpec:
    name: str
    num_classes: int


class MultiDomainAdapterClassifier:
    def __init__(self, *, dim: int, domains: List[DomainSpec], rank: int = 16):
        self.domain_order = [d.name for d in domains]
        self.backbone = SharedBackbone(dim)
        self.adapters = {d.name: TinyAdapter(dim, rank=rank, seed=i) for i, d in enumerate(domains)}
        rng = np.random.default_rng(123)
        self.heads_w = {
            d.name: rng.normal(0.0, 0.02, size=(dim, d.num_classes)).astype(np.float32) for d in domains
        }
        self.heads_b = {d.name: np.zeros((d.num_classes,), dtype=np.float32) for d in domains}

    def forward(self, tokens: np.ndarray, domain: str) -> np.ndarray:
        """tokens: (B, T, D), use CLS token at index 0."""
        if domain not in self.adapters:
            raise KeyError(f"Unknown domain '{domain}'. Expected one of {self.domain_order}")

        x = self.backbone.forward(tokens)
        cls = x[:, 0, :]
        cls = cls + self.adapters[domain].forward(cls)
        return cls @ self.heads_w[domain] + self.heads_b[domain]


def run_demo() -> Dict[str, tuple]:
    rng = np.random.default_rng(0)
    model = MultiDomainAdapterClassifier(
        dim=256,
        domains=[
            DomainSpec("s2", 2),
            DomainSpec("l89", 2),
            DomainSpec("s5p", 2),
        ],
        rank=8,
    )
    x = rng.normal(0.0, 1.0, size=(4, 197, 256)).astype(np.float32)  # batch x tokens x dim
    out = {d: model.forward(x, d).shape for d in ["s2", "l89", "s5p"]}
    return out


if __name__ == "__main__":
    shapes = run_demo()
    for domain, shape in shapes.items():
        print(f"{domain}: logits shape = {tuple(shape)}")
