"""Minimal demo of EMIT hyperspectral -> target multispectral simulation.

This is a public-safe, toy example showing the core idea:
1) interpolate spectral response functions (SRFs),
2) integrate hyperspectral reflectance into target bands,
3) apply lightweight radiometric alignment.

No private paths, datasets, or training code are included.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class RadiometricConfig:
    low_pct: float = 1.0
    high_pct: float = 99.0
    out_scale: float = 10000.0
    out_offset_ratio: float = 0.8
    out_gain_ratio: float = 0.6


def build_srf_matrix(
    emit_wavelengths_nm: np.ndarray,
    target_wavelengths_nm: np.ndarray,
    target_srf: np.ndarray,
) -> np.ndarray:
    """Resample target SRF curves to EMIT wavelengths and L1-normalize each band.

    Args:
        emit_wavelengths_nm: (B_emit,)
        target_wavelengths_nm: (B_target_srf,)
        target_srf: (B_target_srf, C_target)
    Returns:
        matrix: (B_emit, C_target)
    """
    bands = target_srf.shape[1]
    matrix = np.zeros((emit_wavelengths_nm.shape[0], bands), dtype=np.float32)
    for b in range(bands):
        curve = np.interp(
            emit_wavelengths_nm,
            target_wavelengths_nm,
            target_srf[:, b],
            left=0.0,
            right=0.0,
        )
        matrix[:, b] = curve / (curve.sum() + 1e-12)
    return matrix


def hyperspectral_to_multispectral(
    emit_cube: np.ndarray,
    srf_matrix: np.ndarray,
) -> np.ndarray:
    """Bandpass integration: (H, W, B_emit) x (B_emit, C_target) -> (H, W, C_target)."""
    return np.einsum("hwb,bc->hwc", emit_cube, srf_matrix, optimize=True)


def percentile_align(
    multispectral_cube: np.ndarray,
    cfg: RadiometricConfig,
) -> np.ndarray:
    """Simple per-band percentile stretch + scale/offset."""
    out = np.zeros_like(multispectral_cube, dtype=np.float32)
    for b in range(multispectral_cube.shape[-1]):
        band = multispectral_cube[..., b]
        valid = band > 0
        if not np.any(valid):
            continue
        p_lo, p_hi = np.percentile(band[valid], [cfg.low_pct, cfg.high_pct])
        stretched = (band - p_lo) / (p_hi - p_lo + 1e-6)
        aligned = stretched * (cfg.out_scale * cfg.out_gain_ratio) + (cfg.out_scale * cfg.out_offset_ratio)
        out[..., b] = np.where(valid, np.clip(aligned, 0, 65535), 0)
    return out.astype(np.uint16)


def run_demo(seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    # Toy EMIT-like cube: H x W x B_emit
    h, w, b_emit = 128, 128, 80
    emit_waves = np.linspace(420, 2500, b_emit).astype(np.float32)
    emit_cube = rng.random((h, w, b_emit), dtype=np.float32) * 0.3

    # Toy target sensor with 7 bands and Gaussian SRFs
    b_target = 7
    srf_waves = np.linspace(420, 2500, 400).astype(np.float32)
    centers = np.array([480, 560, 655, 865, 1610, 2200, 2300], dtype=np.float32)
    widths = np.array([30, 35, 35, 40, 60, 80, 80], dtype=np.float32)
    target_srf = np.stack(
        [np.exp(-0.5 * ((srf_waves - c) / w0) ** 2) for c, w0 in zip(centers, widths)],
        axis=1,
    ).astype(np.float32)

    srf_matrix = build_srf_matrix(emit_waves, srf_waves, target_srf)
    simulated = hyperspectral_to_multispectral(emit_cube, srf_matrix)
    aligned = percentile_align(simulated, RadiometricConfig())
    return simulated, aligned


if __name__ == "__main__":
    sim, aligned = run_demo(seed=42)
    print("Simulated float cube shape:", sim.shape, sim.dtype)
    print("Aligned uint16 cube shape:", aligned.shape, aligned.dtype)
    print("Per-band mean (aligned):", aligned.reshape(-1, aligned.shape[-1]).mean(axis=0).round(2))
