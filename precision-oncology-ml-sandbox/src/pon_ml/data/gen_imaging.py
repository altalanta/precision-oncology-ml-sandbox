from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from pon_ml.seeds import set_seeds


@dataclass
class ImageSpec:
    size: int
    samples_per_class: int


def _texture_noise(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(128, 30, size=(size, size)).astype(np.float32)
    # Add coarse noise
    for k in [8, 16, 32]:
        base += rng.normal(0, 10, size=(size, size)).astype(np.float32)
        base = base.astype(np.float32)
    base = np.clip(base, 0, 255)
    return base


def _draw_benign(img: Image.Image, seed: int) -> None:
    rng = np.random.default_rng(seed)
    draw = ImageDraw.Draw(img)
    # benign: round gland-like structures
    for _ in range(rng.integers(3, 7)):
        x0, y0 = rng.integers(0, img.width - 40), rng.integers(0, img.height - 40)
        x1, y1 = x0 + rng.integers(20, 50), y0 + rng.integers(20, 50)
        color = (rng.integers(150, 220), rng.integers(80, 160), rng.integers(150, 220))
        draw.ellipse([x0, y0, x1, y1], outline=color, width=2)


def _draw_malignant(img: Image.Image, seed: int) -> None:
    rng = np.random.default_rng(seed)
    draw = ImageDraw.Draw(img)
    # malignant: irregular polygons and streaks
    for _ in range(rng.integers(5, 10)):
        pts = [(int(rng.integers(0, img.width)), int(rng.integers(0, img.height))) for _ in range(rng.integers(3, 7))]
        color = (rng.integers(120, 200), rng.integers(0, 100), rng.integers(120, 200))
        draw.line(pts, fill=color, width=rng.integers(1, 3))


def synthesize_tiles(size: int, samples_per_class: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(samples_per_class):
        # benign
        noise = _texture_noise(size, seed + i)
        img = Image.fromarray(noise.astype(np.uint8)).convert("RGB")
        _draw_benign(img, seed + i * 2)
        img = img.filter(ImageFilter.SMOOTH)
        X.append(np.array(img))
        y.append(0)
        # malignant
        noise = _texture_noise(size, seed + 10_000 + i)
        img2 = Image.fromarray(noise.astype(np.uint8)).convert("RGB")
        _draw_malignant(img2, seed + 10_000 + i * 2)
        img2 = img2.filter(ImageFilter.SHARPEN)
        X.append(np.array(img2))
        y.append(1)
    return np.stack(X), np.array(y)


def save_pngs(X: np.ndarray, y: np.ndarray, out_dir: Path) -> None:
    out_benign = out_dir / "benign"
    out_malig = out_dir / "malignant"
    out_benign.mkdir(parents=True, exist_ok=True)
    out_malig.mkdir(parents=True, exist_ok=True)
    idx = 0
    for arr, label in zip(X, y):
        img = Image.fromarray(arr)
        path = (out_benign if label == 0 else out_malig) / f"tile_{idx:04d}.png"
        img.save(path)
        idx += 1


@hydra.main(config_path="../../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:  # pragma: no cover (thin wrapper)
    set_seeds(cfg.seed)
    icfg = cfg.imaging
    X, y = synthesize_tiles(icfg.image_size, icfg.samples_per_class, seed=cfg.seed)
    out = Path(get_original_cwd()) / "data/imaging"
    save_pngs(X, y, out)


if __name__ == "__main__":
    main()

