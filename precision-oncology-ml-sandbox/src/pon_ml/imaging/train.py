from __future__ import annotations

from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig

from pon_ml.seeds import set_seeds
from pon_ml.tracking import init_mlflow, log_metrics, log_params, mlflow

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
except Exception as e:  # pragma: no cover - optional torch path
    torch = None  # type: ignore


class Tileset(Dataset):  # type: ignore[misc]
    def __init__(self, root: Path, size: int):
        self.paths: list[Path] = sorted((root / "benign").glob("*.png")) + sorted((root / "malignant").glob("*.png"))
        self.labels = [0] * len(sorted((root / "benign").glob("*.png"))) + [1] * len(
            sorted((root / "malignant").glob("*.png"))
        )
        self.tf = transforms.Compose([transforms.ToTensor(), transforms.Resize((size, size))])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        from PIL import Image

        x = self.tf(Image.open(self.paths[idx]).convert("RGB"))
        y = self.labels[idx]
        return x, y


class SmallCNN(nn.Module):  # pragma: no cover - exercised indirectly
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 2))

    def forward(self, x):  # type: ignore[override]
        h = self.conv(x)
        return self.head(h)


def grad_cam(model: SmallCNN, x: torch.Tensor) -> np.ndarray:  # pragma: no cover - small util
    # Simple Grad-CAM on last conv block
    activations = {}

    def hook(module, inp, out):
        activations["feat"] = out.detach()

    handle = model.conv[-3].register_forward_hook(hook)  # second conv layer
    model.eval()
    x.requires_grad_()
    out = model(x)
    cls = out.argmax(dim=1)
    score = out[0, cls]
    score.backward()
    fmap = activations["feat"][0]
    grads = model.conv[-3].weight.grad  # type: ignore
    weights = grads.mean(dim=(1, 2, 3)) if grads is not None else torch.ones(fmap.shape[0])
    cam = (weights.view(-1, 1, 1) * fmap).sum(0).clamp(min=0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
    handle.remove()
    return cam.detach().cpu().numpy()


@hydra.main(config_path="../../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:  # pragma: no cover
    set_seeds(cfg.seed)
    init_mlflow(cfg.mlflow.uri, cfg.mlflow.experiment)
    if torch is None:
        print("Torch not installed; skip imaging training.")
        return

    icfg = cfg.imaging
    root = Path(get_original_cwd()) / "data/imaging"
    if not root.exists():
        # Generate tiles if missing
        from pon_ml.data.gen_imaging import synthesize_tiles, save_pngs

        X, y = synthesize_tiles(icfg.image_size, icfg.samples_per_class, seed=cfg.seed)
        save_pngs(X, y, root)

    ds = Tileset(root, size=icfg.image_size)
    n = len(ds)
    ntr = int(0.8 * n)
    dtr, dva = torch.utils.data.random_split(ds, [ntr, n - ntr])
    dl_tr = DataLoader(dtr, batch_size=icfg.batch_size, shuffle=True)
    dl_va = DataLoader(dva, batch_size=icfg.batch_size)

    device = torch.device("cpu")
    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=icfg.lr)
    crit = torch.nn.CrossEntropyLoss()

    best_loss = float("inf")
    patience = 0
    with mlflow.start_run(run_name="imaging_cnn"):
        log_params({"epochs": icfg.epochs, "batch_size": icfg.batch_size})
        for epoch in range(icfg.epochs):
            model.train()
            for xb, yb in dl_tr:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = crit(logits, yb)
                loss.backward()
                opt.step()
            # val
            model.eval()
            val_losses = []
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in dl_va:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    loss = crit(logits, yb)
                    val_losses.append(loss.item())
                    pred = logits.argmax(1)
                    correct += int((pred == yb).sum())
                    total += int(yb.numel())
            vloss = float(np.mean(val_losses)) if val_losses else 0.0
            acc = correct / max(total, 1)
            log_metrics({"val_loss": vloss, "val_acc": acc}, step=epoch)
            if vloss + 1e-6 < best_loss:
                best_loss = vloss
                patience = 0
            else:
                patience += 1
                if patience >= icfg.early_stop_patience:
                    break

        # Grad-CAM for a sample
        try:
            xb, yb = next(iter(dl_va))
            cam = grad_cam(model, xb[:1])
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(3, 3))
            plt.imshow(cam, cmap="jet")
            out_dir = Path(get_original_cwd()) / "artifacts"
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / "gradcam_sample.png", bbox_inches="tight")
        except Exception:
            pass


if __name__ == "__main__":
    main()

