"""Trains the tile classifier (16 classes: blank, 1..15) on synthetic crops and exports JSON weights.

Seeded and CPU-only. Writes the command line and seed to a `.meta.json` next to --out. Uses only
fonts/train; --extra crop sets are checked by synth.check_training_crops. Run via `npm run train`;
the exported weights are committed at src/vision/weights.json.
"""

from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing
import shlex
import sys
from pathlib import Path

import numpy as np
import synth
import torch
from torch import nn

CLASSES = 16


def make_boards(args: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return crops and labels for `count` rectified boards from one seed (a worker job)."""
    seed, count, crops_per_cell = args
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for i in range(count):
        size = 3 if i % 2 == 0 else 4
        grey, state, margin = synth.render_training_board(rng, size, synth.TRAIN_FONTS)
        edge = grey.shape[0]
        for index, value in enumerate(state):
            row, col = divmod(index, size)
            for _ in range(crops_per_cell):
                # The pipeline's own estimate of the margin and the quad fit are slightly off.
                est = margin + rng.normal(0, 0.008)
                cx, cy, pitch = synth.compute_cell_centre(row, col, size, est, edge)
                cx += rng.normal(0, 0.035) * pitch
                cy += rng.normal(0, 0.035) * pitch
                side = pitch * synth.CROP_WINDOW * float(rng.uniform(0.93, 1.07))
                # Jitter only the pixels the crop will read (plus a little), not the whole board.
                x0, y0 = max(0, int(cx - side)), max(0, int(cy - side))
                window = synth.augment_grey_window(
                    rng, grey[y0 : int(cy + side) + 2, x0 : int(cx + side) + 2]
                )
                xs.append(synth.sample_crop(window, cx - x0, cy - y0, side))
                ys.append(value)
    return np.stack(xs), np.array(ys, dtype=np.int64)


class TileNet(nn.Module):
    """Small CNN that maps a tile crop to 16 classes (blank, 1..15)."""

    def __init__(self) -> None:
        """Build the convolution and classifier layers."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the class logits for a batch of crops."""
        logits: torch.Tensor = self.fc(self.conv(x))
        return logits


def export_weights(model: TileNet, path: Path) -> None:
    """Write the model's layers to `path` as the JSON the TypeScript classifier loads."""

    def flatten_tensor(t: torch.Tensor | None) -> list[float]:
        if t is None:
            raise ValueError("layer has no bias")
        return [float(f"{v:.5g}") for v in t.detach().cpu().numpy().reshape(-1)]

    convs = [m for m in model.conv if isinstance(m, nn.Conv2d)]
    fcs = [m for m in model.fc if isinstance(m, nn.Linear)]
    payload = {
        "input": synth.CROP_PIXELS,
        "convs": [
            {
                "in": c.in_channels,
                "out": c.out_channels,
                "weights": flatten_tensor(c.weight),
                "bias": flatten_tensor(c.bias),
            }
            for c in convs
        ],
        "fcs": [
            {
                "in": f.in_features,
                "out": f.out_features,
                "weights": flatten_tensor(f.weight),
                "bias": flatten_tensor(f.bias),
            }
            for f in fcs
        ],
    }
    path.write_text(json.dumps(payload, separators=(",", ":")))


def write_metadata(path: Path, args: argparse.Namespace, validation_accuracy: float) -> None:
    """Record how the weights were made, next to them, so provenance does not depend on memory."""
    meta = {
        # Exactly what was run, for copy and paste
        # (via `npm run train`, python-env.sh adds the venv's python).
        "command": "python " + shlex.join(sys.argv),
        "seed": args.seed,
        "boards": args.boards,
        "epochs": args.epochs,
        "extra": [str(p) for p in args.extra],
        "weights": path.name,
        "validation_accuracy": round(validation_accuracy, 4),
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    sidecar = path.with_suffix(".meta.json")
    sidecar.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {sidecar}\n{json.dumps(meta, indent=2)}")


def main() -> None:
    """Train the classifier from the command line and export its weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--boards", type=int, default=9000)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--out", type=Path, default=synth.ROOT / "src" / "vision" / "weights.json")
    parser.add_argument(
        "--extra",
        type=Path,
        nargs="*",
        default=[],
        help=(
            "extra .npz crop sets (hard examples) with arrays x, y, fonts, source; "
            "refused if they touch eval fonts or photos"
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    workers = min(4, multiprocessing.cpu_count())
    chunk = args.boards // (workers * 4)
    jobs = [(args.seed * 1000 + i, chunk, 2) for i in range(workers * 4)]
    with multiprocessing.Pool(workers) as pool:
        parts = pool.map(make_boards, jobs)
    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    for extra in args.extra:
        data = np.load(extra)
        synth.check_training_crops(extra, data)
        x = np.concatenate([x, data["x"]])
        y = np.concatenate([y, data["y"]])
    order = np.random.default_rng(args.seed).permutation(len(x))
    x, y = x[order], y[order]
    split = int(len(x) * 0.95)
    x_train, y_train = torch.from_numpy(x[:split, None]), torch.from_numpy(y[:split])
    x_val, y_val = torch.from_numpy(x[split:, None]), torch.from_numpy(y[split:])
    print(f"{len(x_train)} train crops, {len(x_val)} validation crops")

    model = TileNet()
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=3e-3, total_steps=args.epochs * ((len(x_train) + 255) // 256)
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.02)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(x_train))
        for start in range(0, len(perm), 256):
            idx = perm[start : start + 256]
            optimiser.zero_grad()
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            loss.backward()
            optimiser.step()
            schedule.step()
        model.eval()
        with torch.no_grad():
            accuracy = (model(x_val).argmax(1) == y_val).float().mean().item()
        print(f"epoch {epoch + 1}/{args.epochs} validation accuracy {accuracy:.4f}")
    export_weights(model, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")
    write_metadata(args.out, args, accuracy)


if __name__ == "__main__":
    main()
