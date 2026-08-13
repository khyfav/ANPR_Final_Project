import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

from model import EMNISTCNN


def build_transform():
    # EMNIST is stored transposed/rotated relative to normal viewing.
    # Correct orientation here and keep the same 28x28 grayscale convention.
    return transforms.Compose([
        transforms.Lambda(
            lambda img: img.rotate(-90, expand=True).transpose(method=0)
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def accuracy(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x).argmax(1)

            correct += (pred == y).sum().item()
            total += y.numel()

    return correct / total if total else 0.0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--epochs',
        type=int,
        default=5
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=128
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3
    )

    parser.add_argument(
        '--data-dir',
        default='data/emnist'
    )

    # Save to a different model file so the 62-class model is preserved.
    parser.add_argument(
        '--out',
        default='../models/emnist_cnn_36.pt'
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------
    # Load full EMNIST ByClass dataset
    # --------------------------------------------------

    full_ds = datasets.EMNIST(
        args.data_dir,
        split='byclass',
        train=True,
        download=True,
        transform=build_transform()
    )

    print(f"Full EMNIST dataset: {len(full_ds):,} samples")

    # --------------------------------------------------
    # Filter to 36 classes:
    #
    # 0-9   = digits
    # 10-35 = uppercase A-Z
    #
    # Lowercase classes 36-61 are removed.
    # --------------------------------------------------

    keep_indices = torch.where(
        full_ds.targets < 36
    )[0].tolist()

    ds = Subset(
        full_ds,
        keep_indices
    )

    print(f"36-class dataset: {len(ds):,} samples")

    # --------------------------------------------------
    # Reproducible train/validation split
    # --------------------------------------------------

    generator = torch.Generator().manual_seed(42)

    val_n = int(
        0.10 * len(ds)
    )

    train_n = (
        len(ds) - val_n
    )

    train_ds, val_ds = random_split(
        ds,
        [train_n, val_n],
        generator=generator
    )

    print(f"Training samples:   {train_n:,}")
    print(f"Validation samples: {val_n:,}")

    # --------------------------------------------------
    # DataLoaders
    # --------------------------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------
    # 36-class CNN
    # --------------------------------------------------

    model = EMNISTCNN(
        num_classes=36
    ).to(device)

    print(
        f"Model is on: "
        f"{next(model.parameters()).device}"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    loss_fn = nn.CrossEntropyLoss()

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    for epoch in range(
        1,
        args.epochs + 1
    ):

        print(
            f"\nStarting epoch "
            f"{epoch}/{args.epochs}..."
        )

        model.train()

        running_loss = 0.0
        samples_seen = 0

        for batch_idx, (x, y) in enumerate(
            train_loader,
            start=1
        ):

            x = x.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad()

            outputs = model(x)

            loss = loss_fn(
                outputs,
                y
            )

            loss.backward()

            optimizer.step()

            batch_size = y.size(0)

            running_loss += (
                loss.item()
                * batch_size
            )

            samples_seen += batch_size

            # Print occasional progress
            if batch_idx % 500 == 0:
                print(
                    f"  Batch {batch_idx:,} | "
                    f"samples {samples_seen:,}/{train_n:,} | "
                    f"loss {loss.item():.4f}"
                )

        val_acc = accuracy(
            model,
            val_loader,
            device
        )

        epoch_loss = (
            running_loss / train_n
        )

        print(
            f"Epoch {epoch}: "
            f"train_loss={epoch_loss:.4f} "
            f"val_accuracy={val_acc:.4f}"
        )

    # --------------------------------------------------
    # Save trained model
    # --------------------------------------------------

    out = Path(
        args.out
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            'state_dict': model.state_dict(),
            'num_classes': 36
        },
        out
    )

    print(
        f"\nSaved 36-class model to {out}"
    )


if __name__ == '__main__':
    main()
