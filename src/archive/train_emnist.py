import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import EMNISTCNN


def build_transform():
    # EMNIST is stored transposed/rotated relative to normal viewing.
    # Correct orientation here and keep the same 28x28 grayscale convention.
    return transforms.Compose([
        transforms.Lambda(lambda img: img.rotate(-90, expand=True).transpose(method=0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total if total else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--data-dir', default='data/emnist')
    p.add_argument('--out', default='models/emnist_cnn.pt')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = datasets.EMNIST(args.data_dir, split='byclass', train=True, download=True,
                         transform=build_transform())

    # Reproducible train/validation split.
    g = torch.Generator().manual_seed(42)
    val_n = int(0.10 * len(ds))
    train_n = len(ds) - val_n
    train_ds, val_ds = random_split(ds, [train_n, val_n], generator=g)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = EMNISTCNN(num_classes=62).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
        val_acc = accuracy(model, val_loader, device)
        print(f'Epoch {epoch}: train_loss={running/train_n:.4f} val_accuracy={val_acc:.4f}')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'num_classes': 62}, out)
    print(f'Saved model to {out}')


if __name__ == '__main__':
    main()
