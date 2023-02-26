import sys
import numpy as np
from torch.optim import Optimizer
from torch import nn, no_grad, device as Device, Tensor
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm


def train(
    model: nn.Module,
    optim: Optimizer,
    device: Device,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 40,
    verbose: int = 10,
) -> None:
    sys.stdout.write("Start Training\n")
    sys.stdout.flush()
    loss = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()

        total_batches = 0
        total_loss = 0

        for batch, labels in tqdm(train_loader):
            total_batches += 1
            img = batch.to(device)
            logprobs = model(img)

            output = loss(logprobs, labels.to(device))
            optim.zero_grad()
            output.backward()
            optim.step()

            total_loss += output.item()

        sys.stdout.write(
            f"Epoch #{ep} | Loss: {total_loss / total_batches}\n"
        )
        sys.stdout.flush()

        if ep % verbose == 0:
            report = True
        else:
            report = False

        test_cycle(
            model, loss, test_loader, device, report
        )

    return model


def test_cycle(model: nn.Module, loss: nn.Module, loader: DataLoader, device: Device, report: bool):
    model.eval()
    with no_grad():
        total_batches = 0
        total_loss = 0
        preds = []
        targets = []

        for batch, labels in tqdm(loader):
            total_batches += 1
            img = batch.to(device)
            logprobs: Tensor = model(img)

            output = loss(logprobs, labels.to(device))
            total_loss += output.item()

            pred = logprobs.cpu().detach().numpy()
            pred = np.argmax(pred, axis=1)
            preds.extend(pred)
            targets.extend(labels.numpy())
    
    sys.stdout.write(
        f"Test Loss: {total_loss / total_batches}\n"
    )
    sys.stdout.flush()

    if report:
        sys.stdout.write(classification_report(targets, preds))
        sys.stdout.flush()
