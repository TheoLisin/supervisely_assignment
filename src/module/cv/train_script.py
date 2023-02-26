import sys
import random
import numpy as np
from torch import device as Device, initial_seed, Generator
from torch.cuda import is_available
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
from torch.nn import CrossEntropyLoss

from module.cv.model.dataset import AppleDataset
from module.cv.model.classification_model import ClassificationModel
from module.cv.model.train import train, test_cycle


LABEL_ENC = {
    "Apple A": 0,
    "Apple B": 1,
    "Apple C": 2,
    "Apple D": 3,
    "Apple E": 4,
    "Apple F": 5,
}


def seed_worker(worker_id):
    worker_seed = initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    g = Generator()
    g.manual_seed(0)
    size = (256, 256)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ],
    )
    train_dataset = AppleDataset(
        "/home/theo/interview_assignments/supervisely_assignment/src/module/cv/generation/generated_data",
        LABEL_ENC,
        transform=transform,
    )
    val_dataset = AppleDataset(
        "/home/theo/interview_assignments/supervisely_assignment/src/module/cv/generation/generated_test_data",
        LABEL_ENC,
        transform=transform,
    )
    test_dataset = AppleDataset(
        "/home/theo/interview_assignments/supervisely_assignment/src/module/cv/assets",
        LABEL_ENC,
        transform=transform,
        with_annot=False,
    )

    train_loader = DataLoader(
        train_dataset,
        32,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(val_dataset, 64, shuffle=False)
    test_loader = DataLoader(test_dataset, 64, shuffle=False)

    model = ClassificationModel(size, num_of_classes=6)
    device = Device('cuda' if is_available() else 'cpu')
    model.to(device)
    optim = Adam(model.parameters(), lr=1e-4)
    train(model, optim, device, train_loader, val_loader, epochs=5)
    test_cycle(
        model, CrossEntropyLoss(), test_loader, device, report=True,
    )


if __name__ == "__main__":
    main()