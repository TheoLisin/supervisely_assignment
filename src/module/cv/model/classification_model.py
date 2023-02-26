from typing import Tuple
from torch import nn
from torch import Tensor

class ClassificationModel(nn.Module):
    def __init__(self, input_size: Tuple[int, int], num_of_classes: int) -> None:
        super().__init__()
        h, w = input_size
        self.nofcl = num_of_classes

        self.conv_part = nn.Sequential(
            nn.Conv2d(3, 8, (3, 3), padding=1), 
            nn.ReLU(),
            nn.LayerNorm([8, h, w]),
            nn.MaxPool2d((2, 2), stride=2),  # 256
            nn.Conv2d(8, 32, (3, 3), padding=1), 
            nn.ReLU(),
            nn.LayerNorm([32, h // 2, w // 2]), 
            nn.MaxPool2d((2, 2), stride=2),  # 128
            nn.Conv2d(32, 64, (3, 3), padding=1), 
            nn.ReLU(),
            nn.LayerNorm([64, h // 4, w // 4]),
            nn.MaxPool2d((2, 2), stride=2),  # 64
            nn.Conv2d(64, 128, (3, 3), padding=1), 
            nn.ReLU(),
            nn.LayerNorm([128, h // 8, w // 8]),
            nn.MaxPool2d((2, 2), stride=2),  # 32
            # nn.Conv2d(128, 256, (3, 3), padding=1), 
            # nn.ReLU(),
            # nn.LayerNorm([h // 16, w // 16, 256]),
            # nn.MaxPool2d((2, 2), stride=2),  # 16
        )
        self.classification = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16 * 128, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, num_of_classes)
        )
    
    def forward(self, x: Tensor):
        x = self.conv_part(x)

        return self.classification(x)