import os
import pandas as pd
import cv2
from torch.utils.data import Dataset
from pathlib import Path
from typing import Union, Optional, Dict, Any


class AppleDataset(Dataset):
    def __init__(
        self,
        img_dir: Union[Path, str],
        label_enc: Dict[str, int],
        transform: Optional[Any] = None,
        with_annot: bool = True,
    ) -> None:
        self.with_annot = with_annot
        self.paths_and_labels = self._read_img_names(Path(img_dir), label_enc)
        self.transform = transform

    def _read_img_names(self, path: Path, label_enc: Dict[str, int]) -> pd.DataFrame:
        folders = os.listdir(path)
        data = {"image": [], "label": []}

        for folder in folders:
            img_path = path / folder
            if self.with_annot:
                img_path = img_path / "images"
            names = list(img_path.iterdir())
            data["image"].extend(names)
            data["label"].extend([label_enc[folder] for _ in range(len(names))])

        return pd.DataFrame.from_dict(data)

    def __len__(self):
        return len(self.paths_and_labels)

    def __getitem__(self, idx: int):
        img_path = self.paths_and_labels["image"][idx]
        label = self.paths_and_labels["label"][idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
