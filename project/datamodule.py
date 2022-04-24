from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from .dataset import TestDataset, TrainDataset  # noqa


class ProjectDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path],
        hf_model_name: str = "facebook/convnext-tiny-224",
        batch_size: int = 16,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        saved_name = hf_model_name.rsplit("/", maxsplit=1)[-1]
        self.train_data_dir = self.data_dir / f"train_{saved_name}.pt"
        self.test_data_dir = self.data_dir / f"test_{saved_name}.pt"
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", None):
            full_train_data = torch.load(self.train_data_dir)
            len_train = int(len(full_train_data) * 0.9)
            self.train_ds, self.val_ds = random_split(
                full_train_data, [len_train, len(full_train_data) - len_train]
            )

        if stage in ("test", "predict", None):
            self.test_ds = torch.load(self.test_data_dir)

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)
