from pathlib import Path
from typing import Union

import pandas as pd
import PIL
import torch
from rich.progress import track
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor


def get_all_train_data(data_dir: Union[str, Path]) -> pd.DataFrame:
    train_dir = Path(data_dir) / "train"
    label_csvs = list(train_dir.rglob("label.csv"))

    def get_paths(img_name: str):
        "img_name: CASE74_01.png"
        case = img_name.split("_")[0]  # CASE74
        meta_name = img_name[:-4] + ".csv"
        # data\train\CASE74\image\CASE74_01.png
        img_dir = str(train_dir / case / "image" / img_name)
        # data\train\CASE74\meta\CASE74_01.csv
        meta_dir = str(train_dir / case / "meta" / meta_name)

        return pd.Series({"img_dir": img_dir, "meta_dir": meta_dir})

    result = []
    for csv_file in label_csvs:
        label_df = pd.read_csv(csv_file)
        r = label_df["img_name"].apply(get_paths)
        r["leaf_weight"] = label_df["leaf_weight"]
        result.append(r)

    return pd.concat(result).reset_index(drop=True)


def get_all_test_data(data_dir: Union[str, Path]):
    test_dir = Path(data_dir) / "test"
    images = [str(p) for p in test_dir.rglob("*.[pj][np]g")]
    metas = [str(p) for p in test_dir.rglob("*.csv")]
    return pd.DataFrame({"img_dir": images, "meta_dir": metas})


def process_meta(csv_path: str, is_train: bool = True) -> Union[torch.Tensor, None]:
    meta_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if is_train and meta_df.isna().sum().sum() > 10000:
        return None
    meta_df.interpolate("akima", inplace=True)
    meta_df.fillna(0, inplace=True)
    return torch.from_numpy(meta_df.values).float()


def process_img(img_path: str, feature_extractor):
    img = PIL.Image.open(img_path)
    pixel_values = feature_extractor(img, return_tensors="pt").pixel_values
    pixel_values = pixel_values.squeeze(0)
    return pixel_values


class TrainDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[str, Path],
        hf_model_name: str = "facebook/convnext-tiny-224",
    ):
        super().__init__()
        self.train_df = get_all_train_data(data_dir)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(hf_model_name)
        self.data = []

        for row in track(
            self.train_df.itertuples(), "[cyan]Train data...", total=len(self.train_df)
        ):
            img_path = row.img_dir
            meta_path = row.meta_dir
            leaf_weight = row.leaf_weight

            meta_data = process_meta(meta_path)
            if meta_data is None:
                continue

            img_data = process_img(img_path, self.feature_extractor)
            label = torch.tensor(leaf_weight)

            self.data.append((img_data, meta_data, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1], self.data[idx][2]


class TestDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[str, Path],
        hf_model_name: str = "facebook/convnext-tiny-224",
    ):
        super().__init__()
        self.test_df = get_all_test_data(data_dir)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(hf_model_name)
        self.data = []

        for row in track(
            self.test_df.itertuples(), "[green]Test data...", total=len(self.test_df)
        ):
            img_path = row.img_dir
            meta_path = row.meta_dir

            meta_data = process_meta(meta_path, is_train=False)
            img_data = process_img(img_path, self.feature_extractor)

            self.data.append((img_data, meta_data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]


if __name__ == "__main__":
    train_ds = TrainDataset("data")
    torch.save(train_ds, "data/train_data.pt")
    test_ds = TestDataset("data")
    torch.save(test_ds, "data/test_data.pt")
