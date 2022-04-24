from pathlib import Path

import pytorch_lightning as pl
import torch
import typer
from pytorch_lightning.callbacks import EarlyStopping, RichProgressBar

from project import ProjectDataModule, ProjectModel, TestDataset, TrainDataset

app = typer.Typer()


@app.command()
def prepare_dataset(
    hf_model_name: str = "facebook/convnext-tiny-224",
    data_dir: Path = typer.Option(
        "data", exists=True, file_okay=False, dir_okay=True
    ),  # noqa
):
    "훈련 데이터와 테스트 데이터를 Dataset 형태로 생성하고 저장합니다."
    save_name = hf_model_name.rsplit("/", maxsplit=1)[-1]
    train_ds = TrainDataset("data", hf_model_name)
    torch.save(train_ds, Path(data_dir) / f"train_{save_name}.pt")
    test_ds = TestDataset("data", hf_model_name)
    torch.save(test_ds, Path(data_dir) / f"test_{save_name}.pt")


@app.command()
def train(
    hf_model_name: str = typer.Option(
        "facebook/convnext-tiny-224",
        "-m",
        "--model-name",
        help="사용할 transformers 모델 이름",
    ),
    data_dir: Path = typer.Option(
        "data",
        "-d",
        "--data-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="데이터 폴더의 경로",
    ),
    t2v_out: int = typer.Option(512, "--t2v", min=2, help="time2vec 레이어의 output 크기"),
    num_tf_nhead: int = typer.Option(6, "--nhead", help="transformer 레이어의 nhead"),
    num_tf_layer: int = typer.Option(6, "--tf-layer", help="transformer 인코더의 층 수"),
    batch_size: int = typer.Option(
        16,
        "-b",
        "--batch-size",
        help="훈련에 사용할 batch size, auto_scale_batch_size을 사용하므로 무시될 수 있음",
    ),
    epochs: int = typer.Option(20, "-e", "--epochs", help="epochs 수"),
    test: bool = typer.Option(False, help="fast_dev_run 옵션을 사용하여 테스트합니다."),
):
    "모델을 훈련합니다."
    model = ProjectModel(
        hf_model_name,
        t2v_out=t2v_out,
        num_tf_nhead=num_tf_nhead,
        num_tf_layer=num_tf_layer,
    )

    datamodule = ProjectDataModule(data_dir, hf_model_name, batch_size=batch_size)

    early_stop = EarlyStopping("val_loss")

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        logger=False,
        max_epochs=epochs,
        precision=16,
        callbacks=[RichProgressBar(), early_stop],
        auto_scale_batch_size=True,
        auto_lr_find=True,
        fast_dev_run=test,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    app()
