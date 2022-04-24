import pytorch_lightning as pl
import torch
from pytorch_optimizer import Ranger21
from torch import nn
from transformers import AutoModel

from .time2vec import Time2Vec


class ProjectModel(pl.LightningModule):
    def __init__(
        self,
        hf_model_name: str = "facebook/convnext-tiny-224",
        t2v_out: int = 512,
        num_tf_nhead: int = 6,
        num_tf_layer: int = 6,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.image_model = AutoModel.from_pretrained(hf_model_name)

        self.t2v_out = t2v_out
        self.t2v = Time2Vec(1440, t2v_out, 18)

        tf_layer = nn.TransformerEncoderLayer(18, nhead=num_tf_nhead, batch_first=True)
        self.tf = nn.TransformerEncoder(tf_layer, num_tf_layer)

        self.regressor = nn.LazyLinear(1)

        self.loss = nn.L1Loss()

    def forward(self, image, meta):
        # (batch_size, hidden_size)
        img_output = self.image_model(image)[1]

        # (batch_size, t2v_out, 18)
        t2v_output = self.t2v(meta)
        # (batch_size, t2v_out, 18)
        tf_output = self.tf(t2v_output)
        # (batch_size, t2v_out * 18)
        tf_output = tf_output.view(tf_output.size(0), -1)

        # (batch_size, hidden_size + t2v_out * 18)
        x = torch.cat([img_output, tf_output], -1)
        x = self.regressor(x)
        return x.squeeze(1)

    def training_step(self, batch, batch_idx):
        image, meta, label = batch
        pred = self(image, meta)
        loss = self.loss(pred, label)
        return loss

    def configure_optimizers(self):
        optimizer = Ranger21(
            self.parameters(), num_iterations=self.trainer.estimated_stepping_batches
        )
        return optimizer
