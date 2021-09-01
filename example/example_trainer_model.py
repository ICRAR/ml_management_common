#
#    ICRAR - International Centre for Radio Astronomy Research
#    (c) UWA - The University of Western Australia, 2021
#    Copyright by UWA (in the framework of the ICRAR)
#    All rights reserved
#
#    This library is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 2.1 of the License, or (at your option) any later version.
#
#    This library is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with this library; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston,
#    MA 02111-1307  USA
#
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from argparse import ArgumentParser
from typing import List, Optional
from torch.nn import Sequential, Linear, LeakyReLU, Sigmoid


class Model(pl.LightningModule):
    """
    Example model to provide an example of how to implement other pytorch lightning models.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.learning_rate = kwargs["learning_rate"]
        self.best_loss = float("inf")
        self.best_state_dict = None

        self.encoder = Sequential(
            Linear(64, 32),
            LeakyReLU(),
            Linear(32, 16),
            LeakyReLU(),
            Linear(16, 8),
        )

        self.decoder = Sequential(
            Linear(8, 16),
            LeakyReLU(),
            Linear(16, 32),
            LeakyReLU(),
            Linear(32, 64),
            Sigmoid()
        )

    def forward(self, data):
        return self.decode(self.encode(data))

    def encode(self, data):
        return self.encoder(data)

    def decode(self, data):
        return self.decoder(data)

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=0.0005
        )
        return parser

    def setup(self, stage: Optional[str] = None):
        # reset best loss params for training
        self.best_loss = float("inf")
        self.best_state_dict = None

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = self.loss_function(out, batch)

        self.log("loss", loss.item())

        return {"loss": loss}

    def training_epoch_end(self, outputs: List[dict]):
        loss_epoch = sum(out["loss"] for out in outputs) / len(outputs)

        self.log("epoch_loss", loss_epoch)

        if loss_epoch < self.best_loss:
            self.best_loss = loss_epoch
            self.best_state_dict = self.state_dict()

    def load_best_model(self):
        if self.best_state_dict is not None:
            self.load_state_dict(self.best_state_dict)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def loss_function(recons, input_):
        return F.mse_loss(input_, recons)
