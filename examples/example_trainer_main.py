#
#  ICRAR - International Centre for Radio Astronomy Research
#  (c) UWA - The University of Western Australia, 2021
#  Copyright by UWA (in the framework of the ICRAR)
#  All rights reserved
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  This library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public
#  License along with this library; if not, write to the Free Software
#  Foundation, Inc., 59 Temple Place, Suite 330, Boston,
#  MA 02111-1307  USA
#
from argparse import ArgumentParser

import mlflow.pytorch
import numpy as np
import pytorch_lightning as pl
import torch
import tempfile
import os
from torch.utils.data import Dataset, DataLoader
from project_configuration import project_configuration
from example_trainer_model import Model
import matplotlib.pyplot as plt
from math import cos, pi
from matplotlib.backends.backend_pdf import PdfPages

from ml_management_common.ml_flow import MLFlowExperiment
from ml_management_common import TaskTypes

from ml_management_common.clear_ml import ClearMLExperiment

class SampleDataset(Dataset):

    def __init__(self, shape: int, size: int):
        self.data = np.ndarray(shape=(size, shape), dtype=np.float32)
        for i in range(size):
            t = (i / (size - 1)) * 2
            self.data[i] = np.cos(np.linspace(t * np.pi, (2 + t) * np.pi, shape))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item]


def arguments(parser: ArgumentParser):
    return parser


def main():
    parser = ArgumentParser(description="main")
    parser = arguments(parser)
    parser = Model.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)

    device = torch.device("cuda" if torch.cuda.is_available() and (dict_args.get("gpus", 0) > 0) else "cpu")

    #with MLFlowExperiment("main", TaskTypes.application, project_configuration, dict_args) as experiment:
    with ClearMLExperiment("main", TaskTypes.application, project_configuration, dict_args) as experiment:
        model = Model(**dict_args)
        model.to(device)
        experiment.report_model_summary(model, (64, ))

        trainer = pl.Trainer.from_argparse_args(args)
        trainer.fit(model, DataLoader(SampleDataset(64, 5000)))

        model.load_best_model()

        experiment.log_model(model, "best_model")

        for i in range(100):
            experiment.log_metric("example_metric1", cos(i * pi * 0.5))

        experiment.log_text("text here", "special.txt")
        experiment.log_dict({
            "key": "value",
            "key2": 100
        }, "dict.json")
        experiment.log_dict({
            "key": "value",
            "key2": 100
        }, "dict.yaml")
        experiment.log_image(
            np.array(np.random.random_integers(low=0, high=255, size=(320, 320, 3)), dtype=np.uint8),
            "special_image.png"
        )

        model.cpu()
        model.eval()
        with torch.no_grad():
            with experiment.log_figures_pdf("plots.pdf") as pdf:
                for i, sample in enumerate(SampleDataset(64, 10)):
                    fig = plt.figure(figsize=(8.27, 11.69))
                    fig.suptitle(f"Output example {i}")

                    ax = fig.add_subplot(1, 2, 1)
                    ax.set_title("Expected")
                    ax.plot(sample)

                    ax = fig.add_subplot(1, 2, 2)
                    ax.set_title("Output")
                    ax.plot(model(torch.Tensor(sample)).numpy())

                    pdf.savefig()
                    plt.close(fig)


if __name__ == "__main__":
    main()
