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
from typing import Optional, Dict, Any

import mlflow
import torch

from stream_logger_context import StreamLoggerContext
from ..task_types import TaskTypes
from ..configuration import MLProjectConfiguration
from ..model_summary import model_summary


class MLFlowExperiment(object):
    def __init__(
        self,
        run_name: str,
        task_type: TaskTypes,
        configuration: MLProjectConfiguration,
        dict_args: dict,
        run_id: str = None,
        nested: bool = False,
        tags: Optional[Dict[str, Any]] = None,
    ):
        self.run_name = run_name
        self.task_type = task_type
        self.configuration = configuration
        self.dict_args = dict_args
        self.run_id = run_id
        self.nested = nested
        self.tags = tags
        self.run: Optional[mlflow.ActiveRun] = None
        self.logger = StreamLoggerContext()

    def __enter__(self):
        self.logger.__enter__()
        mlflow.set_tracking_uri(self.configuration.output_uri)
        mlflow.set_experiment(self.configuration.project_name)
        self.run = mlflow.start_run(
            run_id=self.run_id,
            run_name=self.run_name,
            nested=self.nested,
            tags={"type": str(self.task_type), **self.tags}
        )
        mlflow.log_params(self.dict_args)
        mlflow.pytorch.autolog()
        self.run.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mlflow.log_text(self.logger.read_all(), "log.txt")
        self.run.__exit__(exc_type, exc_val, exc_tb)
        self.logger.__exit__(exc_type, exc_val, exc_tb)

    def report_model_summary(
            self,
            model,
            input_size,
            batch_size=-1,
            device=torch.device("cuda:0"),
            dtypes=None,
            dot=None,
    ):
        """
        Report the structure of a model using model_summary.
        This will print the structure to the clearml log.
        :param model: The model to log
        :param input_size: Expected model input size
        :param batch_size:
        :param device:
        :param dtypes:
        :param dot:
        :return:
        """
        mlflow.log_text(model_summary(model, input_size, batch_size, device, dtypes, dot), "model_summary.txt")

    def download_model(self, uri: str):
        return mlflow.pytorch.load_model(uri)
