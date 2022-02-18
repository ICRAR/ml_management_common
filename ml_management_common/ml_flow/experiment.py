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
import os
import uuid
from typing import Optional, Dict, Any, Union, TYPE_CHECKING

import mlflow
from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient, artifact_utils
from mlflow.tracking.artifact_utils import _download_artifact_from_uri

from .std_stream_capture import StdStreamCapture
from ..base_experiment import BaseExperiment
from ..configuration import MLProjectConfiguration
from ..task_types import TaskTypes

if TYPE_CHECKING:
    import numpy  # pylint: disable=unused-import
    import PIL  # pylint: disable=unused-import
    from matplotlib.figure import (
        Figure as MatplotlibFigure,
    )  # pylint: disable=unused-import


class MLFlowExperiment(BaseExperiment):
    def __init__(
        self,
        run_name: str,
        task_type: TaskTypes,
        configuration: MLProjectConfiguration,
        dict_args: dict,
        run_id: str = None,
        nested: bool = False,
        tags: Optional[Dict[str, Any]] = None,
        ignore: bool = False,
        **kwargs
    ):
        super().__init__(ngas_client=configuration.ngas_client, **kwargs)
        self.run_name = run_name
        self.task_type = task_type
        self.configuration = configuration
        self.dict_args = dict_args
        self.run_id = run_id
        self.nested = nested
        self.tags = tags
        self.run: Optional[mlflow.ActiveRun] = None
        self.ignore = ignore
        if not ignore:
            self.logger = StdStreamCapture()

    def __enter__(self):
        if self.ignore:
            return

        self.logger.__enter__()
        mlflow.set_tracking_uri(self.configuration.tracking_server)
        mlflow.set_experiment(self.configuration.project_name)
        self.run = mlflow.start_run(
            run_id=self.run_id,
            run_name=self.run_name,
            nested=self.nested,
            tags={"type": str(self.task_type), **(self.tags or {})},
        )

        def cap_string(s: str, length: int):
            return s[:length] if len(s) > length else s

        mlflow.log_params(
            {
                k: cap_string(str(v), 250)
                for k, v in self.dict_args.items()
                if not isinstance(v, dict)
            }
        )
        try:
            mlflow.pytorch.autolog(
                log_models=False  # Only autolog metrics, which are easy to deal with. Don't autolog bigger models
            )
        except ModuleNotFoundError as ignore:
            # this is an acceptable error, as it means we are not running on a pytorch environment
            print(
                "Pytorch-lightning was not found in the current venv, mlflow.pytorch.autolog will not be enabled"
            )

        self.run.__enter__()
        super().__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ignore:
            return

        super().__exit__(
            exc_type, exc_val, exc_tb
        )  # shutdown thread pool first to ensure all artifacts are uploaded
        mlflow.log_text(self.logger.read_all(), "log.txt")
        self.run.__exit__(exc_type, exc_val, exc_tb)
        self.logger.__exit__(exc_type, exc_val, exc_tb)

    def unique_run_id(self):
        return uuid.uuid4() if self.ignore else self.run.info.run_uuid

    def log_text(self, text: str, artifact_path: str):
        if self.ignore:
            return

        mlflow.log_text(text, artifact_path)

    def _log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        if self.ignore:
            return

        mlflow.log_artifact(local_path, artifact_path)

    def _log_artifacts(
        self, local_directory_path: str, artifact_path: Optional[str] = None
    ):
        if self.ignore:
            return

        mlflow.log_artifacts(local_directory_path, artifact_path)

    def log_dict(self, dictionary: Any, artifact_path: str):
        if self.ignore:
            return

        mlflow.log_dict(dictionary, artifact_path)

    def log_image(
        self,
        image: Union["numpy.ndarray", "PIL.Image.Image"],
        artifact_path: str,
        **kwargs
    ):
        if self.ignore:
            return

        mlflow.log_image(image, artifact_path)

    def log_figure(self, figure: "MatplotlibFigure", artifact_path: str, **kwargs):
        if self.ignore:
            return

        mlflow.log_figure(figure, artifact_path)

    def log_metric(self, key: str, value: float, **kwargs):
        if self.ignore:
            return

        mlflow.log_metric(key, value, step=kwargs.get("step"))

    def log_metrics(self, metrics: Dict[str, float], **kwargs):
        if self.ignore:
            return

        mlflow.log_metrics(metrics, step=kwargs.get("step"))

    def log_model(self, pytorch_model, artifact_path: str, **kwargs):
        if self.ignore:
            return

        mlflow.pytorch.log_model(pytorch_model, artifact_path, **kwargs)

    def set_tag(self, k: str, v: Optional[str]):
        if self.ignore:
            return

        mlflow.set_tag(k, v)

    def set_tags(self, dictionary: Dict[str, Any]):
        if self.ignore:
            return

        mlflow.set_tags(dictionary)

    def report_model_summary(
        self,
        model,
        input_size,
        batch_size=-1,
        device=None,
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
        if self.ignore:
            return

        if device is None:
            import torch

            device = torch.device("cuda:0")
        mlflow.log_text(
            self._get_model_summary(model, input_size, batch_size, device, dtypes, dot),
            "model_summary.txt",
        )

    def download_model(self, uri: str, model=None):
        """
        :param uri: Model URI to load from
        :param model: Not required for MLFlow
        :return: Loaded model from the URI
        """
        return None if self.ignore else mlflow.pytorch.load_model(uri)

    def find_latest_file_uri(self, name: str):
        if self.ignore:
            return None, None

        client = MlflowClient()
        best_artifact = None
        best_time = None
        for run in client.list_run_infos(self.run.info.experiment_id):
            if (
                RunStatus.from_string(run.status) != RunStatus.FINISHED
                or run.end_time is None
            ):
                continue  # Not finished, or has no end time ?

            if best_time is None or run.end_time > best_time:
                # search for artifact with name
                found = False
                for artifact in client.list_artifacts(run.run_id):
                    fname = os.path.basename(artifact.path)
                    if fname == name:
                        best_artifact = artifact_utils.get_artifact_uri(
                            run.run_id, artifact.path
                        )
                        found = True
                        break
                if found:
                    best_time = run.end_time

        return best_artifact, best_time

    def download_file(self, uri: str, local_path: str):
        return None if self.ignore else _download_artifact_from_uri(uri, local_path)
