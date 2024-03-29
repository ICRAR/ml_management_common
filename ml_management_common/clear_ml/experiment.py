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
import datetime
import os.path
import re
import shutil
import tempfile

import numpy as np
import torch
import clearml
from typing import Optional, Sequence, Union, Mapping, Dict, Any, List, Tuple

from clearml.utilities.plotly_reporter import SeriesInfo
from matplotlib.figure import Figure as MatplotlibFigure

from ..base_experiment import BaseExperiment
from ..task_types import TaskTypes
from ..configuration import MLProjectConfiguration
from ..model_summary import model_summary


class ClearMLExperiment(BaseExperiment):
    def __init__(
            self,
            task_name: str,
            task_type: TaskTypes,
            configuration: MLProjectConfiguration,
            dict_args: dict,
            tags: Optional[Sequence[str]] = None,
            reuse_last_task_id: Union[bool, str] = True,
            continue_last_task: Union[bool, str] = False,
            auto_connect_arg_parser: Union[bool, Mapping[str, bool]] = True,
            auto_connect_frameworks: Union[bool, Mapping[str, bool]] = True,
            auto_resource_monitoring=True,
            auto_connect_streams: Union[bool, Mapping[str, bool]] = True,
            **kwargs
    ):
        """
        Create a clearml experiment. This should be used inside a `with` block.
        :param str task_name: The name of Task (experiment). If ``task_name`` is ``None``, the Python experiment
            script's file name is used. (Optional)
        :param TaskTypes task_type: The task type.

            Valid task types:

            - ``TaskTypes.training`` (default)
            - ``TaskTypes.testing``
            - ``TaskTypes.inference``
            - ``TaskTypes.data_processing``
            - ``TaskTypes.application``
            - ``TaskTypes.monitor``
            - ``TaskTypes.controller``
            - ``TaskTypes.optimizer``
            - ``TaskTypes.service``
            - ``TaskTypes.qc``
            - ``TaskTypes.custom``
        :param configuration Project wide configuration object that defines the URI and project name for this project
        :param dict_args Dict of all arguments that will be used for this task.
        :param tags: Add a list of tags (str) to the created Task. For example: tags=['512x512', 'yolov3']
        :param bool reuse_last_task_id: Force a new Task (experiment) with a previously used Task ID,
            and the same project and Task name.

            .. note::
               If the previously executed Task has artifacts or models, it will not be reused (overwritten)
               and a new Task will be created.
               When a Task is reused, the previous execution outputs are deleted, including console outputs and logs.

            The values are:

            - ``True`` - Reuse the last  Task ID. (default)
            - ``False`` - Force a new Task (experiment).
            - A string - You can also specify a Task ID (string) to be reused,
                instead of the cached ID based on the project/name combination.

        :param bool continue_last_task: Continue the execution of a previously executed Task (experiment)

            .. note::
                When continuing the executing of a previously executed Task,
                all previous artifacts / models/ logs are intact.
                New logs will continue iteration/step based on the previous-execution maximum iteration value.
                For example:
                The last train/loss scalar reported was iteration 100, the next report will be iteration 101.

            The values are:

            - ``True`` - Continue the the last Task ID.
                specified explicitly by reuse_last_task_id or implicitly with the same logic as reuse_last_task_id
            - ``False`` - Overwrite the execution of previous Task  (default).
            - A string - You can also specify a Task ID (string) to be continued.
                This is equivalent to `continue_last_task=True` and `reuse_last_task_id=a_task_id_string`.

        :param auto_connect_arg_parser: Automatically connect an argparse object to the Task

            The values are:

            - ``True`` - Automatically connect. (default)
            - ``False`` - Do not automatically connect.
            - A dictionary - In addition to a boolean, you can use a dictionary for fined grained control of connected
                arguments. The dictionary keys are argparse variable names and the values are booleans.
                The ``False`` value excludes the specified argument from the Task's parameter section.
                Keys missing from the dictionary default to ``True``, and an empty dictionary defaults to ``False``.

            For example:

            .. code-block:: py

               auto_connect_arg_parser={'do_not_include_me': False, }

            .. note::
               To manually connect an argparse, use :meth:`Task.connect`.

        :param auto_connect_frameworks: Automatically connect frameworks This includes patching MatplotLib, XGBoost,
            scikit-learn, Keras callbacks, and TensorBoard/X to serialize plots, graphs, and the model location to
            the **ClearML Server** (backend), in addition to original output destination.

            The values are:

            - ``True`` - Automatically connect (default)
            - ``False`` - Do not automatically connect
            - A dictionary - In addition to a boolean, you can use a dictionary for fined grained control of connected
                frameworks. The dictionary keys are frameworks and the values are booleans.
                Keys missing from the dictionary default to ``True``, and an empty dictionary defaults to ``False``.

            For example:

            .. code-block:: py

               auto_connect_frameworks={'matplotlib': True, 'tensorflow': True, 'tensorboard': True, 'pytorch': True,
                    'xgboost': True, 'scikit': True, 'fastai': True, 'lightgbm': True, 'hydra': True}

        :param bool auto_resource_monitoring: Automatically create machine resource monitoring plots
            These plots appear in in the **ClearML Web-App (UI)**, **RESULTS** tab, **SCALARS** sub-tab,
            with a title of **:resource monitor:**.

            The values are:

            - ``True`` - Automatically create resource monitoring plots. (default)
            - ``False`` - Do not automatically create.
            - Class Type - Create ResourceMonitor object of the specified class type.

        :param auto_connect_streams: Control the automatic logging of stdout and stderr

            The values are:

            - ``True`` - Automatically connect (default)
            -  ``False`` - Do not automatically connect
            - A dictionary - In addition to a boolean, you can use a dictionary for fined grained control of stdout and
                stderr. The dictionary keys are 'stdout' , 'stderr' and 'logging', the values are booleans.
                Keys missing from the dictionary default to ``False``, and an empty dictionary defaults to ``False``.
                Notice, the default behaviour is logging stdout/stderr the
                `logging` module is logged as a by product of the stderr logging

            For example:

            .. code-block:: py

               auto_connect_streams={'stdout': True, 'stderr': True, 'logging': False}

        """
        super().__init__(ngas_client=configuration.ngas_client, **kwargs)
        self.task_name = task_name
        self.task_type = task_type
        self.configuration = configuration
        self.tags = tags
        self.task: Optional[clearml.Task] = None
        self.dict_args = dict_args
        self.reuse_last_task_id = reuse_last_task_id
        self.continue_last_task = continue_last_task
        self.auto_connect_arg_parser = auto_connect_arg_parser
        self.auto_connect_frameworks = auto_connect_frameworks
        self.auto_resource_monitoring = auto_resource_monitoring
        self.auto_connect_streams = auto_connect_streams
        self.metric_iterations: Dict[str, int] = {}

    def __enter__(self):
        super().__enter__()
        self.task = clearml.Task.init(
            project_name=self.configuration.project_name,
            task_name=self.task_name,
            task_type=clearml.Task.TaskTypes[str(self.task_type)],
            tags=self.tags,
            output_uri=self.configuration.output_uri,
            reuse_last_task_id=self.reuse_last_task_id,
            continue_last_task=self.continue_last_task,
            auto_connect_arg_parser=self.auto_connect_arg_parser,
            auto_connect_frameworks=self.auto_connect_frameworks,
            auto_resource_monitoring=self.auto_resource_monitoring,
            auto_connect_streams=self.auto_connect_streams,
        )
        self.task.connect_configuration(self.dict_args)
        self.metric_iterations.clear()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.task.close()
        super().__exit__(exc_type, exc_val, exc_tb)

    def log_text(self, text: str, artifact_path: str):
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.basename(artifact_path)
            path = os.path.join(directory, filename)
            with open(path, "w") as f:
                f.write(text)
            self.log_artifact(path, artifact_path)

    def _log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        artifact_path = local_path.replace("/", "_").replace("\\", "_") if artifact_path is None else artifact_path
        self.task.upload_artifact(artifact_path, local_path, wait_on_upload=True)

    def _log_artifacts(self, local_directory_path: str, artifact_path: Optional[str] = None):
        self.task.upload_artifact(artifact_path, local_directory_path, wait_on_upload=True)

    def log_dict(self, dictionary: Dict[str, Any], artifact_path: str):
        self.task.logger.report_table(
            artifact_path,
            "Key, Value",
            table_plot=[(k, str(v)) for k, v in dictionary.items()]
        )

    def log_image(self, image: Union["numpy.ndarray", "PIL.Image.Image"], artifact_path: str, **kwargs):
        self.task.logger.report_image(artifact_path, artifact_path, image=image, **kwargs)

    def log_figure(self, figure: MatplotlibFigure, artifact_path: str, **kwargs):
        self.task.logger.report_matplotlib_figure(artifact_path, artifact_path, figure, **kwargs)

    def log_metric(self, key: str, value: float, **kwargs):
        next = self.metric_iterations.get(key, 0) + 1
        self.metric_iterations[key] = next

        self.task.logger.report_scalar(key, key, value, next)

    def log_metrics(self, metrics: Dict[str, float], **kwargs):
        for k, v in metrics.items():
            self.log_metric(k, v)

    def log_model(self, pytorch_model, artifact_path: str, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.basename(artifact_path)
            path = os.path.join(directory, filename)
            torch.save(pytorch_model, path)
            self.task.update_output_model(path, name=artifact_path, **kwargs)

    def set_tag(self, k: str, v: Any):
        if not self.tags:
            self.tags = []
        self.tags.append(f"{k}={v}")
        self.task.set_tags(self.tags)

    def set_tags(self, dictionary: Dict[str, Any]):
        if not self.tags:
            self.tags = []
        for k, v in dictionary.items():
            self.tags.append(f"{k}={v}")
        self.task.set_tags(self.tags)

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
        self.task.logger.report_text(model_summary(model, input_size, batch_size, device, dtypes, dot))

    def download_model(self, uri: str, model=None):
        """
        Download a model from the clearml shared storage, from the provided uri.
        May throw an exception if there is an error downloading the model, or if the uri
        is invalid.
        :param uri: The model's UI in shared storage
        :param model: Pre-created pytorch model to load the saved model into.
        :return: A loaded torch object containing the model weights in model["state_dict"]
        """
        model_path = clearml.Model(uri).get_local_copy()
        loaded_model_data = torch.load(model_path)
        model.load_state_dict(loaded_model_data["state_dict"])
        return model

    def find_latest_file_uri(self, name: str):
        raise Exception("Not implemented")

    def download_file(self, uri: str, local_path: str):
        path = clearml.StorageManager.get_local_copy(uri)
        if path is None:
            return None
        shutil.copy(path, local_path)
        return local_path

