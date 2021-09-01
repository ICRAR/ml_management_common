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

from .configuration import MLProjectConfiguration
from .model_summary import model_summary
from .task_types import TaskTypes
from .null_experiment import NullExperiment

from typing import Optional
from ruamel.yaml import YAML


def create_experiment(name: str, type: TaskTypes, configuration_path: Optional[str], **kwargs):
    """
    Creates an experiment, given a path to a YAML configuration file.
    The experiment server type (mlflow / clearml), and the server that the experiment connect to, are
    determined by the configuration file.

    Example configuration files:

    .. code-block:: yaml
        # Disable logging entirely
        # Note that you can also pass None to "configuration_path" to do the same thing
        server_type: none

    .. code-block:: yaml
        # Write result to a local directory, using mlflow
        project_name: ML-Chevron
        tracking_server: /home/sam/Projects/ML-Chevron/data
        output_uri: /home/sam/Projects/ML-Chevron/data
        server_type: mlflow

    .. code-block:: yaml
        # Write to a MLFLow server
        project_name: ML-Chevron
        tracking_server: http://localhost:25565
        output_uri: s3://mlflow.test/ml-chevron
        server_type: mlflow

    .. code-block:: yaml
        # Write to a ClearML server
        project_name: ML-Chevron
        output_uri: s3://clearml.test/ml-chevron
        server_type: clearml
        # note - clearml requires a clearml.conf file in your home directory: https://clear.ml/docs/latest/docs/configs/clearml_conf/

    Example usage:

    .. code-block:: python
        with create_experiment("Experiment Name", TaskTypes.training, "/configuration/file/path", **kwargs) as experiment:
            # use experiment.log methods
            experiment.log_artifact("/local/path/to/log")

    :param name: The name for this experiment.
    :param type: Experiment type. One of the TaskTypes enum
    :param configuration_path: Path to the configuration file. Set to None to disable the server entirely.
    :param kwargs: All arguments that will be used for this run. These are logged to the run.
    :return: An appropriate experiment instance, usable in a with statement.
    """
    Experiment = NullExperiment
    configuration = None
    if configuration_path is not None:
        try:
            with open(configuration_path, "r") as yaml_file:
                yaml = YAML().load(yaml_file)
                configuration = MLProjectConfiguration(
                    project_name=yaml.get("project_name", ""),
                    tracking_server=yaml.get("tracking_server", None),
                    output_uri=yaml.get("output_uri", None)
                )
                server_type = yaml.get("server_type", None)
                if server_type == "mlflow":
                    from .ml_flow import MLFlowExperiment
                    Experiment = MLFlowExperiment
                elif server_type == "clearml":
                    from .clear_ml import ClearMLExperiment
                    Experiment = ClearMLExperiment
                else:
                    # Invalid or intentionally omitted experiment
                    Experiment = NullExperiment
        except OSError:
            # On failure to open file, default to NullExperiment
            print(f"Failed to read experiment configuration file: ${configuration_path}")
            Experiment = NullExperiment

    return Experiment(name, type, configuration, kwargs)
