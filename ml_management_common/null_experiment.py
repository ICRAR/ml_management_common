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
import re

import torch
from .base_experiment import BaseExperiment
from typing import Any, Union, TYPE_CHECKING, Dict, Optional, Callable

if TYPE_CHECKING:
    import numpy  # pylint: disable=unused-import
    import PIL  # pylint: disable=unused-import
    from matplotlib.figure import Figure as MatplotlibFigure  # pylint: disable=unused-import


def log(func):
    def inner(self):
        if self.logger:
            self.logger.info(f"NullExperiment.{func.func_name}")
    return inner


class NullExperiment(BaseExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = kwargs.get("logger", None)

    @log
    def log_text(self, text: str, artifact_path: str):
        pass

    @log
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        pass

    @log
    def log_artifacts(self, local_directory_path: str, artifact_path: Optional[str] = None):
        pass

    @log
    def _log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        pass

    @log
    def _log_artifacts(self, local_directory_path: str, artifact_path: Optional[str] = None):
        pass

    @log
    def log_dict(self, dictionary: Any, artifact_path: str):
        pass

    @log
    def log_image(self, image: Union["numpy.ndarray", "PIL.Image.Image"], artifact_path: str, **kwargs):
        pass

    @log
    def log_figure(self, figure: "MatplotlibFigure", artifact_path: str, **kwargs):
        pass

    @log
    def log_metric(self, key: str, value: float, **kwargs):
        pass

    @log
    def log_metrics(self, metrics: Dict[str, float], **kwargs):
        pass

    @log
    def log_model(self, pytorch_model, artifact_path: str, **kwargs):
        pass

    @log
    def set_tag(self, k: str, v: Any):
        pass

    @log
    def set_tags(self, dictionary: Dict[str, Any]):
        pass

    @log
    def report_model_summary(self, model, input_size, batch_size=-1, device=torch.device("cuda:0"), dtypes=None,
                             dot=None):
        pass

    @log
    def download_model(self, uri: str, model=None):
        pass

    def find_latest_file_uri(self, name: str, run_filter: Union[str, re.Pattern]):
        return None, None

    def download_file(self, uri: str, local_path: str):
        return None

