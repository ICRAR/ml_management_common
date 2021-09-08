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
from contextlib import contextmanager
from abc import ABC, abstractmethod
from typing import Any, Union, TYPE_CHECKING, Dict, Optional, Generator

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import animation
import tempfile

import bokeh.io as bokeh_io

from .model_summary import model_summary

if TYPE_CHECKING:
    import numpy  # pylint: disable=unused-import
    import PIL  # pylint: disable=unused-import
    from matplotlib.figure import Figure as MatplotlibFigure  # pylint: disable=unused-import
    from bokeh.document import Document  # pylint: disable=unused-import
    from matplotlib.animation import AbstractMovieWriter  # pylint: disable=unused-import


class BaseExperiment(ABC):

    class WrappedPDFPages(PdfPages):
        def __init__(self, parent, filename: str, artifact_name: str):
            super().__init__(filename)
            self.artifact_name = artifact_name
            self.parent: BaseExperiment = parent
            self.plot_index = 0

        def savefig(self, figure=None, **kwargs):
            super().savefig(figure, **kwargs)
            if figure is None:
                figure = plt.gcf()
            title = ""
            if figure:
                title = figure._suptitle.get_text()
            self.parent.log_figure(figure, f"pdf_{self.artifact_name}_{self.plot_index}_{title}")
            self.plot_index += 1

    class BokehWrapper(object):
        def __init__(self, doc: "Document", temp_directory: str):
            self.doc = doc
            self.temp_directory = temp_directory
            self.files: list[str] = []
            self.output_file_name: Optional[str] = None

        def output_file(self, name: str, title="Bokeh Plot", mode=None):
            self.output_file_name = os.path.join(self.temp_directory, name)
            bokeh_io.output_file(self.output_file_name, title, mode)

        def save(self, obj, filename=None, resources=None, title=None, template=None, state=None, **kwargs):
            if filename is not None:
                filename = os.path.join(self.temp_directory, filename)
                self.files.append(filename)
            bokeh_io.save(obj, filename, resources, title, template, state, **kwargs)

    @abstractmethod
    def log_text(self, text: str, artifact_path: str):
        """
        Log a piece of text as an artifact
        :param text: The string to log
        :param artifact_path: Artifact path to save the string to.
        It will be saved as a .txt file.
        """
        pass

    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact.
        :param local_path: Path to the artifact to upload.
        :param artifact_path: Optional artifact path to log the file to.
        If not provided, the artifact will be logged with its local file name.
        """
        pass

    @abstractmethod
    def log_artifacts(self, local_directory_path: str, artifact_path: Optional[str] = None):
        """
        Log a folder of artifacts.
        :param local_directory_path: Local folder containing artifacts to log.
        All files in the directory are logged as artifacts
        :param artifact_path: Optional artifact base path to log the files to.
        If not provided, the artifacts will be logged with their local file names.
        """
        pass

    @abstractmethod
    def log_dict(self, dictionary: Any, artifact_path: str):
        """
        Log a dict of values.
        :param dictionary: Dictionary containing the values to log.
        :param artifact_path: Artifact path to log the dict to
        """
        pass

    @abstractmethod
    def log_image(self, image: Union["numpy.ndarray", "PIL.Image.Image"], artifact_path: str, **kwargs):
        """
        Log an image.
        :param image: The image, as a numpy ndarray or PIL image.
        :param artifact_path: Artifact path to log the image to.
        The file extension used here determines the type of image to save.
        :param kwargs: Additional arguments for logging images. Dependent on subclasses.
        """
        pass

    @abstractmethod
    def log_figure(self, figure: "MatplotlibFigure", artifact_path: str, **kwargs):
        """
        Log a matplotlib figure
        :param figure: The figure to log.
        :param artifact_path: Artifact path to log the figure to.
        :param kwargs: Additional arguments for logging figures. Dependent on subclasses.
        """
        pass

    @abstractmethod
    def log_metric(self, key: str, value: float, **kwargs):
        """
        Log a single new metric value.
        :param key: Metric value name
        :param value: New metric value
        :param kwargs: Additional arguments for logging metrics. Dependent on subclasses.
        """
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], **kwargs):
        """
        Log a dict of metric values.
        Each entry in the dict will be logged as a new data point.
        :param metrics: Dictionary of metrics, with keys defining metric names and values defining new data points.
        :param kwargs: Additional arguments for logging metrics. Dependent on subclasses.
        """
        pass

    @abstractmethod
    def log_model(self, pytorch_model, artifact_path: str, **kwargs):
        """
        Logs a full pytorch model
        :param pytorch_model: The pytorch model.
        :param artifact_path: Artifact path to log the model to.
        :param kwargs: Additional arguments for logging the model. Dependent on subclasses.
        """
        pass

    @abstractmethod
    def set_tag(self, k: str, v: Any):
        """
        Sets a tag key to a value for the run.
        :param k: Tag key
        :param v: Tag value
        """
        pass

    @abstractmethod
    def set_tags(self, dictionary:  Dict[str, Any]):
        """
        Set multiple tags from a dictionary
        :param dictionary: Dictionary containing tag names as keys, and tag values as values
        """
        pass

    @contextmanager
    def log_figures_pdf(self, pdf_name: str, artifact_path: Optional[str] = None):
        """
        Creates a PdfPages instance to save matplotlib plots to in a with statement.
        :param pdf_name: Name of the PDF file to save
        :param artifact_path: Optional artifact path. If not provided, the pdf name will be used
        :return: Context that produces a PdfPages object.
        """
        with tempfile.TemporaryDirectory() as directory:
            temp_filename = os.path.join(directory, pdf_name)
            with BaseExperiment.WrappedPDFPages(self, temp_filename, os.path.basename(pdf_name or artifact_path or "")) as pdf:
                yield pdf
            self.log_artifact(temp_filename, artifact_path)

    @contextmanager
    def log_video(
            self,
            video_name: str,
            figure: "MatplotlibFigure",
            artifact_path: Optional[str] = None,
            fps=5,
            codec=None,
            bitrate=None,
            extra_args=None,
            metadata=None,
            dpi=None
    ) -> Generator["AbstractMovieWriter", Any, Any]:
        """
        Create a matplotlib video writer to write out a video file. Uses FFMpegWriter.
        :param video_name: The name of the video.
        :param figure: The figure to use to write the video. Passed to FFMpegWriter.saving.
        :param artifact_path: Optional artifact path. If not provided, the video name will be used
        :param fps: Frame rate of the animation in frames per second,
        :param codec: Video codec to use.
        :param bitrate: Bitrate to use.
        :param extra_args: Extra args for FFMpeg.
        :param metadata: Dictionary of video metadata,
        :param dpi: DPI of the figure when writing the video.
        :return: Context that produces a matplotlib video writer. The video is automatically saved on context exit.
        """
        with tempfile.TemporaryDirectory() as directory:
            temp_filename = os.path.join(directory, video_name)
            writer = animation.FFMpegWriter(
                fps=fps, codec=codec, bitrate=bitrate, extra_args=extra_args, metadata=metadata
            )
            with writer.saving(figure, temp_filename, dpi=dpi) as video:
                yield video
            self.log_artifact(temp_filename, artifact_path)

    @contextmanager
    def log_bokeh(self):
        """
        Creates a context to log bokeh plots to.
        The save and output_file bokeh functions should be called on the returned wrapper instead of
        the bokeh package.
        Each plot is uploaded with whatever name it is saved with.
        All plots are saved locally to a temporary directory.
        :return: Context that boken plots can be logged to.
        """
        doc: "Document" = bokeh_io.curdoc()
        old_doc_state = doc.to_json_string()
        old_doc_title = doc.title

        doc.clear()

        try:
            with tempfile.TemporaryDirectory() as directory:
                wrapper = BaseExperiment.BokehWrapper(doc, directory)
                yield wrapper
                for file in wrapper.files:
                    self.log_artifact(file)
                if wrapper.output_file_name is not None:
                    self.log_artifact(wrapper.output_file_name)
        finally:
            doc.from_json_string(old_doc_state)
            doc.title = old_doc_title

    @abstractmethod
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
        :param model:
        :param input_size:
        :param batch_size:
        :param device:
        :param dtypes:
        :param dot:
        :return:
        """
        pass

    def _get_model_summary(
            self,
            model,
            input_size,
            batch_size=-1,
            device=None,
            dtypes=None,
            dot=None
    ):
        return model_summary(model, input_size, batch_size, device, dtypes, dot)

    @abstractmethod
    def download_model(self, uri: str, model=None):
        """
        Downloads a previously trained model from a URI
        and returns it

        Some frameworks don't support the full re-creation of the
        model locally, and will require the model to be created and
        passed in the model parameter so that it can be loaded from
        a state dict.
        :param uri: URI of the saved model to load
        :param model: Model to load into. Only used if the framework does
        not support loading the model structure from a saved model
        """
        pass

    @abstractmethod
    def find_latest_file_uri(self, name: str):
        """
        Search through all runs in this experiment and find the URI of latest file with
        the given name.
        :param name: Name of the file to search for.
        :return: Tuple of file path and modified time, or (None, None) if no file could be found.
        """
        pass

    @abstractmethod
    def download_file(self, uri: str, local_path: str):
        """
        Downloads a file from the server and saves it locally.
        :param uri: The file's URI
        :param local_path: Local path to save the file to
        :return: Local path, or None if the file could not be downloaded.
        """
        pass

    @contextmanager
    def download_temp_file(self, uri: str):
        """
        Downloads a file from the server, saves it locally, and opens it, returning the opened File object.
        :raise Exception If the URI could not be found.
        :param uri: URI to download
        :return: Open file object.
        """
        with tempfile.TemporaryDirectory() as directory:
            local_file = self.download_file(uri, directory)
            if local_file is None:
                raise Exception(f"Could not download URI {uri}")
            yield local_file
