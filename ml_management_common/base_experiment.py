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
import json
import os
from concurrent.futures import ThreadPoolExecutor, Future, wait as wait_for_futures
from contextlib import contextmanager
from abc import ABC, abstractmethod
from typing import Any, Union, TYPE_CHECKING, Dict, Optional, Generator, Callable, TypeVar

import tempfile
from .ngas import NGASClient, NGASConfiguration

if TYPE_CHECKING:
    import numpy  # pylint: disable=unused-import
    import PIL  # pylint: disable=unused-import
    from matplotlib.figure import Figure as MatplotlibFigure  # pylint: disable=unused-import
    from bokeh.document import Document  # pylint: disable=unused-import
    from matplotlib.animation import AbstractMovieWriter  # pylint: disable=unused-import

T = TypeVar('T')


class BaseExperiment(ABC):

    class NGASWrapper(object):
        """
        Wrapper to provide a context manager for NGAS.
        """
        def __init__(
            self,
            parent: "BaseExperiment",
            ngas_client: Union[NGASClient, NGASConfiguration]
        ):
            """
            :param parent: The experiment that is using this object.
            :param ngas_client: The NGAS Client to use, or configuration for the NGAS client.
            """
            self.parent = parent
            if isinstance(ngas_client, NGASClient):
                self.ngas_client = ngas_client
            else:
                self.ngas_client = NGASClient(ngas_client)

        def download_file(self, artifact_uri: str, local_path: str):
            """
            Download a file from NGAS that was previously uploaded to NGAS and MLFlow via log_artifact or log_artifacts.
            :param artifact_uri: The URI of the artifact to download. This will be the JSON document in MLFlow.
            :param local_path: Local path to save the downloaded file to.
            :return:
            """
            ngas_path = self._get_artfiact_ngas_path(artifact_uri)
            # download this path from NGAS to the provided local path
            self.ngas_client.retrieve(ngas_path).write(local_path)

        def download_named_file(self, ngas_name: str, local_path: str, file_version: Optional[int] = None):
            """
            Download a file directly from NGAS by name.

            This can be used to download the the latest version of a file, or a specific version of a file, that was
            uploaded with log_artifact with an ngas_name specified.

            :param ngas_name: The name of the file to download.
            :param local_path: Local path to save the downloaded file to.
            :param file_version: Optional version of the file to download.
            """
            self.ngas_client.retrieve(ngas_name, file_version).write(local_path)

        def delete_named_file(self, ngas_name: str, file_version: Optional[int] = None):
            """
            Delete a file directly from NGAS by name
            :param ngas_name: The name of the file to delete.
            :param file_version: Optional version of the file to delete.
            """
            self.ngas_client.remfile(file_id=ngas_name, file_version=file_version)

        def delete_artifact_file(self, artifact_uri: str):
            """
            Deletes an artifact from NGAS, using the artfiact URI that it was logged to with log_artfiact.
            :param artifact_uri: The artfiact URI to delete.
            """
            ngas_path = self._get_artfiact_ngas_path(artifact_uri)
            self.delete_named_file(ngas_path)

        def log_artifact(
            self,
            local_path: str,
            artifact_path: Optional[str] = None,
            ngas_name: Optional[str] = None,
            ngas_version: Optional[int] = None
        ):
            """
            Log an artifact to the NGAS server.

            This will upload the given file to the NGAS server, and leave a link to the file in MLFlow via a JSON
            document so that the file can be retrieved from NGAS as needed.

            :param local_path: The path to the artifact on the local machine.
            :param artifact_path: Artifact path to use in MLFlow and NGAS. This is inserted before the basename of the
                local_path. If None, the basename of the local_path is used directly.
            :param ngas_name If provided, the artifact will be stored on the server using this exact name.
                This allows you to write files to a specific name to retrieve them with download_named_file.
            :param ngas_version: If provided, the artifact will be stored on the server using this exact version.
                Otherwise, the artifacts will be logged as a new version.
                Only useful when using ngas_name to specify the exact name to save the artifact to.
            """
            return self.parent._thread_execute(self._log_artifact, local_path, artifact_path, ngas_name, ngas_version)

        def _log_artifact(
            self,
            local_path: str,
            artifact_path: Optional[str] = None,
            ngas_name: Optional[str] = None,
            ngas_version: Optional[int] = None
        ):
            # upload artifact to NGAS
            local_path_base = os.path.basename(local_path)
            if ngas_name is not None:
                # Use the exact specified name
                ngas_path = ngas_name
            elif artifact_path is not None:
                ngas_path = os.path.join(self.parent.unique_run_id(), artifact_path, local_path_base)
            else:
                ngas_path = os.path.join(self.parent.unique_run_id(), local_path_base)
            result, filename = self.ngas_client.archive(
                local_path,
                ngas_path.replace('/', '_'),
                file_version=ngas_version
            )
            ngas_storage_info = vars(result)
            ngas_storage_info["ngas_path"] = filename
            with tempfile.TemporaryDirectory() as temp_directory:
                tempfilename = os.path.join(temp_directory, local_path_base)
                # Write linking file into temporary directory for upload as artifact.
                # this will contain the name of the file written to NGAS.
                with open(tempfilename, "w") as f:
                    json.dump(ngas_storage_info, f, indent=4, default=str)
                self.parent._log_artifact(tempfilename, artifact_path)

        def log_artifacts(self, local_directory_path: str, artifact_path: Optional[str] = None):
            """
            Log a directory of artifacts to the NGAS server.

            This will upload all files in the given directory to the NGAS server, and leave upload links to each
            file in MLFlow via a JSON document so that the files can be retrieved from NGAS as needed.

            :param local_directory_path: The path to the directory to upload.
            :param artifact_path: Artifact path to use in MLFlow and NGAS. This is inserted before the basename of each
                file in the local_directory_path. If None, the basename of each file is used directly.
            """
            return self.parent._thread_execute(self._log_artifacts, local_directory_path, artifact_path)

        def _get_artfiact_ngas_path(self, artifact_uri: str):
            # We expect the downloaded file from the machine learning server to be a JSON file created by
            # _log_artifact.
            with self.parent.download_temp_file(artifact_uri) as f:
                try:
                    with open(f, 'r') as json_file:
                        loaded_json = json.load(json_file)
                        if not isinstance(loaded_json, dict):
                            raise ValueError(f"The downloaded file {artifact_uri} was not a JSON object.")
                        return loaded_json["ngas_path"]

                except json.JSONDecodeError:
                    raise ValueError(f"The downloaded file {artifact_uri} was not valid JSON.")
                except KeyError:
                    raise ValueError(f"The downloaded file {artifact_uri} did not contain a 'ngas_path' key.")

        def _log_artifacts(self, local_directory_path: str, artifact_path: Optional[str] = None):
            for f in os.listdir(local_directory_path):
                if os.path.isfile(f):
                    p = os.path.join(artifact_path, f) if artifact_path is not None else None
                    self._log_artifact(os.path.join(local_directory_path, f), p)

    def __init__(
        self,
        upload_threads=2,
        ngas_client: Union[NGASClient, NGASConfiguration, None] = None
    ):
        """
        :param upload_threads: If non zero, uploads will be queued and performed in separate threads.
        Threads are used because the upload is IO bound and is sped up significantly despite GIL.
        :param ngas_client: If provided, this client will be used to upload artifacts to NGAS. If not provided here,
            it can be provided to the ngas() method when creating an ngas interface.
        """
        self._upload_threads = upload_threads
        self._upload_thread_pool: ThreadPoolExecutor
        self._futures_in_progress: set[Future] = set()
        self._ngas_client = ngas_client

    def __enter__(self):
        if self._upload_threads > 0:
            self._upload_thread_pool = ThreadPoolExecutor(max_workers=self._upload_threads)
            self._upload_thread_pool.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._upload_thread_pool:
            self._upload_thread_pool.__exit__(exc_type, exc_val, exc_tb)

    def ngas(self, ngas_client: Union[NGASClient, NGASConfiguration, None] = None):
        """
        Create an NGAS interface using the given NGASClient, NGASConfiguration, or the NGASClient / NGASConfiguration
        provided the the Experiment constructor.
        :param ngas_client: The NGASClient or NGASConfiguration to use.
        :return: An NGAS interface for logging artifacts to NGAS.
        """
        if self._ngas_client is None and ngas_client is None:
            raise ValueError("NGAS client needs to be passed to Experiment constructor, or to ngas() call")
        return self.NGASWrapper(self, ngas_client or self._ngas_client)

    @abstractmethod
    def unique_run_id(self) -> str:
        """
        :return: A unique run id for the current experiment
        """
        pass

    def wait_upload_jobs(self, timeout: Union[float, int, None] = None):
        """
        Waits for all upload jobs to complete
        :param timeout: Optional timeout to wait for, in seconds
        """
        wait_for_futures(self._futures_in_progress, timeout=timeout)

    def _thread_execute(self, function: Callable[..., T], *args, **kwargs) -> Union[Future[T], T]:
        if self._upload_thread_pool:
            future = self._upload_thread_pool.submit(function, *args, **kwargs)
            self._futures_in_progress.add(future)
            future.add_done_callback(self._thread_done)
            return future
        else:
            return function(*args, **kwargs)

    def _thread_done(self, future: Future[T]):
        self._futures_in_progress.remove(future)
        e = future.exception()
        if e:
            print(f"Exception in worker thread: {e}")

    @abstractmethod
    def log_text(self, text: str, artifact_path: str):
        """
        Log a piece of text as an artifact
        :param text: The string to log
        :param artifact_path: Artifact path to save the string to.
        It will be saved as a .txt file.
        """
        pass

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact.
        :param local_path: Path to the artifact to upload.
        :param artifact_path: Optional artifact path to log the file to.
        If not provided, the artifact will be logged with its local file name.
        """
        self._thread_execute(self._log_artifact, local_path, artifact_path)

    @abstractmethod
    def _log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact.
        To be implemented by the inheritor.
        :param local_path: Path to the artifact to upload.
        :param artifact_path: Optional artifact path to log the file to.
        If not provided, the artifact will be logged with its local file name.
        """
        pass

    def log_artifacts(self, local_directory_path: str, artifact_path: Optional[str] = None):
        """
        Log a folder of artifacts.
        :param local_directory_path: Local folder containing artifacts to log.
        All files in the directory are logged as artifacts
        :param artifact_path: Optional artifact base path to log the files to.
        If not provided, the artifacts will be logged with their local file names.
        """
        self._thread_execute(self._log_artifacts, local_directory_path, artifact_path)

    @abstractmethod
    def _log_artifacts(self, local_directory_path: str, artifact_path: Optional[str] = None):
        """
        Log a folder of artifacts.
        To be implemented by the inheritor.
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
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        class WrappedPDFPages(PdfPages):
            """
            Wrapper to provide a context manager for PdfPages.

            Use savefig() to save a figure to the PDF.
            """
            def __init__(self, parent: "BaseExperiment", filename: str, artifact_name: str):
                """
                :param parent: The experiment that is using this object.
                :param filename: The PDF Filename to write to.
                :param artifact_name: Artifact name to log the PDF to.
                """
                super().__init__(filename)
                self.artifact_name = artifact_name
                self.parent = parent
                self.plot_index = 0

            def savefig(self, figure=None, **kwargs):
                """
                Save a figure to the PDF.

                This will also log the figure to the ML Server.
                :param figure: The figure to save.
                :param kwargs: Additional arguments to pass to matplotlib.
                """
                super().savefig(figure, **kwargs)
                if figure is None:
                    figure = plt.gcf()
                title = ""
                if figure:
                    title = figure._suptitle.get_text()
                self.parent.log_figure(figure, f"pdf_{self.artifact_name}_{self.plot_index}_{title}")
                self.plot_index += 1

        with tempfile.TemporaryDirectory() as directory:
            temp_filename = os.path.join(directory, pdf_name)
            with WrappedPDFPages(self, temp_filename, os.path.basename(pdf_name or artifact_path or "")) as pdf:
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
        from matplotlib import animation

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
        import bokeh.io as bokeh_io

        class BokehWrapper(object):
            """
            Wrapper to provide a context manager for Bokeh.
            """
            def __init__(self, doc: "Document", temp_directory: str):
                """
                :param doc: The Bokeh Document to wrap.
                :param temp_directory: Local temporary directory to save Bokeh files to.
                """
                self.doc = doc
                self.temp_directory = temp_directory
                self.files: list[str] = []
                self.output_file_name: Optional[str] = None

            def output_file(self, name: str, title="Bokeh Plot", mode=None):
                """
                Set the Bokeh output file.

                See `output_file <https://docs.bokeh.org/en/latest/docs/reference/io.html#bokeh.io.output.output_file>`_
                for more information.

                :param name: The name of the file to save to.
                :param title: The title of the plot.
                :param mode: The mode to save the file in.
                """
                self.output_file_name = os.path.join(self.temp_directory, name)
                bokeh_io.output_file(self.output_file_name, title, mode)

            def save(self, obj, filename=None, resources=None, title=None, template=None, state=None):
                """
                Save the Bokeh Document to a file.

                See `save <https://docs.bokeh.org/en/latest/docs/reference/io.html#bokeh.io.save>`_
                for more information.
                """
                if filename is not None:
                    filename = os.path.join(self.temp_directory, filename)
                    self.files.append(filename)
                bokeh_io.save(obj, filename, resources, title, template, state)

        doc: "Document" = bokeh_io.curdoc()
        old_doc_state = doc.to_json_string()
        old_doc_title = doc.title

        doc.clear()

        try:
            with tempfile.TemporaryDirectory() as directory:
                wrapper = BokehWrapper(doc, directory)
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
        from .model_summary import model_summary
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
