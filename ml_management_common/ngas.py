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
import io
import os
from contextlib import contextmanager

import requests
import glob

from abc import ABC, abstractmethod
from io import StringIO
from datetime import datetime
from enum import Enum
from urllib.parse import urlparse, urlunparse
from defusedxml.ElementTree import parse as xmlparse
from xml.etree.ElementTree import ElementTree, Element, tostring
from typing import TypeVar, Iterator, Optional, Union, Callable

from requests import Response
from requests.structures import CaseInsensitiveDict

T = TypeVar('T')


class NGASLoggingAdapter(ABC):
    """
    Logging interface that the NGAS client will use to log messages.
    """
    @abstractmethod
    def debug(self, msg: str):
        pass

    @abstractmethod
    def info(self, msg: str):
        pass

    @abstractmethod
    def warn(self, msg: str):
        pass

    @abstractmethod
    def error(self, msg: str):
        pass


class NullNGASLoggingAdapter(NGASLoggingAdapter):
    """
    Null logging adapter that does nothing
    """
    def debug(self, msg: str):
        pass

    def info(self, msg: str):
        pass

    def warn(self, msg: str):
        pass

    def error(self, msg: str):
        pass


class PrintNGASLoggingAdapter(NGASLoggingAdapter):
    """
    Default logging adapter that logs via print()
    """
    def debug(self, msg: str):
        print(f"{datetime.now()} DEBUG: {msg}")

    def info(self, msg: str):
        print(f"{datetime.now()} INFO: {msg}")

    def warn(self, msg: str):
        print(f"{datetime.now()} WARN: {msg}")

    def error(self, msg: str):
        print(f"{datetime.now()} ERROR: {msg}")


class PythonNGASLoggingAdapter(NGASLoggingAdapter):
    """
    Logging adapter to work with the 'logging' or 'loguru' logging libraries
    """
    def __init__(self, logger):
        self.logger = logger

    def debug(self, msg: str):
        self.logger.debug(msg)

    def info(self, msg: str):
        self.logger.info(msg)

    def warn(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)


class NGASConfiguration(object):
    """
    Configuration parameters for the NGAS client.

    This object can be created directly, or via the static method `from_dict()`
    """

    @staticmethod
    def from_dict(d: dict):
        """
        Create an NGAS configuration from a python dict.
        See the `__init__` method for the expected format.

        :param d: The dict
        :return: Configuration created from dict.
        """
        return NGASConfiguration(
            d.get('host', "localhost"),
            d.get('port', 7777),
            d.get('protocol', "http"),
            d.get('cache_dir', None),
            d.get('force_cache', False),
            d.get('logging', None)
        )

    def __init__(
        self,
        host="localhost",
        port=7777,
        protocol="http",
        cache_dir: Optional[str] = None,
        force_cache=False,
        logging: Union[NGASLoggingAdapter, bool, None] = None
    ):
        """
        Create a new NGAS configuration with the provided parameters.

        :param host: NGAS server host to connect to, defaults to localhost
        :param port: NGAS server port to connect to, defaults to 7777
        :param protocol: Protocol to use when communicating with the server, defaults to http
        :param cache_dir: Optional cache directory to use. If provided, the client will cache
            files in this directory and serve them from the directory if they exist, instead of
            requesting them from the server.
        :param force_cache: Do not perform any web requests and only serve files from the cache, if specified.
        :param logging: If False, no logging will be performed. If True, then a PrintNGASLoggingAdapter will
            be used. If a NGASLoggingAdapter is provided, then that adapter will be used.
        """
        self.host = host
        self.port = port
        self.protocol = protocol
        self.cache_dir = cache_dir
        self.force_cache = force_cache
        if logging is None or logging is False:
            self.logging = NullNGASLoggingAdapter()
        elif logging is True:
            self.logging = PrintNGASLoggingAdapter()
        else:
            self.logging = logging


class NGASResponse(object):
    """
    Base response object for NGAS requests.
    """
    def __init__(self, url: str, status: int, headers: CaseInsensitiveDict[str]):
        """
        :param url: The URL that the request was made to. May be a local path if the response was served from the cache.
        :param status: The HTTP status of the request.
        :param headers: Dict of headers that were returned with the response
        """
        self.http_url = url
        self.http_status = status
        self.headers = headers

    @property
    def http_ok(self):
        """
        :return: True if the HTTP status is a success status (200 range), or false otherwise.
        """
        return 200 <= self.http_status < 300


class NGASStatusResponse(NGASResponse):
    """
    Response object for NGAS methods that return an XML status as their response.
    """
    def __init__(self, url: str, status: int, headers: CaseInsensitiveDict[str], body: ElementTree):
        """
        :param url: The URL that the request was made to. May be a local path if the response was served from the cache.
        :param status: The HTTP status of the request.
        :param headers: Dict of headers that were returned with the response
        :param body: ElementTree root element of the XML response returned by the server.
        """
        super().__init__(url, status, headers)

        root = body.getroot().find("Status")
        self.date = datetime.fromisoformat(root.attrib["Date"])
        self.version = root.attrib["Version"]
        self.host_id = root.attrib["HostId"]
        self.message = root.attrib["Message"]
        self.status = root.attrib["Status"]
        self.state = root.attrib["State"]
        self.sub_state = root.attrib["SubState"]

    def __repr__(self):
        """
        :return: String representation of the response.
        """
        return f"NGASStatusResponse(url='{self.http_url}', status='{self.http_status}', date={self.date}, version={self.version}, host_id='{self.host_id}', message='{self.message}', status='{self.status}', state='{self.state}', sub_state='{self.sub_state}')"


class IterStream(io.RawIOBase):
    """
    Iterator-based stream that can be used as a file-like object.
    """

    def __init__(self, iterator: Iterator):
        """
        :param iterator: Iterator to use to generate data.
        """
        self.leftover = None
        self.iterator = iterator

    def readable(self):
        """
        :return: True, always readable.
        """
        return True

    def readinto(self, b):
        """
        Read data from the iterator into the given buffer.

        :param b: Buffer to write to.
        :return: Number of bytes written.
        """
        try:
            length = len(b)  # We're supposed to return at most this much
            chunk = self.leftover or next(self.iterator)
            output, self.leftover = chunk[:length], chunk[length:]
            b[:len(output)] = output
            return len(output)
        except StopIteration:
            return 0    # indicate EOF


class NGASFileResponse(NGASResponse):
    """
    Response object for NGAS methods that return a file as their response.
    """
    def __init__(
        self,
        url: str,
        status: int,
        headers: CaseInsensitiveDict[str],
        filename: str,
        data_iter: Optional[Iterator] = None,
        local_path: Optional[str] = None
    ):
        """
        One of data_iter or local_path must be provided, but not both.

        :exception ValueError : If both data_iter and local_path are provided, or neither are provided.
        :param url: The URL that the request was made to. May be a local path if the response was served from the cache.
        :param status: The HTTP status of the request.
        :param headers: Dict of headers that were returned with the response
        :param filename: Name of the file that was returned.
        :param data_iter: Optional data iterator that contains the HTTP response.
        :param local_path: Optional local path of a cached file that is being returned.
        """
        super().__init__(url, status, headers)
        self.filename = filename
        if bool(data_iter) ^ bool(local_path):
            raise ValueError("Either data_iter or local_path must be provided, but not both.")
        self.data_iter = data_iter
        self.local_path = local_path

    def __repr__(self):
        return f"NGASFileResponse(url='{self.http_url}', status='{self.http_status}', filename='{self.filename}')"

    @property
    def is_cached(self):
        """
        :return: True if the response is served from the cache.
        """
        return self.local_path is not None

    @contextmanager
    def stream(self):
        """
        Context manager that yields a file-like object that can be used to read the response.

        :return: file-like object to read the file from.
        """
        if self.is_cached:
            with open(self.local_path, 'rb') as f:
                yield f
        else:
            yield IterStream(self.data_iter)

    def write(self, filename: Optional[str] = None):
        """
        Write the response to a file.

        :param filename: Name of the file to write to. If not provided, the filename is taken from the response.
        :return: The final filename that was writen to.
        """
        final_filename = self.filename if filename is None else filename
        with open(final_filename, "wb") as f, self.stream() as s:
            f.write(s.read())
        return final_filename


class NGASException(Exception):
    """
    Exception thrown by the NGAS client if an error occurs while processing a request.
    """
    def __init__(self, message: str, response: Response, status: Optional[NGASStatusResponse] = None):
        """
        :param message: Error message.
        :param response: HTTP response object.
        :param status: Status response returned by the server.
        """
        super().__init__(f"{message}: {status.message}" if status is not None else message)
        self.response = response
        self.status = status


class NGASCommand(str, Enum):
    """
    Enumeration of all NGAS commands
    """

    STATUS = "STATUS"
    RETRIEVE = "RETRIEVE"
    CRETRIEVE = "CRETRIEVE"
    UNSUBSCRIBE = "UNSUBSCRIBE"
    SUBSCRIBE = "SUBSCRIBE"
    REGISTER = "REGISTER"
    REMFILE = "REMFILE"
    REMDISK = "REMDISK"
    OFFLINE = "OFFLINE"
    ONLINE = "ONLINE"
    LABEL = "LABEL"
    INIT = "INIT"
    EXIT = "EXIT"
    CREMOVE = "CREMOVE"
    CLIST = "CLIST"
    CDESTROY = "CDESTROY"
    CCREATE = "CCREATE"
    CAPPEND = "CAPPEND"
    CLONE = "CLONE"
    ARCHIVE = "ARCHIVE"
    QARCHIVE = "QARCHIVE"
    REARCHIVE = "REARCHIVE"
    CARCHIVE = "CARCHIVE"


def logcommand(command: NGASCommand):
    """
    Decorator that logs the command being executed.

    :param command: Command to log.
    :return: The decorated function.
    """
    def _logcommand(func: Callable[[...], T]):
        def wrap(*args, **kwargs) -> T:
            args[0].logging.info(f"Executing {command} command")
            try:
                return func(*args, **kwargs)
            except Exception as e:
                args[0].logging.error(f"Error executing {command} command: {e}")
                raise e
            finally:
                args[0].logging.info(f"Finished {command} command")
        return wrap
    return _logcommand


def parse_http_kv_list(header: str) -> dict[str, str]:
    """
    Parse a key, value list from an HTTP header

    :param header: Header to parse.
    :return: Dictionary of key, value pairs.
    """
    kv = {}
    for el in header.split(";"):
        parts = el.split("=")
        key = parts[0].strip("\" ")
        if len(parts) > 1:
            value = parts[1].strip("\" ")
        else:
            value = ""
        kv[key] = value
    return kv


def file_list_to_xml_string(file_list: list[str]) -> str:
    """
    Convert a list of files to an XML string.
    :param file_list: List of files to convert.
    :return: XML File list string.
    """
    root = Element("FileList")
    for file in file_list:
        root.append(Element('File', {'FileId': file}))
    return tostring(root, encoding="utf-8")


def is_known_pull_url(s: str):
    """
    Check if the given URL is a known pull URL for NGAS file storage requests.

    A pull URL uses one of the following protocols:
     - http:
     - https:
     - file:
     - ftp:

    :param s: URL to check
    :return: True if the url is a pull url, False otherwise.
    """
    return s.startswith('file:') or \
           s.startswith('http:') or \
           s.startswith('https:') or \
           s.startswith('ftp:')


class NGASClient(object):
    """
    NGAS client for interacting with an NGAS server
    """
    def __init__(self, config: NGASConfiguration, timeout=10):
        """
        :param config: NGAS configuration for the client to use.
        :param timeout: Requests time out after this many seconds. Default is 10 seconds.
        """
        self.config = config
        self.timeout = timeout
        self.logging = config.logging

    @logcommand(NGASCommand.EXIT)
    def exit(self):
        """
        NGAS `EXIT`_ command

        .. _EXIT: https://www.ngas.nrao.edu/NGASManual/html/ngas-commands.html#exit

        :exception NGASException: If the request fails.
        :return: NGAS Status Response.
        """
        return self._ngas_request_xml(NGASCommand.EXIT)

    @logcommand(NGASCommand.INIT)
    def init(self):
        """
        NGAS INIT command

        :exception NGASException: If the request fails.
        :return: NGAS Status Response.
        """
        return self._ngas_request_xml(NGASCommand.INIT)

    @logcommand(NGASCommand.LABEL)
    def label(self, slot_id: str, host_id: str):
        """
        NGAS LABEL command

        :exception NGASException: If the request fails.
        :return: NGAS Status Response.
        """
        return self._ngas_request_xml(NGASCommand.LABEL, {
            "slot_id": slot_id,
            "host_id": host_id
        })

    @logcommand(NGASCommand.OFFLINE)
    def offline(self):
        """
        NGAS `OFFLINE`_ command

        .. _OFFLINE: https://www.ngas.nrao.edu/NGASManual/html/ngas-commands.html#offline

        :exception NGASException: If the request fails.
        :return: NGAS Status Response.
        """
        return self._ngas_request_xml(NGASCommand.OFFLINE)

    @logcommand(NGASCommand.ONLINE)
    def online(self):
        """
        NGAS `ONLINE`_ command

        .. _ONLINE: https://www.ngas.nrao.edu/NGASManual/html/ngas-commands.html#offline

        :exception NGASException: If the request fails.
        :return: NGAS Status Response.
        """
        return self._ngas_request_xml(NGASCommand.ONLINE)

    @logcommand(NGASCommand.REMDISK)
    def remdisk(self, disk_id: str, execute=False):
        """
        NGAS `REMDISK`_ command

        .. _REMDISK: https://ngas.readthedocs.io/en/master/commands/storage.html#remdisk

        :exception NGASException: If the request fails.
        :param disk_id: ID of disk/media to remove from NGAS node.
        :param execute: False: is a dummy run which will only report what will happen if the command is executed.
            True: executes the command which will deleted the storage media and the associated files.
        :return: NGAS Status Response.
        """
        return self._ngas_request_xml(NGASCommand.REMDISK, {
            "disk_id": disk_id,
            "execute": "1" if execute else "0"
        })

    @logcommand(NGASCommand.REMFILE)
    def remfile(
        self,
        disk_id: Optional[str] = None,
        file_id: Optional[str] = None,
        file_version: Optional[int] = None,
        execute=False
    ):
        """
        NGAS `REMFILE`_ command

        .. _REMFILE: https://ngas.readthedocs.io/en/master/commands/storage.html#remfile

        :exception NGASException: If the request fails.
        :param disk_id: disk ID where the file to be deleted exists.
        :param file_id: ID of the file to be deleted.
        :param file_version: version of the file to be deleted.
        :param execute: False: is a dummy run which will only report what will happen if the command is executed.
            True: executes the command which will delete the file.
        :return: NGAS Status Response.
        """
        params = {"execute": "1" if execute else "0"}
        if disk_id is not None:
            params["disk_id"] = disk_id
        if file_id is not None:
            params["file_id"] = file_id
        if file_version is not None:
            params["file_version"] = str(file_version)
        return self._ngas_request_xml(NGASCommand.REMFILE, params)

    @logcommand(NGASCommand.REGISTER)
    def register(self, path: str, asyncronous=False):
        """
        NGAS `REGISTER`_ command

        .. _REGISTER: https://ngas.readthedocs.io/en/master/commands/storage.html#register

        :exception NGASException: If the request fails.
        :param path: The root path under which NGAS will look for candidate files to register. It is also possible to specify a complete path to a single file.
        :param asyncronous: Whether the registration should be carried out asynchronously or not.
        :return: NGAS Status Response.
        """
        return self._ngas_request_xml(NGASCommand.REGISTER, {
            "path": path,
            "async": '1' if asyncronous else '0'
        })

    @logcommand(NGASCommand.QARCHIVE)
    def qarchive(
        self,
        file: Union[str, os.PathLike, io.IOBase],
        filename: Optional[str] = None,
        mime_type="application/octet-stream",
        asyncronous=False,
        versioning=True,
        file_version: Optional[int] = None,
        **kwargs
    ):
        """
        NGAS `QARCHIVE`_ command

        .. _QARCHIVE: https://ngas.readthedocs.io/en/master/commands/storage.html#qarchive

        :exception NGASException: If the request fails.
        :param file: File, or path to file to archive.
            If the file is a remote file (http, ftp), it will be downloaded by the NGAS server and stored with
            the filename it has. Local files will be stored with their local file names, or the override filename
            if given
        :param filename: Override filename to use when storing a local file. If provided, this name will
            be used on the NGAS server instead of the file's name.
        :param mime_type: describes the content-type of the file. If not given, NGAS tries to guess it based on the
            filename’s extension, and the internal mime-type information stored in the NGAS configuration.
        :param asyncronous: Whether the registration should be carried out asynchronously or not.
        :param versioning: used to switch the automatic versioning on (True, the default behavior) or off (False).
        :param file_version: specifies which file version to overwrite. Only taken into account when versioning=False
        :param kwargs: Additional parameters to be passed to the command.
        :return: Tuple containing the Status returned by th server, and the name of the file.
        """
        return self._archive(
            NGASCommand.QARCHIVE,
            file,
            filename,
            mime_type,
            asyncronous,
            versioning,
            file_version,
            **kwargs
        )

    @logcommand(NGASCommand.ARCHIVE)
    def archive(
        self,
        file: Union[str, os.PathLike, io.IOBase],
        filename: Optional[str] = None,
        mime_type="application/octet-stream",
        asyncronous=False,
        versioning=True,
        file_version: Optional[int] = None,
        **kwargs
    ):
        """
        NGAS `ARCHIVE`_ command

        .. _ARCHIVE: https://ngas.readthedocs.io/en/master/commands/storage.html#archive

        :exception NGASException: If the request fails.
        :param file: File, or path to file to archive.
            If the file is a remote file (http, ftp), it will be downloaded by the NGAS server and stored with
            the filename it has. Local files will be stored with their local file names, or the override filename
            if given
        :param filename: Override filename to use when storing a local file. If provided, this name will
            be used on the NGAS server instead of the file's name.
        :param mime_type: describes the content-type of the file. If not given, NGAS tries to guess it based on the
            filename’s extension, and the internal mime-type information stored in the NGAS configuration.
        :param asyncronous: Whether the registration should be carried out asynchronously or not.
        :param versioning: used to switch the automatic versioning on (True, the default behavior) or off (False).
        :param file_version: specifies which file version to overwrite. Only taken into account when versioning=False
        :param kwargs: Additional parameters to be passed to the command.
        :return: Tuple containing the Status returned by th server, and the name of the file.
        """
        return self._archive(
            NGASCommand.ARCHIVE,
            file,
            filename,
            mime_type,
            asyncronous,
            versioning,
            file_version,
            **kwargs
        )

    @logcommand(NGASCommand.REARCHIVE)
    def rearchive(self, file_uri: str, file_info_xml: Element, **kwargs):
        """
        NGAS `REARCHIVE`_ command

        .. _REARCHIVE: https://ngas.readthedocs.io/en/master/commands/storage.html#rearchive

        :exception NotImplementedError: This command is not implemented yet.
        """
        raise NotImplementedError("Not implemented for now")

    @logcommand(NGASCommand.CLONE)
    def clone(
        self,
        file_id: Optional[str] = None,
        disk_id: Optional[str] = None,
        file_version: Optional[int] = None,
        target_disk_id: Optional[str] = None,
        asyncronous=False
    ):
        """
        NGAS `CLONE`_ command

        .. _CLONE: https://ngas.readthedocs.io/en/master/commands/storage.html#clone

        :exception NGASException: If the request fails.
        :param file_id: ID of the files to be cloned.
        :param disk_id: disk ID where the files to be cloned exist.
        :param file_version: file version of the files to be cloned.
        :param target_disk_id: ID of disk where the files cloned should be stored
        :param asyncronous: Whether the registration should be carried out asynchronously or not.
        :return: NGAS Status Response.
        """
        params = {"async": '1' if asyncronous else '0'}
        if file_id is not None:
            params["file_id"] = file_id
        if disk_id is not None:
            params["disk_id"] = disk_id
        if file_version is not None:
            params["file_version"] = str(file_version)
        if target_disk_id is not None:
            params["target_disk_id"] = target_disk_id
        return self._ngas_request_xml(NGASCommand.CLONE, params)

    @logcommand(NGASCommand.CARCHIVE)
    def carchive(self, directory: str, files_mtype: str, qarchive=False):
        """
        NGAS `CARCHIVE`_ command

        .. _CARCHIVE: https://ngas.readthedocs.io/en/master/commands/containers.html#carchive

        :exception NotImplementedError: This command is not implemented yet.
        """
        raise NotImplementedError("Not implemented for now as multipart container uploads are a bit complex")

    @logcommand(NGASCommand.CAPPEND)
    def cappend(
        self,
        file_id: Optional[str] = None,
        file_id_list: Optional[list[str]] = None,
        container_id: Optional[str] = None,
        container_name: Optional[str] = None,
        force=False,
        close_container=False
    ):
        """
        NGAS `CAPPEND`_ command

        .. _CAPPEND: https://ngas.readthedocs.io/en/master/commands/containers.html#cappend

        :exception NGASException: If the request fails.
        :exception ValueError: If both file_id and file_id_list are provided, or neither are provided.
        :exception ValueError: If both container_id and container_name are provided, or neither are provided.
        :param file_id: File ID to append to the container
        :param file_id_list: List of files to append to the container.
        :param container_id: ID of the container to append to.
        :param container_name: Name of the container to append to. Can be used instead of container_id
            if the name of the container uniquely identifies it.
        :param force: Whether to force the operation or not.
        :param close_container: Whether to close the container after appending the files.
        :return: NGAS Status Response.
        """
        if not (bool(container_id) ^ bool(container_name)):
            raise ValueError("Either container_id or container_name must be specified")
        if not (bool(file_id) ^ bool(file_id_list)):
            raise ValueError("Either file_id or file_id_list must be specified")

        params = {}
        if container_id is not None:
            params["container_id"] = container_id
        if container_name is not None:
            params["container_name"] = container_name
        if force:
            params["force"] = "1"
        if close_container:
            params["close_container"] = "1"

        if file_id is not None:
            params["file_id"] = file_id
            return self._ngas_request_xml(NGASCommand.CAPPEND, params)

        return self._ngas_request_xml(
            NGASCommand.CAPPEND,
            params,
            method="POST",
            headers={"Content-Type": "text/xml"},
            body=file_list_to_xml_string(file_id_list)
        )

    @logcommand(NGASCommand.CCREATE)
    def ccreate(
        self,
        container_name: Optional[str] = None,
        parent_container_id: Optional[str] = None,
        container_hierarchy: Optional[Element] = None
    ):
        """
        NGAS `CCREATE`_ command

        .. _CCREATE: https://ngas.readthedocs.io/en/master/commands/containers.html#ccreate

        :exception NGASException: If the request fails.
        :exception ValueError: If both container_name and container_hierarchy are provided, or neither are provided.
        :param container_name: Name of the container to create
        :param parent_container_id: The ID of the parent container.
        :param container_hierarchy: Container hierarchy to create. Can be used instead of container_name.
        :return: NGAS Status Response.
        """
        if not (bool(container_name) ^ bool(container_hierarchy)):
            raise ValueError("Either container_name or container_hierarchy must be specified")

        if container_name is not None:
            params = {"container_name": container_name}
            if parent_container_id is not None:
                params["parent_container_id"] = parent_container_id
            return self._ngas_request_xml(NGASCommand.CCREATE, params)

        return self._ngas_request_xml(
            NGASCommand.CCREATE,
            method="POST",
            headers={"Content-Type": "text/xml"},
            data=tostring(container_hierarchy, encoding="utf-8")
        )

    @logcommand(NGASCommand.CDESTROY)
    def cdestroy(self, container_name: Optional[str] = None, container_id: Optional[str] = None, recursive=False):
        """
        NGAS `CDESTROY`_ command

        .. _CDESTROY: https://ngas.readthedocs.io/en/master/commands/containers.html#cdestroy

        :exception NGASException: If the request fails.
        :exception ValueError: If both container_name and container_id are provided, or neither are provided.
        :param container_name: Name of the container to destroy.
        :param container_id: ID of the container to destroy. Can be used instead of container_name.
        :param recursive: If True, then children of this container will be removed recursively.
        :return: NGAS Status Response.
        """
        if not (bool(container_name) ^ bool(container_id)):
            raise ValueError("Either container_name or container_id must be specified")

        params = {}
        if container_name is not None:
            params["container_name"] = container_name
        if container_id is not None:
            params["container_id"] = container_id
        if recursive:
            params["recursive"] = "1"

        return self._ngas_request_xml(NGASCommand.CDESTROY, params)

    @logcommand(NGASCommand.CLIST)
    def clist(self, container_name: Optional[str] = None, container_id: Optional[str] = None):
        """
        NGAS `CLIST`_ command

        .. _CLIST: https://ngas.readthedocs.io/en/master/commands/containers.html#clist

        :exception NGASException: If the request fails.
        :exception ValueError: If both container_name and container_id are provided, or neither are provided.
        :param container_name: Name of the container to list.
        :param container_id: ID of the container to list. Can be used instead of container_name.
        :return: NGAS Status Response.
        """
        if not (bool(container_name) ^ bool(container_id)):
            raise ValueError("Either container_name or container_id must be specified")

        params = {}
        if container_name is not None:
            params["container_name"] = container_name
        if container_id is not None:
            params["container_id"] = container_id

        return self._ngas_request_xml(NGASCommand.CLIST, params)

    @logcommand(NGASCommand.CREMOVE)
    def cremove(
        self,
        file_id: Optional[str] = None,
        file_id_list: Optional[list[str]] = None,
        container_id: Optional[str] = None,
        container_name: Optional[str] = None
    ):
        """
        NGAS `CREMOVE`_ command

        .. _CREMOVE: https://ngas.readthedocs.io/en/master/commands/containers.html#cremove


        :exception NGASException: If the request fails.
        :exception ValueError: If both file_id and file_id_list are provided, or neither are provided.
        :exception ValueError: If both container_id and container_name are provided, or neither are provided.
        :param file_id: File to remove
        :param file_id_list: List of files to remove. Can be used instead of file_id.
        :param container_id: Container to remove files from.
        :param container_name: Container to remove files from. Can be used instead of container_id.
        :return: NGAS Status Response.
        """
        if not (bool(container_id) ^ bool(container_name)):
            raise TypeError("Either container_id or container_name must be specified")

        if not (bool(file_id) ^ bool(file_id_list)):
            raise TypeError("Either file_id or file_id_list must be specified")

        params = {}
        if container_id is not None:
            params["container_id"] = container_id
        if container_name is not None:
            params["container_name"] = container_name

        if file_id is not None:
            params["file_id"] = file_id
            return self._ngas_request_xml(NGASCommand.CREMOVE, params)

        if file_id_list is not None:
            return self._ngas_request_xml(
                NGASCommand.CREMOVE,
                params,
                method="POST",
                headers={'Content-Type': 'text/xml'},
                data=file_list_to_xml_string(file_id_list)
            )

        # Should not be able to get here
        raise TypeError("Either file_id or file_id_list must be specified")

    @logcommand(NGASCommand.CRETRIEVE)
    def cretrieve(self, container_name: Optional[str], container_id: Optional[str] = None):
        """
        NGAS `CRETRIEVE`_ command

        .. _CRETRIEVE: https://ngas.readthedocs.io/en/master/commands/containers.html#cretrieve

        :exception NGASException: If the request fails.
        :exception ValueError: If both container_name and container_id are provided, or neither are provided.
        :exception RuntimeError: If force_cache=True
        :param container_name: Name of the container to retrieve.
        :param container_id: ID of the container to retrieve. Can be used instead of container_name.
        :return: NGAS File Response of a TAR archive that contains all of the files in the container.
        """
        if self.config.force_cache:
            # Prevent the upload in force_cache mode, since the user is trying to conserve
            # internet bandwidth.
            raise RuntimeError("Cannot use cretrieve with force_cache=True")

        if not container_name and not container_id:
            raise ValueError("Must specify container_id or container_name")

        params = {
            # Easier to always accept tar files instead of needing to split the response into separate multipart files.
            "format": "application/x-tar"
        }
        if container_name is not None:
            params["container_name"] = container_name
        if container_id is not None:
            params["container_id"] = container_id

        return self._ngas_request_file(NGASCommand.CRETRIEVE, params)

    @logcommand(NGASCommand.RETRIEVE)
    def retrieve(
        self,
        file_id: str,
        file_version: Optional[int] = None,
        processing: Optional[str] = None,
        processing_pars: Optional[str] = None,
        ignore_cache: bool = False,
    ):
        """
        NGAS `RETRIEVE`_ command

        .. _RETRIEVE: https://ngas.readthedocs.io/en/master/commands/storage.html#retrieve

        :exception NGASException: If the request fails.
        :exception RuntimeError: Thrown if force_cache is set to True and the file is not in the cache.
        :param file_id: The NGAS ID of the file to retrieve.
        :param file_version: The version of the file to retrieve. If not provided, the latest file is retrieved.
        :param processing: Processing plugin to apply on the server.
        :param processing_pars: Parameters for the processing plugin.
        :param ignore_cache: If true, the file will always be downloaded from the NGAS server even if it exists
            in the cache directory. If the force_cache parameter is set to true, then an exception will be thrown.
        :return: An NGASFileResponse that contains the file, either referring to local cache or to the http file
            stream directory.
        """
        params = {
            "file_id": file_id,
        }
        if file_version is not None:
            params["file_version"] = str(file_version)
        if processing is not None:
            params["processing"] = processing
            if processing_pars is not None:
                params["processing_pars"] = processing_pars

        if self.config.cache_dir is not None and ignore_cache is False:
            # Check the cache first for this file.
            # Check exact
            cache_path = self._find_cache_path(file_id, file_version)
            if cache_path is not None:
                # We have a cached version of it
                return NGASFileResponse(cache_path, 200, CaseInsensitiveDict(), file_id, local_path=cache_path)

        if self.config.force_cache:
            # Prevent the upload in force_cache mode, since the user is trying to conserve
            # internet bandwidth.
            raise RuntimeError("No cached version, will not download because force_cache=True")

        response = self._ngas_request_file(NGASCommand.RETRIEVE, params)
        if response.http_ok and self.config.cache_dir:
            # Cache the file
            name = file_id
            if file_version is not None:
                name += f"_{file_version}"
            cache_path = os.path.join(self.config.cache_dir, name)
            response.write(cache_path)
            return NGASFileResponse(cache_path, 200, response.headers, file_id, local_path=cache_path)

        return response

    @logcommand(NGASCommand.SUBSCRIBE)
    def subscribe(
        self,
        url: str,
        subscr_id: str,
        concurrent_threads: Optional[int] = None,
        priority: Optional[int] = None,
        start_date: Optional[str] = None,
        filter_plug_in: Optional[str] = None,
        plug_in_pars: Optional[str] = None
    ):
        """
        NGAS `SUBSCRIBE`_ command

        .. _SUBSCRIBE: https://ngas.readthedocs.io/en/master/commands/others.html#subscribe

        :exception NGASException: If the request fails.
        :param url: The URL to which the archived file(s) will be delivered.
        :param subscr_id: Subscription ID that should be unique.
        :param concurrent_threads: Number of simultaneous file data delivery threads.
        :param priority: Priority for delivering data to this Data Subscriber. The lower the number, the higher t
            he priority. Clients with a higher priority, get more CPU time in connection with the data delivery.
        :param start_date: Date from which the data to deliver is taken into account. If not specified the time when
            the SUBSCRIBE command was received is taken as start date.
        :param filter_plug_in: Name of a Filter Plug-In to invoke on the file(s).
        :param plug_in_pars: A set of parameters to transfer to the Filter Plug-In when it is invoked.
        :return: NGAS Status Response.
        """
        params = {"url": url, "subscr_id": subscr_id}
        if priority is not None:
            params["priority"] = priority
        if start_date is not None:
            params["start_date"] = start_date
        if filter_plug_in is not None:
            params["filter_plug_in"] = filter_plug_in
        if plug_in_pars is not None:
            params["plug_in_pars"] = plug_in_pars
        if concurrent_threads is not None:
            params["concurrent_threads"] = concurrent_threads
        return self._ngas_request_xml(NGASCommand.SUBSCRIBE, params)

    @logcommand(NGASCommand.UNSUBSCRIBE)
    def unsubscribe(self, subscr_id: str):
        """
        NGAS `UNSUBSCRIBE`_ command

        .. _UNSUBSCRIBE: https://ngas.readthedocs.io/en/master/commands/others.html#unsubscribe

        :exception NGASException: If the request fails.
        :param subscr_id: Subscription ID to unsubscribe.
        :return: NGAS Status Response.
        """
        return self._ngas_request_xml(NGASCommand.UNSUBSCRIBE, {
            "subscr_id": subscr_id
        })

    @logcommand(NGASCommand.STATUS)
    def status(self) -> NGASStatusResponse:
        """
        NGAS `STATUS`_ command

        .. _STATUS: https://ngas.readthedocs.io/en/master/commands/core.html#status

        :exception NGASException: If the request fails.
        :return: NGAS Status Response.
        """
        return self._ngas_request_xml(NGASCommand.STATUS)

    def _ngas_request_file(
        self,
        command: NGASCommand,
        params: Optional[dict[str, str]] = None,
        headers: Optional[dict[str, str]] = None,
        method="GET",
        response_file_chunk_size=8192,
        **kwargs
    ):
        """
        Perform an NGAS request and return the response as a file stream.

        :exception NGASException: If the request fails.
        :param command: The NGAS command to perform.
        :param params: Parameters to pass to the command.
        :param headers: Headers to pass to the command.
        :param method: HTTP method to use.
        :param response_file_chunk_size: Size of the chunks to read from the response file.
        :param kwargs: Additional keyword arguments to pass to the `requests.request`_ method.
        :return: NGAS File Response.
        """
        response, url_str = self._ngas_request(command, params, headers, method, stream=True, **kwargs)
        disposition = parse_http_kv_list(response.headers.get("Content-Disposition", ""))
        self.logging.debug(f"Content-Disposition: {disposition}")
        iterator = response.iter_content(chunk_size=response_file_chunk_size)
        file_response = NGASFileResponse(
            url_str,
            response.status_code,
            response.headers,
            disposition.get("filename", ""),
            iterator
        )
        self.logging.debug(f"{file_response}")
        return file_response

    def _ngas_request_xml(
        self,
        command: NGASCommand,
        params: Optional[dict[str, str]] = None,
        headers: Optional[dict[str, str]] = None,
        method="GET",
        **kwargs
    ):
        """
        Perform an NGAS request and return the response as an XML document.

        :exception NGASException: If the request fails.
        :param command: The NGAS command to perform.
        :param params: Parameters to pass to the command.
        :param headers: Headers to pass to the command.
        :param method: HTTP method to use.
        :param kwargs: Additional keyword arguments to pass to the `requests.request`_ method.
        :return: NGAS Status Response.
        """
        response, url_str = self._ngas_request(command, params, headers, method, **kwargs)
        status = NGASStatusResponse(url_str, response.status_code, response.headers, xmlparse(StringIO(response.text)))
        self.logging.debug(f"{status}")
        return status

    def _ngas_request(
        self,
        command: NGASCommand,
        params: Optional[dict[str, str]] = None,
        headers: Optional[dict[str, str]] = None,
        method="GET",
        **kwargs
    ):
        """
        Perform an NGAS request and return the response.

        :exception NGASException: If the request fails.
        :param command: The NGAS command to perform.
        :param params: Parameters to pass to the command.
        :param headers: Headers to pass to the command.
        :param method: HTTP method to use.
        :param kwargs: Additional keyword arguments to pass to the `requests.request`_ method.
        :return: Tuple containing the response and the URL string.
        """
        url = urlparse(f"{self.config.protocol}://{self.config.host}:{self.config.port}/{command}")

        if params is not None:
            parts = [f'{k}={v}' for k, v in params.items()]
            url = url._replace(query='&'.join(parts))

        url_str: str = urlunparse(url)
        self.logging.debug(f"Making {method} request: {url_str}, with headers: {str(headers)}")
        response = requests.request(
            method,
            url_str,
            allow_redirects=True,
            timeout=self.timeout,
            headers=headers,
            **kwargs
        )

        if not response.ok:
            try:
                status = NGASStatusResponse(
                    url_str,
                    response.status_code,
                    response.headers,
                    xmlparse(StringIO(response.text))
                )
            except Exception as e:
                status = None
            # Let the user handle the error case, if they want to log it
            raise NGASException(f"NGAS request failed", response, status)

        self.logging.debug(f"Request success: {url_str}")
        return response, url_str

    def _cache_file_path(self, file_id: str, file_version: Optional[int] = None) -> str:
        """
        Get the file path for a cached file.

        :param file_id: NGAS ID of the file.
        :param file_version: Version of the file.
        :return: Cache path for the file.
        """
        name = file_id
        if file_version is not None:
            name += f"_{file_version}"
        return os.path.join(self.config.cache_dir, name)

    def _find_cache_path(self, file_id: str, file_version: Optional[int] = None):
        """
        Find the cache path for a file, with an optional version/

        If a specific version is not requested, then the highest version is returned.

        :param file_id: NGAS ID of the file.
        :param file_version: Version of the file.
        :return: Cache path for the file, or None if the file doesn't exist in the cache.
        """
        path = self._cache_file_path(file_id, file_version)
        if os.path.exists(path):
            # Exact path exists for this file.
            self.logging.info(f"Found cache file: {path}")
            return path

        if file_version is None:
            # see if there's a versioned path
            paths: list[str] = glob.glob(f"{path}*")
            if len(paths) > 0:
                # the remaining bit of the path should be _{number} only.
                def get_version(path: str):
                    index = path.rfind('_')
                    if index < 0:
                        return -1
                    try:
                        return int(path[index+1:])
                    except:
                        return -1
                real_paths = [(path, get_version(path)) for path in paths if get_version(path) >= 0]
                real_paths.sort(key=lambda x: x[1], reverse=True)
                if len(real_paths) > 0:
                    # return the highest versioned path
                    self.logging.info(f"Found cache file: {real_paths[0][0]}")
                    return real_paths[0][0]

        # Could not find something with the specified file version :(
        self.logging.info(f"Could not find cached file {path}")
        return None

    def _archive(
        self,
        command: NGASCommand,
        file: Union[str, os.PathLike, io.IOBase],
        filename: Optional[str] = None,
        mime_type="application/octet-stream",
        asyncronous=False,
        versioning=True,
        file_version: Optional[int] = None,
        **kwargs
    ) -> tuple[NGASStatusResponse, str]:
        """
        Perform an ARCHIVE or QARCHIVE request.

        :exception RuntimeError: If force_cache is True.
        :param file: File, or path to file to archive.
            If the file is a remote file (http, ftp), it will be downloaded by the NGAS server and stored with
            the filename it has. Local files will be stored with their local file names, or the override filename
            if given
        :param filename: Override filename to use when storing a local file. If provided, this name will
            be used on the NGAS server instead of the file's name.
        :param mime_type: describes the content-type of the file. If not given, NGAS tries to guess it based on the
            filename’s extension, and the internal mime-type information stored in the NGAS configuration.
        :param asyncronous: Whether the registration should be carried out asynchronously or not.
        :param versioning: used to switch the automatic versioning on (True, the default behavior) or off (False).
        :param file_version: specifies which file version to overwrite. Only taken into account when versioning=False
        :param q_archive: Use QARCHIVE instead of ARCHIVE.
        :param kwargs: Additional arguments to pass to the request.
        :return: Tuple containing the Status returned by th server, and the name of the file.
        """
        if self.config.force_cache:
            # Prevent the upload in force_cache mode, since the user is trying to conserve
            # internet bandwidth.
            raise RuntimeError("Cannot use archive with force_cache=True")

        params = {
            "async": '1' if asyncronous else '0',
            "versioning": '1' if versioning else '0',
        }
        if file_version is not None:
            params["file_version"] = str(file_version)

        if isinstance(file, str):
            if is_known_pull_url(file):
                self.logging.debug(f"Using pull url {file}")
                params["filename"] = file
                if mime_type is not None:
                    params["mime_type"] = mime_type
                return self._ngas_request_xml(command, params), os.path.basename(file)

        close_file = False
        if isinstance(file, io.IOBase):
            self.logging.debug(f"Provided with open file handle {file}")
        else:
            self.logging.debug(f"Opening local file {file}")
            # We're opening this file internally and will need to close it
            file = open(file, 'rb')
            close_file = True

        # File is now guaranteed to be an open IOBase
        try:
            fname = filename if filename is not None else os.path.basename(file.name)
            params["filename"] = fname
            response = self._ngas_request_xml(
                command,
                params,
                method="POST",
                data=file,
                headers={'Content-Type': mime_type or "ngas/archive-request"},
                **kwargs
            )
            if self.config.cache_dir is not None:
                # succeeded, so now we want to write the file out to the cache directory
                cache_path = self._cache_file_path(fname, file_version if versioning and file_version is not None else None)
                self.logging.info(f"Caching archived file locally: {cache_path}")
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'wb') as f:
                    file.seek(0)
                    f.write(file.read())
            return response, fname
        finally:
            if close_file:
                self.logging.debug(f"Closing local file {file.name}")
                file.close()
