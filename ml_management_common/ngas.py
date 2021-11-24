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
import requests

from io import StringIO
from datetime import datetime
from enum import Enum
from urllib.parse import urlparse, urlunparse
from defusedxml.ElementTree import parse as xmlparse
from xml.etree.ElementTree import ElementTree, Element, tostring
from typing import TypeVar, Type, Iterator, Optional, Union

from requests import Response

T = TypeVar('T')


class NGASConfiguration(object):
    def __init__(self, host="localhost", port=7777, protocol="http"):
        self.host = host
        self.port = port
        self.protocol = protocol


class NGASResponse(object):
    def __init__(self, url: str, status: int):
        self.http_url = url
        self.http_status = status

    @property
    def http_ok(self):
        return 200 <= self.http_status < 300


class NGASStatusResponse(NGASResponse):
    def __init__(self, url: str, status: int, body: ElementTree):
        super().__init__(url, status)

        root = body.getroot().find("Status")
        self.date = datetime.fromisoformat(root.attrib["Date"])
        self.version = root.attrib["Version"]
        self.host_id = root.attrib["HostId"]
        self.message = root.attrib["Message"]
        self.status = root.attrib["Status"]
        self.state = root.attrib["State"]
        self.sub_state = root.attrib["SubState"]

    def __repr__(self):
        return f"NGASStatusResponse(url='{self.http_url}', status='{self.http_status}', date={self.date}, version={self.version}, host_id='{self.host_id}', message='{self.message}', status='{self.status}', state='{self.state}', sub_state='{self.sub_state}')"


class NGASFileResponse(NGASResponse):
    def __init__(self, url: str, status: int, filename: str, data_iter: Iterator):
        super().__init__(url, status)
        self.filename = filename
        self.data_iter = data_iter

    def __repr__(self):
        return f"NGASFileResponse(url='{self.http_url}', status='{self.http_status}', filename='{self.filename}')"

    def write(self, filename: Optional[str] = None):
        final_filename = self.filename if filename is None else filename
        with open(final_filename, "wb") as f:
            for chunk in self.data_iter:
                f.write(chunk)
        return final_filename


class NGASException(Exception):
    def __init__(self, message: str, response: Response, status: Optional[NGASStatusResponse] = None):
        super().__init__(message)
        self.response = response
        self.status = status


class NGASCommand(str, Enum):
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


def parse_http_kv_list(header: str) -> dict[str, str]:
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
    root = Element("FileList")
    for file in file_list:
        root.append(Element('File', {'FileId': file}))
    return tostring(root, encoding="utf-8")


def is_known_pull_url(s: str):
    return s.startswith('file:') or \
           s.startswith('http:') or \
           s.startswith('https:') or \
           s.startswith('ftp:')


class NGASClient(object):
    def __init__(self, config: NGASConfiguration, timeout=10):
        self.config = config
        self.timeout = timeout

    def exit(self):
        return self._ngas_request_xml(NGASCommand.EXIT)

    def init(self):
        return self._ngas_request_xml(NGASCommand.INIT)

    def label(self, slot_id: str, host_id: str):
        return self._ngas_request_xml(NGASCommand.LABEL, {
            "slot_id": slot_id,
            "host_id": host_id
        })

    def offline(self):
        return self._ngas_request_xml(NGASCommand.OFFLINE)

    def online(self):
        return self._ngas_request_xml(NGASCommand.ONLINE)

    def remdisk(self, disk_id: str, execute=False):
        return self._ngas_request_xml(NGASCommand.REMDISK, {
            "disk_id": disk_id,
            "execute": "1" if execute else "0"
        })

    def remfile(self, disk_id: str, file_id: str, file_version: Optional[int] = None, execute=False):
        params = {
            "disk_id": disk_id,
            "file_id": file_id,
            "execute": "1" if execute else "0"
        }
        if file_version is not None:
            params["file_version"] = str(file_version)
        return self._ngas_request_xml(NGASCommand.REMFILE, params)

    def register(self, path: str, asyncronous=False):
        return self._ngas_request_xml(NGASCommand.REGISTER, {
            "path": path,
            "async": '1' if asyncronous else '0'
        })

    def archive(
        self,
        file: Union[str, os.PathLike, io.IOBase],
        filename: Optional[str] = None,
        mime_type="application/octet-stream",
        asyncronous=False,
        versioning=True,
        file_version: Optional[int] = None,
        q_archive=False,
        **kwargs
    ):
        """

        :param file: File, or path to file to archive.
        If the file is a remote file (http, ftp), it will be downloaded by the NGAS server and stored with
        the filename it has.
        Local files will be stored with their local file names, or the override filename if given
        :param filename: Override filename to use when storing a local file. If provided, this name will
        be used on the NGAS server instead of the file's name.
        :param mime_type:
        :param asyncronous:
        :param versioning:
        :param file_version:
        :param q_archive:
        :param kwargs:
        :return:
        """

        params = {
            "async": '1' if asyncronous else '0',
            "versioning": '1' if versioning else '0',
        }
        if file_version is not None:
            params["file_version"] = str(file_version)
        command = NGASCommand.QARCHIVE if q_archive else NGASCommand.ARCHIVE

        if isinstance(file, str):
            if is_known_pull_url(file):
                params["filename"] = file
                if mime_type is not None:
                    params["mime_type"] = mime_type
                return self._ngas_request_xml(command, params), os.path.basename(file)

        close_file = False
        if not isinstance(file, io.IOBase):
            # We're opening this file internally and will need to close it
            file = open(file, 'rb')
            close_file = True

        # File is now guaranteed to be an open IOBase
        try:
            fname = filename if filename is not None else os.path.basename(file.name)
            params["filename"] = fname
            return self._ngas_request_xml(
                command,
                params,
                method="POST",
                data=file,
                headers={'Content-Type': mime_type or "ngas/archive-request"},
                **kwargs
            ), fname
        finally:
            if close_file:
                file.close()

    def rearchive(self, file_uri: str, file_info_xml: Element, **kwargs):
        raise NotImplementedError("Not implemented for now")

    def clone(
        self,
        file_id: Optional[str] = None,
        disk_id: Optional[str] = None,
        file_version: Optional[int] = None,
        target_disk_id: Optional[str] = None,
        asyncronous=False
    ):
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

    def carchive(self, directory: str, files_mtype: str, qarchive=False):
        raise NotImplementedError("Not implemented for now as multipart container uploads are a bit complex")

    def cappend(
        self,
        file_id: Optional[str] = None,
        file_id_list: Optional[list[str]] = None,
        container_id: Optional[str] = None,
        container_name: Optional[str] = None,
        force=False,
        close_container=False
    ):
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

    def ccreate(
        self,
        container_name: Optional[str] = None,
        parent_container_id: Optional[str] = None,
        container_hierarchy: Optional[Element] = None
    ):
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

    def cdestroy(self, container_name: Optional[str] = None, container_id: Optional[str] = None, recursive=False):
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

    def clist(self, container_name: Optional[str] = None, container_id: Optional[str] = None):
        if not (bool(container_name) ^ bool(container_id)):
            raise ValueError("Either container_name or container_id must be specified")

        params = {}
        if container_name is not None:
            params["container_name"] = container_name
        if container_id is not None:
            params["container_id"] = container_id

        return self._ngas_request_xml(NGASCommand.CLIST, params)

    def cremove(
        self,
        file_id: Optional[str] = None,
        file_id_list: Optional[list[str]] = None,
        container_id: Optional[str] = None,
        container_name: Optional[str] = None
    ):
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

    def cretrieve(self, container_name: Optional[str], container_id: Optional[str] = None):
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

    def retrieve(
        self,
        file_id: str,
        file_version: Optional[int] = None,
        processing: Optional[str] = None,
        processing_pars: Optional[str] = None,
    ):
        params = {
            "file_id": file_id,
        }
        if file_version is not None:
            params["file_version"] = str(file_version)
        if processing is not None:
            params["processing"] = processing
        if processing_pars is not None:
            params["processing_pars"] = processing_pars
        return self._ngas_request_file(NGASCommand.RETRIEVE, params)

    def subscribe(
        self,
        url: str,
        priority: Optional[int] = None,
        start_date: Optional[str] = None,
        filter_plugin: Optional[str] = None,
        filter_plugin_pars: Optional[str] = None
    ):
        params = {"url": url}
        if priority is not None:
            params["priority"] = priority
        if start_date is not None:
            params["start_date"] = start_date
        if filter_plugin is not None:
            params["filter_plugin"] = filter_plugin
        if filter_plugin_pars is not None:
            params["filter_plugin_pars"] = filter_plugin_pars
        return self._ngas_request_xml(NGASCommand.SUBSCRIBE, params)

    def unsubscribe(self, url: str):
        return self._ngas_request_xml(NGASCommand.UNSUBSCRIBE, {
            "url": url
        })

    def status(self) -> NGASStatusResponse:
        """
        Get the status of the NGAS server
        :return:
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
        response, url_str = self._ngas_request(command, params, headers, method, stream=True, **kwargs)
        disposition = parse_http_kv_list(response.headers.get("Content-Disposition", ""))
        iterator = response.iter_content(chunk_size=response_file_chunk_size)
        return NGASFileResponse(url_str, response.status_code, disposition.get("filename", ""), iterator)

    def _ngas_request_xml(
        self,
        command: NGASCommand,
        params: Optional[dict[str, str]] = None,
        headers: Optional[dict[str, str]] = None,
        method="GET",
        **kwargs
    ):
        response, url_str = self._ngas_request(command, params, headers, method, **kwargs)
        return NGASStatusResponse(url_str, response.status_code, xmlparse(StringIO(response.text)))

    def _ngas_request(
        self,
        command: NGASCommand,
        params: Optional[dict[str, str]] = None,
        headers: Optional[dict[str, str]] = None,
        method="GET",
        **kwargs
    ):
        url = urlparse(f"{self.config.protocol}://{self.config.host}:{self.config.port}/{command}")

        if params is not None:
            parts = [f'{k}={v}' for k, v in params.items()]
            url = url._replace(query='&'.join(parts))

        url_str: str = urlunparse(url)
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
                status = NGASStatusResponse(url_str, response.status_code, xmlparse(StringIO(response.text)))
            except Exception as e:
                status = None
            raise NGASException(f"NGAS request failed", response, status)

        return response, url_str


def main():
    client = NGASClient(NGASConfiguration(host="130.95.218.14"))
    try:
        print(client.status())
        #print(client.archive("/home/sam/get-pip.py", mime_type="application/octet-stream"))
        stream = client.retrieve("get-pip.py")
        print(stream)
        stream.write("/home/sam/get-pip-response.py")
    except NGASException as e:
        print(e.status)


if __name__ == '__main__':
    main()
