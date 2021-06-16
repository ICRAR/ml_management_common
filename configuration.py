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

class MLProjectConfiguration(object):
    def __init__(self, project_name: str, output_uri: str):
        """
        Create an ML Project configuration

        :param str project_name: The name of the project in which the experiment will be created. If the project does
            not exist, it is created.
        :param str output_uri: The default location for output models and other artifacts.
            If True is passed, the default files_server will be used for model storage.
            In the default location, ClearML creates a subfolder for the output.
            The subfolder structure is the following:
            <output destination name> / <project name> / <task name>.<Task ID>

            The following are examples of ``output_uri`` values for the supported locations:

            - A shared folder: ``/mnt/share/folder``
            - S3: ``s3://bucket/folder``
            - Google Cloud Storage: ``gs://bucket-name/folder``
            - Azure Storage: ``azure://company.blob.core.windows.net/folder/``
            - Default file server: True

            .. important::

               For cloud storage, you must install the **ClearML** package for your cloud storage type,
               and then configure your storage credentials. For detailed information, see
               `ClearML Python Client Extras <./references/clearml_extras_storage/>`_ in the "ClearML Python Client
               Reference" section.
        """
        self.project_name = project_name
        self.output_uri = output_uri

