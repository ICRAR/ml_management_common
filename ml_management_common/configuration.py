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
from typing import Optional, Union

from ml_management_common.ngas import NGASClient, NGASConfiguration


class MLProjectConfiguration(object):
    def __init__(
        self,
        project_name: str,
        tracking_server: Optional[str] = None,
        output_uri: Optional[str] = None,
        ngas_client: Union[NGASClient, NGASConfiguration, None] = None,
    ):
        """
        Create an ML Project configuration

        :param str project_name: The name of the project in which the experiment will be created. If the project does
            not exist, it is created.
        :param str tracking_server: URL to the tracking server. Required for MLFlow, but not requried for ClearML.
            ClearML uses `clearml.conf` in your user directory instead.
        :param str output_uri: The location for output models and other artifacts. Not required for MLFlow, but is
            required for ClearML.
        :param ngas_client: The NGAS client to use. If not provided here, then a client can be provided to Experiment.ngas()
        """
        self.project_name = project_name
        self.tracking_server = tracking_server
        self.output_uri = output_uri
        self.ngas_client = ngas_client

