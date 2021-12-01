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
from ml_management_common import TaskTypes, MLProjectConfiguration
from ml_management_common.ml_flow import MLFlowExperiment

from ml_management_common.ngas import NGASConfiguration


def main():
    with MLFlowExperiment(
            "test_artifact_upload",
            TaskTypes.testing,
            MLProjectConfiguration(
                project_name="test_ngas",
                tracking_server="http://130.95.218.14:5000",
                output_uri="/mnt/mlflow_artifact_nfs/artifacts",
                ngas_client=NGASConfiguration(host="130.95.218.14")
            ),
            {}
    ) as experiment:
        #experiment.ngas().log_artifact("/home/sam/get-pip.py", "get_pip_xxx")
        experiment.ngas().log_artifact("/home/sam/get-pip.py", "get_pip_xxx", ngas_name="get-pip")


def main2():
    with MLFlowExperiment(
            "test_artifact_download",
            TaskTypes.testing,
            MLProjectConfiguration(
                project_name="test_ngas",
                tracking_server="http://130.95.218.14:5000",
                output_uri="/mnt/mlflow_artifact_nfs/artifacts",
                ngas_client=NGASConfiguration(host="130.95.218.14")
            ),
            {}
    ) as experiment:
        #experiment.ngas().download_file(
        #    "/mnt/mlflow_artifact_nfs/1/6e4485edfd364446aa05731a0a25d309/artifacts/get_pip_xxx/get-pip.py",
        #    "/home/sam/downloaded-get-pip.py"
        #)
        experiment.ngas().download_named_file("get-pip", "/home/sam/downloaded-get-pip.py")


if __name__ == '__main__':
    #main()
    main2()
