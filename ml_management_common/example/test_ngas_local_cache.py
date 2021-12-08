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
        "test_artifact_upload_caching",
        TaskTypes.testing,
        MLProjectConfiguration(
            project_name="test_ngas",
            tracking_server="http://130.95.218.14:5000",
            output_uri="/mnt/mlflow_artifact_nfs/artifacts",
            ngas_client=NGASConfiguration(
                host="130.95.218.14",
                cache_dir="/home/sam/ngas_test_cache",
                logging=True,
                force_cache=True
            )
        ),
        {}
    ) as experiment:
        #experiment.ngas().log_artifact("/home/sam/get-pip.py", "get_pip_1", ngas_name="get-pip-test-2")
        #experiment.ngas().log_artifact("/home/sam/get-pip.py", "get_pip_2")

        # Actually wait for everything to be uploaded before trying to download...
        #experiment.wait_upload_jobs()

        experiment.ngas().download_file("/mnt/mlflow_artifact_nfs/1/95a5c7e56b134e9e92abcd7ee1d2aac0/artifacts/get_pip_2/get-pip.py", "/home/sam/test-pip-download-2")


def main2():
    with MLFlowExperiment(
            "test_artifact_download_caching",
            TaskTypes.testing,
            MLProjectConfiguration(
                project_name="test_ngas",
                tracking_server="http://130.95.218.14:5000",
                output_uri="/mnt/mlflow_artifact_nfs/artifacts",
                ngas_client=NGASConfiguration(
                    host="130.95.218.14",
                    cache_dir="/home/sam/ngas_test_cache",
                )
            ),
            {}
    ) as experiment:
        experiment.ngas().download_file()


if __name__ == '__main__':
    main()