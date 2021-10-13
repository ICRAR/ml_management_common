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
import argparse
from functools import reduce

import time
from ml_management_common import TaskTypes, MLProjectConfiguration
from ml_management_common.ml_flow import MLFlowExperiment


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("large_file", type=str)
    return vars(args.parse_args())


def main():
    args = parse_args()

    large_file = args.get("large_file")
    if large_file is None:
        print("Invalid large file")
        return
    print(f"Using file {large_file}")

    # Try uploading this file and see how long it takes

    expr_times = []
    times = []
    for i in range(1):
        start_expr = time.time()
        print(f"Test Experiment {i} @ {start_expr}")
        with MLFlowExperiment(
                "test_artifact_upload",
                TaskTypes.testing,
                MLProjectConfiguration(
                    project_name="test_artifact_upload",
                    tracking_server="http://130.95.218.59:5000",
                    output_uri="/mnt/mlflow_artifact_nfs/artifacts"
                ),
                {}
        ) as experiment:
            for a in range(1):
                start = time.time()
                print(f"Starting upload {a} @ {start}")
                experiment.log_artifact(large_file)
                end = time.time()
                print(f"Ending upload {a} @ {end}: Took {end - start}")
                times.append(end - start)
        end_expr = time.time()
        print(f"End test {i} @ {end_expr}: Took {end_expr - start_expr}")
        expr_times.append(end_expr - start_expr)
    print(f"Ave expr: {reduce(lambda x, y: x + y, expr_times) / len(expr_times)}")
    print(f"Ave: {reduce(lambda x, y: x + y, times) / len(times)}")


if __name__ == "__main__":
    main()
