#!/bin/bash
source /home/mlflow/mlflow/venv/bin/activate
mlflow server \
    --backend-store-uri mysql://mlflow:clearml@localhost/mlflow \
    --default-artifact-root /mnt/mlflow_artifact_nfs \
    --host 130.95.218.14
