#!/bin/bash

mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root s3://honey.mlflow.test/artifacts \
    --host 0.0.0.0 \
    --port 25565