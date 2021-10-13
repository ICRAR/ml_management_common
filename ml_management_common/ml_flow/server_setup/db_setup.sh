#!/bin/bash

MLFLOW_DATABASE_NAME="mlflow"
MLFLOW_USERNAME="mlflow"
MLFLOW_PASSWORD="clearml"
sudo mysql -e "
	CREATE DATABASE ${MLFLOW_DATABASE_NAME};
	CREATE USER '${MLFLOW_USERNAME}'@'localhost' IDENTIFIED BY '${MLFLOW_PASSWORD}';  
	GRANT ALL PRIVILEGES ON ${MLFLOW_DATABASE_NAME}.* TO '${MLFLOW_USERNAME}'@'localhost' WITH GRANT OPTION;
"
