#!/bin/bash

CLEARML_PATH="/opt/clearml/"

echo "Installing server to ${CLEARML_PATH}"
sudo mkdir -p "$CLEARML_PATH"
sudo chown -R 1000:1000 "$CLEARML_PATH"
sudo mkdir -p "${CLEARML_PATH}data/elastic_7"
sudo mkdir -p "${CLEARML_PATH}data/mongo/db"
sudo mkdir -p "${CLEARML_PATH}data/mongo/configdb"
sudo mkdir -p "${CLEARML_PATH}data/redis"
sudo mkdir -p "${CLEARML_PATH}logs"
sudo mkdir -p "${CLEARML_PATH}config"
sudo mkdir -p "${CLEARML_PATH}data/fileserver"
cd "$CLEARML_PATH" || exit
sudo curl https://raw.githubusercontent.com/allegroai/trains-server/master/docker/docker-compose.yml -o docker-compose.yml
echo "ClearML server installed to ${CLEARML_PATH}"
echo "Start the server with: docker-compose -f ${CLEARML_PATH}docker-compose.yml up -d"