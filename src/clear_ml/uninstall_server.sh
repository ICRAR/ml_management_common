#!/bin/bash

CLEARML_PATH="/opt/clearml/"

echo "Uninstalling server in ${CLEARML_PATH}"
cd "$CLEARML_PATH" || exit
docker-compose -f docker-compose.yml down
sudo rm -rf "$CLEARML_PATH"
echo "Server removed from ${CLEARML_PATH}"