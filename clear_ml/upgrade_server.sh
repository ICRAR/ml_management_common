#!/bin/bash

CLEARML_PATH="/opt/clearml/"

echo "Upgrading server in ${CLEARML_PATH}"
cd "$CLEARML_PATH" || exit
docker-compose -f docker-compose.yml down
docker-compose -f "${CLEARML_PATH}docker-compose.yml" pull
echo "ClearML server updated in ${CLEARML_PATH}"
echo "Start the server with: docker-compose -f ${CLEARML_PATH}docker-compose.yml up -d"