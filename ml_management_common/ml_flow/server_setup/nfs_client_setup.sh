#!/bin/bash
sudo apt install nfs-common
sudo mkdir /mnt/mlflow_artfiact_nfs
echo "130.95.218.59:/home/mlflow/mlflow/artifact_nfs /mnt/mlflow_artfiact_nfs nfs rsize=8192,wsize=8192,timeo=14,intr" | sudo tee -a /etc/fstab
