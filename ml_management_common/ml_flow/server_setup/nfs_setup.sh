#!/bin/bash
sudo systemctl start nfs-kernel-server.service
mkdir ~/mlflow/artifact_nfs
echo "/home/mlflow/mlflow/artifact_nfs *(rw,sync,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports 
sudo ln -s /home/mlflow/mlflow/artifact_nfs /mnt/mlflow_artfiact_nfs
