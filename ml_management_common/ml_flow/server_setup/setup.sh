sudo apt install python3.9 python3.9-dev curl mysql-server python3.9-venv nfs-kernel-server libmysqlclient-dev openssh-server
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.9 get-pip.py
sudo mysql_secure_installation
mkdir mlflow
cd mlflow
python3.9 -m venv venv
source venv/bin/activate
pip install mlflow mysqlclient

./db_setup.sh
./nfs_setup.sh

sudo cp ./mlflow.server.service /etc/systemd/system
sudo systemctl enable mlflow.server
sudo systemctl start mlflow.server
