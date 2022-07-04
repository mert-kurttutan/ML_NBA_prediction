#!/bin/sh

set -e

# install adduser and add the airflow user
adduser -s /bin/bash -d "${MLFLOW_USER_HOME}" mlflow

mkdir -p ${MLFLOW_USER_HOME}/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ${MLFLOW_USER_HOME}/miniconda3/miniconda.sh
bash ${MLFLOW_USER_HOME}/miniconda3/miniconda.sh -b -u -p ${MLFLOW_USER_HOME}/miniconda3
rm -rf ${MLFLOW_USER_HOME}/miniconda3/miniconda.sh

# Activate conda
${MLFLOW_USER_HOME}/miniconda3/bin/conda init bash
${MLFLOW_USER_HOME}/miniconda3/bin/conda init zsh
source ~/.bashrc


# Upgrade pip version to latest
#python3 -m 
pip3 install --upgrade pip



# Install wheel to avoid legacy setup.py install
pip3 install wheel

pip3 install -r /requirements.txt


# install additional python dependencies
if [ -n "${PYTHON_DEPS}" ]; then pip3 install $PIP_OPTION "${PYTHON_DEPS}"; fi



# install watchtower for Cloudwatch logging
pip3 install $PIP_OPTION watchtower==1.0.1



# Location for custom python modules
mkdir -p ${MLFLOW_USER_HOME}/custom_src


# install awscli v2, according to https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-linux.html#cliv2-linux-install
zip_file="awscliv2.zip"
cd /tmp
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o $zip_file
unzip $zip_file
./aws/install
rm $zip_file
rm -rf ./aws
cd -  # Return to previous directory



# snapshot the packages
if [ -n "$INDEX_URL" ]
then
  pip3 freeze > /requirements.txt
else
  # flask-swagger depends on PyYAML that are known to be vulnerable
  # even though Airflow 1.10 names flask-swagger as a dependency, it doesn't seem to use it.
  if [ "$AIRFLOW_VERSION" = "1.10.12" ]
  then
    pip3 uninstall -y flask-swagger
  fi
fi
