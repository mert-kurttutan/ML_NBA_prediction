#!/bin/sh

set -e
yum update -y

# install basic python environment
yum install -y python37 gcc gcc-g++ python3-devel

# needed for downloading files
yum install -y wget

# JDBC and PyODBC dependencies
yum install -y java-1.8.0-openjdk unixODBC-devel 

# Archiving Libraries
yum install -y zip unzip bzip2 gzip


#### Required Libraries for entrypoint.sh script

# jq is used to parse ECS-injected AWSSecretsManager secrets
yum install -y jq

# nc is used to check DB connectivity
yum install -y nc

# Install additional system library dependencies. Provided as a string of libraries separated by space
if [ -n "${SYSTEM_DEPS}" ]; then yum install -y "${SYSTEM_DEPS}"; fi

yum clean all
