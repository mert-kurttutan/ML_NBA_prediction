#!/usr/bin/env bash

# Credentials
aws configure set aws_access_key_id ${AWS_ACCESS_KEY_ID} --profile default
aws configure set aws_secret_access_key ${AWS_SECRET_ACCESS_KEY} --profile default

# Config
aws configure set region us-east-1 --profile default
aws configure set output json --profile default
aws configure set cli_auto_prompt off --profile default
