#!/usr/bin/env bash

aws cloudformation create-stack \
--stack-name=nba-mlflow-stack-dynamodb-v1 \
--template-body file://nba-app-dynamodb.yml \
--parameters 