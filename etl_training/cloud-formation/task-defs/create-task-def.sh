#!/usr/bin/env bash

aws cloudformation create-stack \
--stack-name=nba-mlflow-dag-stack-v1 \
--template-body file://mlflow_dag_task_def.yml \
--parameters 
