#!/usr/bin/env bash

aws cloudformation create-stack \
--stack-name=nba-mlflow-stack-ecs-cluster-v1 \
--template-body file://mlflow-ecs-cluster.yml \
--parameters ParameterKey=ClusterName,ParameterValue="NBA-ML-FLOW-CLUSTER"