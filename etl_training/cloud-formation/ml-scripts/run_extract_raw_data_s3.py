#!/usr/bin/env python3
import os
from pathlib import Path


# Settings for bucket
BUCKET_NAME = "mert-kurttutan-nba-ml-project-raw-data"
ROOT_DATA_DIR = "./data"
RAW_DATA_DIR = "raw_data"


# Make sure parent directories exists
Path(f'{ROOT_DATA_DIR}/').mkdir(parents=True, exist_ok=True)
os.system(f"aws s3 cp s3://{BUCKET_NAME}/ {ROOT_DATA_DIR}/{RAW_DATA_DIR} --recursive >> ~/entrypoint_log.log")