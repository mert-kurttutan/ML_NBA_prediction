#!/usr/bin/env bash
###############################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# This file is licensed under the Apache License, Version 2.0 (the "License").
#
# You may not use this file except in compliance with the License. A copy of
# the License is located at http://aws.amazon.com/apache2.0/.
#
# This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
###############################################################################
#// snippet-start:[s3.bash.bucket-operations.complete]
source ./awsdocs_general.sh
source ./bucket_operations.sh

# Settings for bucket
BUCKET_NAME="mert-kurttutan-nba-project-transformed-data-v1"
region="us-east-1"


# If the bucket already exists, we don't want to try to create it.
if (bucket_exists $BUCKET_NAME); then 
  errecho "ERROR: A bucket with that name already exists. Try again."
  #return 1
fi

create_bucket -b $BUCKET_NAME -r $region