
# Settings for bucket
BUCKET_NAME="mert-kurttutan-nba-ml-project-raw-data"
RAW_DATA_DIR="./raw_data"

aws s3 cp s3://$BUCKET_NAME/ $RAW_DATA_DIR --recursive



# Settings for bucket
#BUCKET_NAME="mert-kurttutan-nba-project-transformed-data-v1"

#aws s3 cp s3://$BUCKET_NAME/ $TEMP_DATA_DIR --recursive