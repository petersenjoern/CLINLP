#!/bin/bash

MY_PATH="`dirname \"$0\"`"
DIR="${MY_PATH}/data/ctgov" 
DB_NAME=ctgov

#20210601 or more general: yyyymm01
DATE=$1
CTGOV_FILE_NAME="${DATE}_clinical_trials.zip"

mkdir -p "$DIR"

# if ! wget -P "$DIR" "https://aact.ctti-clinicaltrials.org/static/static_db_copies/monthly/$CTGOV_FILE_NAME"
# then
#   rm -f "$DIR/$CTGOV_FILE_NAME"
#   echo "Download failed."
#   exit 1
# fi

# unzip -o -d "$DIR" "$DIR/$CTGOV_FILE_NAME"
set -e

echo "Starting docker-compose postgres db"
docker-compose --env-file ".env" up