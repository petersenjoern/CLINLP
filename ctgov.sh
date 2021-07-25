#!/bin/bash

MY_PATH="`dirname \"$0\"`"
DIR_DOWNLOAD="${MY_PATH}/data/ctgov/download"
DIR_PSQL_EXTRACTION="${MY_PATH}/data/ctgov/extraction"

#20210601 or more general: yyyymm01
DATE=$1
CTGOV_FILE_NAME="${DATE}_clinical_trials.zip"

mkdir -p "$DIR_DOWNLOAD"
mkdir -p "$DIR_PSQL_EXTRACTION"

if ! wget -P "$DIR_DOWNLOAD" "https://aact.ctti-clinicaltrials.org/static/static_db_copies/monthly/$CTGOV_FILE_NAME"
then
  rm -f "$DIR_DOWNLOAD/$CTGOV_FILE_NAME"
  echo "Download failed."
  exit 1
fi

unzip -o -d "$DIR_DOWNLOAD" "$DIR_DOWNLOAD/$CTGOV_FILE_NAME"
set -e

echo "Starting docker-compose postgres db and extract data with scripts/ctgov/01-init.sh"
docker-compose --env-file ".env" up