#!/bin/bash
# Compress a bunch of wav files to tta format
# using ffmpeg

INPUT_DIR=$1
OUTPUT_DIR=$2
FILES=`find ${INPUT_DIR} | grep wav$`

# create sha256 checksums to be able to check decoded files
# sha256sum kk > checksums.txt

mkdir -p ${OUTPUT_DIR}

for FILE in $FILES
do
  OUTPUT_FILE="${OUTPUT_DIR}/$(basename ${FILE} .wav).tta"
  ffmpeg -i ${FILE} -f tta ${OUTPUT_FILE}
done
