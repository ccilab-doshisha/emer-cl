#!/bin/bash

# [NEED DATASET DOWNLOAD]
# PLEASE CHECK README IN THIS DIR

datasets=(
  DEAM
  PMEmo
)

# path setting
if [ -v $WORKDIR ]; then
  printf "\e[1;31m%s\n\e[m" "Please set WORKDIR environment variable" >&2
  exit 1
fi
DATA_PATH=$WORKDIR/src/data

# Validation
# zip unzipped_dir save_dir
dataArray=(
  "$DATA_PATH/PMEmo/dataset/PMEmo2019.zip $DATA_PATH/PMEmo/dataset/PMEmo2019 $DATA_PATH/PMEmo/dataset"
  "$DATA_PATH/DEAM/dataset/DEAM_audio.zip $DATA_PATH/DEAM/dataset/MEMD_audio $DATA_PATH/DEAM/dataset"
  "$DATA_PATH/DEAM/dataset/DEAM_Annotations.zip $DATA_PATH/DEAM/dataset/annotations $DATA_PATH/DEAM/dataset"
  "$DATA_PATH/DEAM/dataset/metadata.zip $DATA_PATH/DEAM/dataset/metadata $DATA_PATH/DEAM/dataset"
)

UNZIPPED=false
ERROR=false
printf "\e[1;33m%s\e[m %s\n" "[NOTE]" "check zipfile of DEAM and PMEmo"
for i in "${dataArray[@]}"; do
  pathes=(${i[@]})
  if [ -d ${pathes[1]} ]; then
    printf "\e[1;36m%s\e[m %s\n" "[OK]" ${pathes[1]}
  elif [ -e ${pathes[0]} ]; then
    printf "\e[1;36m%s\e[m %s\n" "[OK]" ${pathes[0]}
    printf "\e[1;33m%s\e[m %s\n" "[NOTE]" "${pathes[0]} is unzipping now"
    unzip ${pathes[0]} -d ${pathes[2]} > /dev/null
    printf "\e[1;33m%s\e[m %s\n" "[NOTE]" "${pathes[0]} was unzipped"
    UNZIPPED=true
  else
    printf "\e[1;31m%s\e[m %s\n" "[NG]" $path.zip
    ERROR=true
  fi
done

if $ERROR ; then
  printf "\e[1;31m%s\n\e[m" "Zip file of dataset not found" >&2
  exit 1
elif $UNZIPPED ; then
  printf "\e[1;33m%s\e[m %s\n" "[NOTE]" "upzip done!"
fi