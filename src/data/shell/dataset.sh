#!/bin/bash
# path setting
if [ -v $WORKDIR ]; then
  printf "\e[1;31m%s\n\e[m" "Please set WORKDIR environment variable" >&2
  exit 1
fi
DATA_PATH=$WORKDIR/src/data
datasets=(
  DEAM
  PMEmo
)
# create feature
for dataset in ${datasets[@]}; do
    printf "\e[1;32m%s\e[m %s: %s\n" "[LOG]" "$dataset" "convert mp3 to wav"
    python $DATA_PATH/$dataset/mp3toWav.py
    printf "\e[1;32m%s\e[m %s: %s\n" "[LOG]" "$dataset" "extract VGGish sequences"
    python $DATA_PATH/$dataset/create_vggish.py
    printf "\e[1;32m%s\e[m %s: %s\n" "[LOG]" "$dataset" "organise va sequences"
    python $DATA_PATH/$dataset/create_va.py
    printf "\e[1;32m%s\e[m %s: %s\n" "[LOG]" "$dataset" "re-organise metadata"
    python $DATA_PATH/$dataset/create_metadata.py
done