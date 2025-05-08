#!/bin/bash
# path setting
if [ -v $WORKDIR ]; then
  printf "\e[1;31m%s\n\e[m" "Please set WORKDIR environment variable" >&2
  exit 1
fi
DATA_PATH=$WORKDIR/src/data

# vggish will be cloned.
if [ -d "$DATA_PATH/vggish" ]; then 
  printf "\e[1;36m%s\e[m %s\n" "[OK]" "VGGish was cloned! ($DATA_PATH/vggish)"
else
  printf "\e[1;33m%s\e[m %s\n" "[NOTE]" "VGGish will be installed"
  # data setting
  git clone https://github.com/tensorflow/models.git $DATA_PATH/models
  cp -r $DATA_PATH/models/research/audioset/vggish $DATA_PATH/vggish
  rm -rf $DATA_PATH/models/
  curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
  curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz

  # vggish check
  printf "\e[1;32m%s\e[m %s\n" "[LOG]" "Make sure VGGish works properly"
  python $DATA_PATH/vggish/vggish_smoke_test.py
  # copy into vggish dir
  mv $DATA_PATH/vggish_model.ckpt $DATA_PATH/vggish_pca_params.npz $DATA_PATH/vggish/
  printf "\e[1;33m%s\e[m %s\n" "[NOTE]" "VGGish is installed!"
fi