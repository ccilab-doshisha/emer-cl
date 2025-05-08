#!/bin/bash

# path setting
if [ -v $WORKDIR ]; then
  printf "\e[1;31m%s\n\e[m" "Please set WORKDIR environment variable" >&2
  exit 1
fi
DATA_PATH=$WORKDIR/src/data

printf "\e[1;33m%s\e[m %s\n" "[NOTE]" "Pre-processing of the each dataset (DEAM/PMEmo) for EMER-CL will be started"

# dataset check (zip will be unzipped)
source $DATA_PATH/shell/dataset_check.sh

# vggish will be cloned.
source $DATA_PATH/shell/vggish.sh

# create dataset
source $DATA_PATH/shell/dataset.sh

printf "\e[1;33m%s\e[m %s\n" "[NOTE]" "Pre-processing of the each dataset (DEAM/PMEmo) for EMER-CL is complete!"
