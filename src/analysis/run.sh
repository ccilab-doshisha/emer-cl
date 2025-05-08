#! /bin/bash
export TF_CPP_MIN_LOG_LEVEL=2

# check setting
if [ -v $WORKDIR ]; then
  printf "\e[1;31m%s\n\e[m" "Please set WORKDIR environment variable" >&2
  exit 1
elif [ -v $DEAM_DATETIME ]; then
  printf "\e[1;31m%s\n\e[m" "Please set DEAM_DATETIME environment variable" >&2
  exit 1
elif [ -v $PMEmo_DATETIME ]; then
  printf "\e[1;31m%s\n\e[m" "Please set PMEmo_DATETIME environment variable" >&2
  exit 1
fi

ANALTSIS_PATH=$WORKDIR/src/analysis
dataArray=(
    "DEAM ${DEAM_DATETIME}"
    "PMEmo ${PMEmo_DATETIME}"
)

for i in "${dataArray[@]}"; do
    data=(${i[@]})
    dataset=${data[0]}
    datetime=${data[1]}

    printf "\e[1;33m%s\e[m %s\n" "[NOTE]" "Classify $dataset result based on the ranks"
    python $ANALTSIS_PATH/create_eval.py --dataset=$dataset --datetime=$datetime

    printf "\e[1;33m%s\e[m %s\n" "[NOTE]" "Perform a detailed analysis for $dataset"
    python $ANALTSIS_PATH/detailed_analysis.py --dataset=$dataset

    printf "\e[1;33m%s\e[m %s\n" "[NOTE]" "Create a figure of the result of the detailed analysis for $dataset"
    python $ANALTSIS_PATH/create_figures.py --dataset=$dataset
done
