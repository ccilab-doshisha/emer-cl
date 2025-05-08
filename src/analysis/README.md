# Detailed analysis

## What
Using the trained model obtained by running `train.py`, you can examine the results of the EMER-CL recognition in more detail.

Specifically, we can perform the following analysis, including the results of the detailed analysis reported in the paper.

* M2E (identifying emotions from music)
    - DEAM/PMEmo common
        - Cosine similarity: calculation of the cosine similarity between music query and music corresponding to the emotion embedding.
* E2M (identifying music from emotion)
    - Common to DEAM/PMEmo
        - Cosine similarity: Calculates the cosine similarity between emotion query and emotion corresponding to the music embedding.
    - Only for DEAM
        - average precision : analyses how high the category of the discretised emotional query is in the top K%.
        - mAP: mean of average precision.
        - entropy: analyses how many discrete emotions are mixed together, corresponding to the top 5% of detected music.

## Usage

**It is assumed that `train.py` has been executed in advance**.

## Procedure
The Python files that need to be executed are listed in a shell file. 
The easiest way to do this is to run it.
```sh
# datetime format: YYYY_MMDD_HH_mm_ss (Please check model dir) 
# 1. 
DEAM_DATETIME=DEAM_saved_datetime PMEmo_DATETIME=PMEmo_saved_datetime $WORKDIR/src/analysis/run.sh
# 2.
export DEAM_DATETIME=DEAM_saved_datetime PMEmo_DATETIME=PMEmo_saved_datetime
$WORKDIR/src/analysis/run.sh
# 3. (Running manually)
[1] python $WORKDIR/src/analysis/create_eval.py --dataset=[DEAM/PMEmo] --datetime=YYYY_MMDD_HH_mm_ss
[2] python $WORKDIR/src/analysis/detailed_analysis.py --dataset=[DEAM/PMEmo]
[3] python $WORKDIR/src/analysis/create_figures.py --dataset=[DEAM/PMEmo]
```
