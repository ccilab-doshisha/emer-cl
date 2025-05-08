# Evaluation of EMER-CL features for Music Emotion Recognition (MER)

This folder contains the code required to evaluate the features learnt by EMER-CL for MER. 

## Note

The data from the PMEmo and DEAM datasets need to be prepared before executing this code. Please refer to `$WORKDIR/src/data/README.md` for details.

## Execution instructions

To obtain the results of the EMER-CL evaluation for MER, four steps need to be carried out:

1. Preparation of the 10 folds for cross-validation on the target dataset -> `create_10_fold_cv.py`
2. Training of the EMER-CL encoders on each fold -> `train_10_fold_cv.py`
3. Extraction of the music encoder features on each fold -> `compute_music_encoder_features.py`
4. Training and evaluation of a soft-margin regression SVM (C-SVR) to predict arousal and valence levels -> `train_regression_model.py`

All scripts for steps #1, #3 and #4 can be executed after modifying the hyper-parameters defined at the beginning of the `main()` function according to preferences.

The training script for step #2 can be executed by specifying its arguments on command line (`python train_10_fold_cv.py --help` or refer to `$WORKDIR/src/data/README.md` to get more information about them). The same parameters as for the main study were used, i.e.:
- `--lr 1e-5`
- `--epoch 5001`
- `--batch_size 512`
- `--embedding_dim 1024`
- `--margin 1`
- `--_lambda 0.5`

## Results

The detailed results of the experiments carried out on the DEAM and PMEmo datasets can be found in _MER_analysis_results.xlsx_.


