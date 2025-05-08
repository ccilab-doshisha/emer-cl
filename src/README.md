# Usage for train and evaluation

## Note
You need to prepare the data at first.
Please refer to ``$WORKDIR/src/data/README.md`` for details.

## Train EMER-CL model
`train.py` is used to train and evaluate the model.

```sh
# Training execution
python $WORKDIR/src/train.py --dataset=[DEAM/PMEmo] --epoch=[num of epoch]

# Evaluation results
============================== M2E ==============================
<M2E MRR result>
<M2E AR result>
=================================================================

============================== E2M ==============================
<E2M MRR result>
<E2M AR result>
=================================================================
```
If you want to check the details of the command arguments or change the parameters, you can use `-h` option to check and change the parameters.

```sh
python train.py -h

usage: train.py [-h] [--CUDA_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES]
                [--epoch EPOCH] [--lr LR] [--batch_size BATCH_SIZE]
                [--embedding_dim EMBEDDING_DIM] [--margin MARGIN]
                [--_lambda _LAMBDA] [--corr_p CORR_P] --dataset DATASET

optional arguments:
  -h, --help            show this help message and exit
  --CUDA_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
                        CUDA GPU number
  --epoch EPOCH         epoch num for training
  --lr LR               learning rate for training
  --batch_size BATCH_SIZE
                        batch size for training
  --embedding_dim EMBEDDING_DIM
                        embedding dimension will be written. Default : 1024
  --margin MARGIN       the margin for rankloss
  --_lambda _LAMBDA     the control parameter lambda (CCALOSS vs RANKLOSS)
  --corr_p CORR_P       threshold for correlation-based similarity in evaluation (need float)
  --dataset DATASET     dataset name (DEAM or PMEmo)

```

## Restore the trained model
After training, the trained model is saved in `$WORKDIR/src/model`.
You can use `restore.py` to restore and evaluate the saved models.

```sh
# Restore execution
python $WORKDIR/src/restore.py --datetime=[YYYY_mm_HH_MM_SS] --dataset=[DEAM/PMEmo]

# Evaluation results
============================== M2E ==============================
<M2E MRR result>
<M2E AR result>
=================================================================

============================== E2M ==============================
<E2M MRR result>
<E2M AR result>
=================================================================
```
