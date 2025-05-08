import numpy as np
from pdb import set_trace as st
import os
from math import floor
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--dataset",
    default=None,
    type=str,
    required=True,
    help="dataset name (DEAM or PMEmo)",
)
params = parser.parse_args()

# Function to shuffle data and array in unisson
def shuffleInUnisson(a,b,returnPermutation=False):
    assert len(a)==len(b)
    permutation = np.random.permutation(len(a))
    if returnPermutation:
        return a[permutation], b[permutation], permutation
    else:
        return a[permutation], b[permutation]


# Function to create k folds on the target dataset and save the data
def createKFolds(
        dataPath, # Path to the computed VGGish features (.npy format)
        labelPath, # Path to the extracted arousal/valence labels (.npy format)
        savePath, # Path where to save the folds and associated labels
        k=10, # Number of folds
        dataset='DEAM', # 'DEAM' or 'PMEmo'
        ):

    print('Creating %d folds on the specified dataset ...' % k)
    
    # Load data and labels
    data = np.load(dataPath,allow_pickle=True)
    labels = np.load(labelPath,allow_pickle=True)

    # Shuffle data and labels in unisson
    data, labels = shuffleInUnisson(data,labels)

    # Separate the dataset into k parts of approximate size
    foldSize = int(floor(len(data)/k))
    foldInfo = {} # Dictionary with (key,value) = (foldIdx,(foldData,foldLabels))

    for foldIdx in range(k):
        if foldIdx != k-1:
            foldInfo[foldIdx] = (data[foldSize*foldIdx:foldSize*(foldIdx+1)],labels[foldSize*foldIdx:foldSize*(foldIdx+1)])
        else:
            foldInfo[foldIdx] = (data[foldSize*foldIdx:],labels[foldSize*foldIdx:])

    # Build the folds and save them in the specified folder
    for foldIdx in range(k):
        print('Building fold %d/%d' % (foldIdx+1,k))
        testData, testLabels = foldInfo[foldIdx]
        # Initialise numpy arrays for the training set
        if foldIdx == k-1:
            nbTrainExamples = (k-1)*foldSize
        else:
            nbTrainExamples = (k-2)*foldSize + len(foldInfo[k-1][0])
        
        if dataset=="DEAM":
            trainData = np.zeros((nbTrainExamples,testData.shape[1],testData.shape[2]))
            trainLabels = np.zeros((nbTrainExamples,testLabels.shape[1],testLabels.shape[2]))
        
            # Fill out training data and label arrays
            currentIdx = 0
            for idx in foldInfo.keys():
                if idx != foldIdx:
                    currentNbExamples = foldInfo[idx][0].shape[0]
                    trainData[currentIdx:currentIdx+currentNbExamples] = foldInfo[idx][0]
                    trainLabels[currentIdx:currentIdx+currentNbExamples] = foldInfo[idx][1]
                    currentIdx += currentNbExamples
        
        elif dataset=="PMEmo": 
            if foldIdx != 0:
                trainData = foldInfo[0][0]
                trainLabels = foldInfo[0][1]
                for key, value in foldInfo.items():
                    if key != 0 and key != foldIdx:
                        trainData = np.concatenate((trainData,value[0]))
                        trainLabels = np.concatenate((trainLabels,value[1]))
            else:
                trainData = foldInfo[1][0]
                trainLabels = foldInfo[1][1]
                for key, value in foldInfo.items():
                    if key != 1 and key != foldIdx:
                        trainData = np.concatenate((trainData,value[0]))
                        trainLabels = np.concatenate((trainLabels,value[1]))

        else:
            print('Unsupported dataset %s!' % dataset)
            print('Currently supported: "DEAM" and "PMEmo"')
            exit()
         
        # Save data in the specified folder
        currentPath = os.path.join(savePath,str(foldIdx+1).zfill(2))
        try:
            os.makedirs(currentPath)
        except FileExistsError as error:
            print('Note: directory %s already exists. Contents may be overwritten!' % currentPath)
    
        np.save(os.path.join(currentPath,'trainData.npy'),trainData)
        np.save(os.path.join(currentPath,'trainLabels.npy'),trainLabels)
        np.save(os.path.join(currentPath,'testData.npy'),testData)
        np.save(os.path.join(currentPath,'testLabels.npy'),testLabels)

    print('All folds created at %s' % savePath)


##############################################################################################################
### Main
### TODO: modify parameters there according to preferences before running the script
##############################################################################################################
def main(dataset):
    createKFolds(
        dataPath=f"./../data/{dataset}/vggish_feat.npy",
        labelPath=f"./../data/{dataset}/va_data.npy",
        savePath=f"./../data/{dataset}/10_fold_cv_dataset/",
        k=10,
        dataset=dataset
    ) 

if __name__ == '__main__':
    main(params.dataset)
