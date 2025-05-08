import numpy as np
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from pdb import set_trace as st
from math import sqrt

from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.backend import clear_session


### Function to train a decision tree for regression
def trainRegressionDecisionTree(
        dataPath: str, # Path to the folder containing the saved features and labels
        nbTrees: int, # Number of trees to use
        foldIdx: int, # Index of the fold to be used
        dataset: str, # "DEAM" or "PMEmo"
        featuresToUse: list, # List indicating the features to use among 'cca', 'mu_kl', 'sigma_kl'
        target: str, # "arousal" or "valence"
        verbose: bool = True, # Prints the detailed results in terminal if set to True 
        ):

    if verbose:
        print('')
        print('Training a regression decision forest for %s on fold %d ...' % (target,foldIdx))

    # Load the features and labels
    trainLabels = np.load(os.path.join(dataPath,str(foldIdx).zfill(2),"trainLabels.npy"))
    testLabels = np.load(os.path.join(dataPath,str(foldIdx).zfill(2),"testLabels.npy"))

    allTrainFeatures = []
    allTestFeatures = []
    if 'cca' in featuresToUse:
        allTrainFeatures += [np.load(os.path.join(dataPath,str(foldIdx).zfill(2),"trainCCA.npy"))]
        allTestFeatures += [np.load(os.path.join(dataPath,str(foldIdx).zfill(2),"testCCA.npy"))]
    if 'mu_kl' in featuresToUse:
        allTrainFeatures += [np.load(os.path.join(dataPath,str(foldIdx).zfill(2),"trainMuKl.npy"))]
        allTestFeatures += [np.load(os.path.join(dataPath,str(foldIdx).zfill(2),"testMuKl.npy"))]
    if 'mu_sigma' in featuresToUse:
        allTrainFeatures += [np.load(os.path.join(dataPath,str(foldIdx).zfill(2),"trainSigmaKl.npy"))]
        allTestFeatures += [np.load(os.path.join(dataPath,str(foldIdx).zfill(2),"testSigmaKl.npy"))]
    trainData = np.concatenate(tuple(allTrainFeatures),axis=1)
    testData = np.concatenate(tuple(allTestFeatures),axis=1)
    
    
    # On DEAM, the target labels are the average over time of arousal and valence
    if dataset=="DEAM": # Process labels by averaging them over time
        if target == "arousal":
            trainLabels = np.mean(trainLabels[:,:,0],axis=1)
            testLabels = np.mean(testLabels[:,:,0],axis=1)
        else:
            trainLabels = np.mean(trainLabels[:,:,1],axis=1)
            testLabels = np.mean(testLabels[:,:,1],axis=1)
    # On PMEmo, the target is the last value of arousal and valence that is not equal to a padding value (default: 1e-5)
    else: 
        if target == "arousal":
            labelIdx = 0
        else:
            labelIdx = 1

        tmpTrainLabels = np.zeros(len(trainLabels))
        tmpTestLabels = np.zeros(len(testLabels))

        for idx in range(len(trainLabels)):
            paddedLabels = trainLabels[idx,:,labelIdx]
            nonPaddedLabels = paddedLabels[paddedLabels!=1e-5]
            tmpTrainLabels[idx] = nonPaddedLabels[-1]
        
        for idx in range(len(testLabels)):
            paddedLabels = testLabels[idx,:,labelIdx]
            nonPaddedLabels = paddedLabels[paddedLabels!=1e-5]
            tmpTestLabels[idx] = nonPaddedLabels[-1]

        trainLabels = tmpTrainLabels
        testLabels = tmpTestLabels

    # Train the regression forest
    model = DecisionTreeRegressor()
    model.fit(trainData,trainLabels)
    estimatedTargets = model.predict(testData)

    # Compute evaluation metrics
    rmse = sqrt(mean_squared_error(testLabels, estimatedTargets))
    r2 = r2_score(testLabels, estimatedTargets)

    # Print results
    if verbose:
        #print('################################################################')
        print('Regression results for %s on fold %d with %d trees:' % (target, foldIdx, nbTrees))
        print('RMSE = %f' % rmse)
        #print('R = %f' % sqrt(r2))
        print('R^2 = %f' % r2)
        print('')
        print('################################################################')

    # Return evaluation metrics
    if r2>=0:
        return rmse, sqrt(r2), r2
    else:
        return rmse, 0, r2

### Function to train a MLP for regression
def trainRegressionMlp(
        dataPath: str, # Path to the folder containing the saved features and labels
        foldIdx: int, # Index of the fold to be used
        dataset: str, # "DEAM" or "PMEmo"
        featuresToUse: list, # List indicating the features to use among 'cca', 'mu_kl', 'sigma_kl'
        target: str, # "arousal" or "valence"
        layers: list, # List of integers containing information regarding how many neurons are per layer
        epochs: int, # Number of epochs for the model training
        activation: str = 'relu', # Activation function for the hidden layers 
        loss: str = 'mean_squared_error', # Loss used for the training of the MLP
        optimizer: str = 'adadelta', # Optimizer used for the training of the MLP
        batchSize: int = 16, # Batch size
        verbose: bool = True, # Prints the detailed results in terminal if set to True 
        ):

    if verbose:
        print('')
        print('Training a regression MLP for %s on fold %d ...' % (target,foldIdx))

    # Load the features and labels
    trainLabels = np.load(os.path.join(dataPath,str(foldIdx).zfill(2),"trainLabels.npy"))
    testLabels = np.load(os.path.join(dataPath,str(foldIdx).zfill(2),"testLabels.npy"))

    allTrainFeatures = []
    allTestFeatures = []
    if 'cca' in featuresToUse:
        allTrainFeatures += [np.load(os.path.join(dataPath,str(foldIdx).zfill(2),"trainCCA.npy"))]
        allTestFeatures += [np.load(os.path.join(dataPath,str(foldIdx).zfill(2),"testCCA.npy"))]
    if 'mu_kl' in featuresToUse:
        allTrainFeatures += [np.load(os.path.join(dataPath,str(foldIdx).zfill(2),"trainMuKl.npy"))]
        allTestFeatures += [np.load(os.path.join(dataPath,str(foldIdx).zfill(2),"testMuKl.npy"))]
    if 'mu_sigma' in featuresToUse:
        allTrainFeatures += [np.load(os.path.join(dataPath,str(foldIdx).zfill(2),"trainSigmaKl.npy"))]
        allTestFeatures += [np.load(os.path.join(dataPath,str(foldIdx).zfill(2),"testSigmaKl.npy"))]
    trainData = np.concatenate(tuple(allTrainFeatures),axis=1)
    testData = np.concatenate(tuple(allTestFeatures),axis=1)
    
    
    # On DEAM, the target labels are the average over time of arousal and valence
    if dataset=="DEAM": # Process labels by averaging them over time
        if target == "arousal":
            trainLabels = np.mean(trainLabels[:,:,0],axis=1)
            testLabels = np.mean(testLabels[:,:,0],axis=1)
        else:
            trainLabels = np.mean(trainLabels[:,:,1],axis=1)
            testLabels = np.mean(testLabels[:,:,1],axis=1)
    # On PMEmo, the target is the last value of arousal and valence that is not equal to a padding value (default: 1e-5)
    else: 
        if target == "arousal":
            labelIdx = 0
        else:
            labelIdx = 1

        tmpTrainLabels = np.zeros(len(trainLabels))
        tmpTestLabels = np.zeros(len(testLabels))

        for idx in range(len(trainLabels)):
            paddedLabels = trainLabels[idx,:,labelIdx]
            nonPaddedLabels = paddedLabels[paddedLabels!=1e-5]
            tmpTrainLabels[idx] = nonPaddedLabels[-1]
        
        for idx in range(len(testLabels)):
            paddedLabels = testLabels[idx,:,labelIdx]
            nonPaddedLabels = paddedLabels[paddedLabels!=1e-5]
            tmpTestLabels[idx] = nonPaddedLabels[-1]

        trainLabels = tmpTrainLabels
        testLabels = tmpTestLabels

    # Define the regression MLP model
    nbLayers = len(layers)
    inputShape = trainData.shape[1:]
    inputLayer = Input(shape=inputShape)
    flatten = Flatten()(inputLayer)
    hiddenLayer = Dense(layers[0], activation=activation)(flatten)
    if nbLayers > 1:
        for idx in range(1,nbLayers):
            hiddenLayer = Dense(layers[idx], activation=activation)(hiddenLayer)
    outputLayer = Dense(1, activation='linear')(hiddenLayer)
    model = Model(inputs=inputLayer,outputs=outputLayer)
    model.compile(loss=loss, optimizer=optimizer, metrics=['mean_absolute_error'])
    
    # Train the regression MLP
    model.fit(trainData, trainLabels, epochs=epochs, batch_size=batchSize, verbose=verbose)
    estimatedTargets = model.predict(testData)
    
    # Compute evaluation metrics
    rmse = sqrt(mean_squared_error(testLabels, estimatedTargets))
    r2 = r2_score(testLabels, estimatedTargets)

    # Print results
    if verbose:
        #print('################################################################')
        print('Regression results for %s on fold %d with %d trees:' % (target, foldIdx, nbTrees))
        print('RMSE = %f' % rmse)
        #print('R = %f' % sqrt(r2))
        print('R^2 = %f' % r2)
        print('')
        print('################################################################')

    # Return evaluation metrics
    if r2>=0:
        return rmse, sqrt(r2), r2
    else:
        return rmse, 0, r2


### Function to train a SVM for regression
def trainRegressionSvm(
        C: float, # Soft-margin parameter
        kernel: str, # Kernel (e.g. "rbf", "linear")
        gamma, # Kernel parameter of type float. If set to None, the value of gamma is set automatically to 1/nbFeatures*var(data) (c.f. sklearn documenttation for parameter 'scale')
        dataPath: str, # Path to the folder containing the saved features and labels
        foldIdx: int, # Index of the fold to be used
        dataset: str, # "DEAM" or "PMEmo"
        featuresToUse: list, # List indicating the features to use among 'cca', 'mu_kl', 'sigma_kl'
        target: str, # "arousal" or "valence"
        verbose: bool = True, # Print detailed results in terminal if set to True
        ):

    if verbose:
        print('')
        print('Training a regression SVM for %s on fold %d ...' % (target,foldIdx))
        if gamma is None:
            print('Kernel set to %s, C=%.3f, gamma="scale"' % (kernel, C))
        else:
            print('Kernel set to %s, C=%.3f, gamma=%.3f' % (kernel, C, gamma))

    # Load the features and labels
    trainLabels = np.load(os.path.join(dataPath,str(foldIdx).zfill(2),"trainLabels.npy"))
    testLabels = np.load(os.path.join(dataPath,str(foldIdx).zfill(2),"testLabels.npy"))

    allTrainFeatures = []
    allTestFeatures = []
    if 'cca' in featuresToUse:
        allTrainFeatures += [np.load(os.path.join(dataPath,str(foldIdx).zfill(2),"trainCCA.npy"))]
        allTestFeatures += [np.load(os.path.join(dataPath,str(foldIdx).zfill(2),"testCCA.npy"))]
    if 'mu_kl' in featuresToUse:
        allTrainFeatures += [np.load(os.path.join(dataPath,str(foldIdx).zfill(2),"trainMuKl.npy"))]
        allTestFeatures += [np.load(os.path.join(dataPath,str(foldIdx).zfill(2),"testMuKl.npy"))]
    if 'mu_sigma' in featuresToUse:
        allTrainFeatures += [np.load(os.path.join(dataPath,str(foldIdx).zfill(2),"trainSigmaKl.npy"))]
        allTestFeatures += [np.load(os.path.join(dataPath,str(foldIdx).zfill(2),"testSigmaKl.npy"))]
    trainData = np.concatenate(tuple(allTrainFeatures),axis=1)
    testData = np.concatenate(tuple(allTestFeatures),axis=1)
    
    # On DEAM, the target labels are the average over time of arousal and valence
    if dataset=="DEAM": # Process labels by averaging them over time
        if target == "arousal":
            trainLabels = np.mean(trainLabels[:,:,0],axis=1)
            testLabels = np.mean(testLabels[:,:,0],axis=1)
        else:
            trainLabels = np.mean(trainLabels[:,:,1],axis=1)
            testLabels = np.mean(testLabels[:,:,1],axis=1)
    # On PMEmo, the target is the last value of arousal and valence that is not equal to a padding value (default: 1e-5)
    else: 
        if target == "arousal":
            labelIdx = 0
        else:
            labelIdx = 1

        tmpTrainLabels = np.zeros(len(trainLabels))
        tmpTestLabels = np.zeros(len(testLabels))

        for idx in range(len(trainLabels)):
            paddedLabels = trainLabels[idx,:,labelIdx]
            nonPaddedLabels = paddedLabels[paddedLabels!=1e-5]
            tmpTrainLabels[idx] = nonPaddedLabels[-1]
        
        for idx in range(len(testLabels)):
            paddedLabels = testLabels[idx,:,labelIdx]
            nonPaddedLabels = paddedLabels[paddedLabels!=1e-5]
            tmpTestLabels[idx] = nonPaddedLabels[-1]

        trainLabels = tmpTrainLabels
        testLabels = tmpTestLabels
    
    # Train the regression forest
    if gamma is None:
        gamma = 'scale'
    model = SVR(kernel=kernel, gamma=gamma, C=C)
    model.fit(trainData,trainLabels)
    estimatedTargets = model.predict(testData)

    # Compute evaluation metrics
    rmse = sqrt(mean_squared_error(testLabels, estimatedTargets))
    r2 = r2_score(testLabels, estimatedTargets)

    # Print results
    if verbose:
        #print('################################################################')
        print('Regression results for %s on fold %d with SVR:' % (target, foldIdx))
        print('RMSE = %f' % rmse)
        #print('R = %f' % sqrt(r2))
        print('R^2 = %f' % r2)
        print('')
        print('################################################################')

    # Return evaluation metrics
    if r2>=0:
        return rmse, sqrt(r2), r2
    else:
        return rmse, 0, r2
