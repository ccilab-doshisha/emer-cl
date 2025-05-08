import numpy as np
import os
from mer_analysis_util.regression_model import trainRegressionSvm
from pdb import set_trace as st


############################################################################################################
### Main
### TODO: change hyper-parameters accordingly before executing the script
###########################################################################################################
if __name__ == '__main__':

    # Hyper parameters for the grid search with RBF C-SVR
    allC = [0.5,1,2,32,128,256,512] # Values of the soft-margin to be tested
    allGamma = [0.5,1,2,32,'scale'] # Values of the RBF kernel to be tested
    target = 'valence' # 'arousal' or 'valence'
    dataset = 'PMEmo' # 'DEAM' or 'PMEmo'
    features = ['cca', 'mu_kl', 'sigma_kl'] # EMER-CL features to be used. Must contain one or several of 'cca', 'mu_kl', 'sigma_kl'
    data_path = os.path.join('./../data', dataset, 'music_encoder_features') # Path to the folder containing the saved features and labels


    # Grid search
    # Variable initialisation
    bestR2 = 0 # optimisation using R2 as main evaluation metric
    bestC = -1
    bestGamma = -1

    print('')
    print('Training %s regression models on the %s dataset ...' % (target, dataset))

    # Starting the grid search
    for C in allC:
        for gamma in allGamma:
            
            print('')
            print('##############################################################')
            if type(gamma) != str:
                print('Testing configuration with C=%.5f and gamma=%.5f' % (C,gamma))
            else:
                print('Testing configuration with C=%.5f and gamma=%s' % (C,gamma))

            allRmse = np.zeros(10)
            allR = np.zeros(10)
            allR2 = np.zeros(10)

            # Loop on all folds
            for idx in range(1,11):

                rmse, r, r2 = trainRegressionSvm(
                    C=C, # Soft-margin parameter
                    kernel='rbf', # Kernel (e.g. "rbf", "linear")
                    gamma=gamma, # Kernel parameter of type float. If set to None, the value of gamma is set automatically to 1/nbFeatures*var(data) (c.f. sklearn documentation for parameter 'scale')
                    dataPath=data_path, # Path to the folder containing the saved features and labels
                    foldIdx=idx, # Index of the fold to be used
                    dataset=dataset, # "DEAM" or "PMEmo"
                    featuresToUse=features, # List indicating the features to use among 'cca', 'mu_kl', 'sigma_kl'
                    target=target, # "arousal" or "valence"
                    verbose=False
                    )

                allRmse[idx-1] = rmse
                allR[idx-1] = r
                allR2[idx-1] = r2

            # Compute average R2
            if np.mean(allR2) > bestR2:
                bestR2 = np.mean(allR2)
                bestC = C
                bestGamma = gamma
                bestFoldRmse = allRmse
                bestFoldR = allR
                bestFoldR2 = allR2

            print('--------------------------------------------------------------')
            print('Overall RMSE: %.6f +- %.6f' % (np.mean(allRmse),np.std(allRmse)))
            print('Overall R: %.6f +- %.6f' % (np.mean(allR),np.std(allR)))
            print('Overall R^2: %.6f +- %.6f' % (np.mean(allR2),np.std(allR2)))
            print('##############################################################')

    # Display best parameters
    print('')
    if type(bestGamma) != str:
        print('Best R2 obtained for C=%.5f and gamma=%.5f' % (bestC, bestGamma))
    else:
        print('Best R2 obtained for C=%.5f and gamma=%s' % (bestC, bestGamma))

    print('')
    print('Best RMSE: %.6f +- %.6f' % (np.mean(bestFoldRmse),np.std(bestFoldRmse)))
    print(bestFoldRmse)
    print('')
    print('Best R: %.6f +- %.6f' % (np.mean(bestFoldR),np.std(bestFoldR)))
    print(bestFoldR)
    print('')
    print('Best R2: %.6f +- %.6f' % (np.mean(bestFoldR2),np.std(bestFoldR2)))
    print(bestFoldR2)
