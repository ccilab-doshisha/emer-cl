import numpy as np
import tensorflow as tf
import os
from pdb import set_trace as st
import typing as t
from mer_analysis_util.loss import CompositeLoss
from mer_analysis_util.model import EMER_CL
from nptyping import NDArray
from mer_analysis_util.linear_cca import LinearCCA
from mer_analysis_util.parser import parser, save_params
from mer_analysis_util.util import batch_generator, get_now
from tensorflow.keras.preprocessing.sequence import pad_sequences



# Function to load a trained model to compute and save music encoder features given some dataset
def computeMusicEncoderFeatures(
        sess: tf.compat.v1.Session,
        model: EMER_CL, # Model to be used for inference
        model_path: str, # Path to the folder where the models for the 10-fold-cv are saved
        fold_idx: int, # Index of the fold to consider
        data_path: str, # Path to the folder where the data for the 10-fold-cv are saved
        save_path: str, # Path to the folder where the features shoudl be saved
        dataset: str, # 'DEAM' or 'PMEmo'
    ) -> None:
    
    output_tensors = model.get_output_tensors()

    # Restoring the saved session
    path_to_fold_model = os.path.join(model_path, str(fold_idx).zfill(2))
    trained_models = list(os.listdir(path_to_fold_model)) # NOTE: if several models are contained in the folder, the latest trained one is assumed to be the relevant one
    trained_models.sort()
    path_to_fold_model = os.path.join(path_to_fold_model, trained_models[-1])
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, os.path.join(path_to_fold_model, 'model.ckpt'))
    
    # Load the data
    path_to_fold_data = os.path.join(data_path, str(fold_idx).zfill(2))
    train_data = np.load(os.path.join(path_to_fold_data, 'trainData.npy'),allow_pickle=True) # VGGish features on the training set
    test_data = np.load(os.path.join(path_to_fold_data, 'testData.npy'),allow_pickle=True) # VGGish features on the testing set
    train_labels = np.load(os.path.join(path_to_fold_data, 'trainLabels.npy'),allow_pickle=True) # Arousal/valence ratings on the training set
    test_labels = np.load(os.path.join(path_to_fold_data, 'testLabels.npy'),allow_pickle=True) # Arousal/valence ratins on the testing set

    # For DEAM, average the music encoder (MLP) input over time 
    if dataset == 'DEAM':
        train_data = np.mean(train_data,axis=1)
        test_data = np.mean(test_data,axis=1)
    else: # Padding sequences is required since different lengths are available on PMEmo
        train_data = pad_sequences(train_data, padding="post", dtype="float32", value=1e-5)
        test_data = pad_sequences(test_data, padding="post", dtype="float32", value=1e-5)
        train_labels = pad_sequences(train_labels, padding="post", dtype="float32", value=1e-5)
        test_labels = pad_sequences(test_labels, padding="post", dtype="float32", value=1e-5) 

    # Feed the data to the music encoder and save its output
    train_cca, train_mu_kl, train_sigma_kl = sess.run(
            [
                output_tensors["music_enc"]["emb"],
                output_tensors["music_enc"]["mu"],
                output_tensors["music_enc"]["log_sigma"], 
            ],
            feed_dict = {model.input_music: train_data, model.input_emotion: train_labels}            
            )

    test_cca, test_mu_kl, test_sigma_kl = sess.run(
            [
                output_tensors["music_enc"]["emb"],
                output_tensors["music_enc"]["mu"],
                output_tensors["music_enc"]["log_sigma"], 
            ],
            feed_dict = {model.input_music: test_data, model.input_emotion: test_labels}            
            )

    # Save computed features and labels
    save_feature_path = os.path.join(save_path, str(fold_idx).zfill(2))
    if not os.path.exists(save_feature_path):
        os.makedirs(save_feature_path)

    np.save(os.path.join(save_feature_path,"trainLabels.npy"), train_labels)
    np.save(os.path.join(save_feature_path,"testLabels.npy"), test_labels)
    np.save(os.path.join(save_feature_path,"trainCCA.npy"), train_cca)
    np.save(os.path.join(save_feature_path,"trainMuKl.npy"), train_mu_kl)
    np.save(os.path.join(save_feature_path,"trainSigmaKl.npy"), train_sigma_kl)
    np.save(os.path.join(save_feature_path,"testCCA.npy"), test_cca)
    np.save(os.path.join(save_feature_path,"testMuKl.npy"), test_mu_kl)
    np.save(os.path.join(save_feature_path,"testSigmaKl.npy"), test_sigma_kl)


####################################################################################################################
### Main
### TODO: change the hyper-parameters accordingly before executing the script
###################################################################################################################
if __name__ == "__main__":

    # Hyper-parameters
    dataset = "PMEmo" # 'DEAM' or 'PMEmo'
    model_path = os.path.join('./../model', dataset) # Path where the trained models are saved
    data_path = os.path.join('./../data', dataset, '10_fold_cv_dataset') # Path to the folder where the data for the 10-fold-cv are saved
    save_path = os.path.join('./../data', dataset, 'music_encoder_features')  # Path where to save the music encoder features

    # Feature computation
    for fold_idx in range(1,11):
        print('Computing features for fold %d ...' % fold_idx)
        # Graph initialisation
        tf.compat.v1.reset_default_graph()

        # Model definition
        model = EMER_CL(dataset, 1024)
        tf_config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
        sess = tf.compat.v1.Session(config=tf_config)

        computeMusicEncoderFeatures(
            sess=sess,
            model=model,
            model_path=model_path,
            fold_idx=fold_idx, # Index of the fold to consider
            data_path=data_path,
            save_path=save_path,
            dataset=dataset,
        )
