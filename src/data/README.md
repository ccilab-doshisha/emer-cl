# Pre-processing of dataset
The data sets are pre-processed to convert them into the data format used by our EMER-CL.
Specifically, music records are feature extracted using VGGish, and Arousal and Valence are formatted into a appropriate format for input to the model.

## Usage
1. Download the DEAM/PMEmo dataset (links are provided in [datasets](#datasets))
    - The downloaded dataset files (zip file) should be placed as follows:
     ```
     data/
          |___README.md
          |___run.sh
          |___shell/
          |___DEAM/
          |    |___ create_va.py
          |    |___ create_vggish.py
          |    |___ create_metadata.py
          |    |___ mp3toWav.py
          |    |___ dataset/
          |          |___ [THIS] DEAM_audio.zip 
          |          |___ [THIS] DEAM_Annotations.zip 
          |          |___ [THIS] metadata.zip 
          |___PMEmo/
               |___ create_vggish.py
               |___ create_va.py
               |___ create_metadata.py
               |___ mp3toWav.py
               |___ dataset/
                    |___ [THIS] PMEmo2019.zip 
     ```
2. Running ``$WORKDIR/src/data/run.sh``.
     - It is possible to do this manually, but it is easiest to run `run.sh`, which contains the following set of operations:
          - Checking the dataset download
          - Setting up VGGish
          - Create npy files from the detaset
     - At the same time, unzip downloaded dataset zip files, so if you have the above file structure, you can finish pre-processing only run `run.sh`.


# datasets
- [DEAM dataset - The MediaEval Database for Emotional Analysis of Music](https://cvml.unige.ch/databases/DEAM/)
- [PMEmo: A Dataset For Music Emotion Computing](https://github.com/HuiZhangDB/PMEmo)
