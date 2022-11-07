# Author: Morgan Sandler (sandle20@msu.edu)
#
# Generates a CREMA-D pandas DataFrame that prepares data for
# embedding extraction
#
# Experiment 1 - Step 1
# AKA
# Experiment 3 - Step 1

import numpy as np
import pandas as pd
import os

# SECTION 1 -------------- Load CREMA Data

# regular crema TODO: Add your own CREMA path here
Crema = "/research/iprobe-sandle20/sandle20/CREMA-D/AudioWAV/"

crema_directory_list = os.listdir(Crema)

file_emotion = []
file_path = []
file_subject = []

for file in crema_directory_list:
    # storing file paths
    file_path.append(Crema + file)
    # storing file emotions
    part=file.split('_')
    
    # part[0] refers to the subject number. start new list for this information
    file_subject.append(int(part[0]))
    
    if part[2] == 'SAD':
        file_emotion.append('sad')
    elif part[2] == 'ANG':
        file_emotion.append('angry')
    elif part[2] == 'DIS':
        file_emotion.append('disgust')
    elif part[2] == 'FEA':
        file_emotion.append('fear')
    elif part[2] == 'HAP':
        file_emotion.append('happy')
    elif part[2] == 'NEU':
        file_emotion.append('neutral')
    else:
        file_emotion.append('Unknown')
        
# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])

# dataframe for the subject numbers
subjects_df = pd.DataFrame(file_subject, columns=['Subject'])

Crema_df = pd.concat([emotion_df, path_df, subjects_df], axis=1)

Crema_df.to_parquet('./working_data/noisy_crema.pq')
print('Successfully saved the DF to a parquet in ./working_data')
