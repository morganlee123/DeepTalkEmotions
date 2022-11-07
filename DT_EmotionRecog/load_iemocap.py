# Author: Morgan Sandler (sandle20@msu.edu)
#
# Generates a IEMOCAP pandas DataFrame that prepares data for
# embedding extraction
#

import numpy as np
import pandas as pd
import os

# SECTION 1 -------------- Load IEMOCAP Data

# parameters
root='/research/iprobe/datastore/datasets/speech/usc-iemocap-speech'
emotions=['ang', 'hap', 'exc', 'sad', 'neu']
sessions=[3]#[1, 2, 4, 5] # change this to create test and train sets accordingly
script_impro=['script', 'impro']
genders=['M', 'F']

_ext_audio = '.wav'
_emotions = { 'ang': 0, 'hap': 1, 'exc': 1, 'sad': 3, 'fru': 4, 'fea': 5, 'sur': 6, 'neu': 7, 'xxx': 8 }


data = []
for i in range(1, 6):
    # Define path to evaluation files of this session
    path = os.path.join(root, 'Session' + str(i), 'dialog', 'EmoEvaluation')

    # Get list of evaluation files
    files = [file for file in os.listdir(path) if file.endswith('.txt')]

    # Iterate through evaluation files to get utterance-level data
    for file in files:
        # Open file
        f = open(os.path.join(path, file), 'r')

        # Get list of lines containing utterance-level data. Trim and split each line into individual string elements.
        data += [line.strip()
                        .replace('[', '')
                        .replace(']', '')
                        .replace(' - ', '\t')
                        .replace(', ', '\t')
                        .split('\t')
                    for line in f if line.startswith('[')]

# Get session number, script/impro, speaker gender, utterance number
data = [d + [d[2][4], d[2].split('_')[1], d[2][-4], d[2][-3:]] for d in data]

# Create pandas dataframe
df = pd.DataFrame(data, columns=['start', 'end', 'file', 'emotion', 'activation', 'valence', 'dominance', 'session', 'script_impro', 'gender', 'utterance'], dtype=np.float32)

# Filter by emotions
filtered_emotions = df['emotion'].isin(emotions)
df = df[filtered_emotions]

# Filter by sessions
filtered_sessions = df['session'].isin(sessions)
df = df[filtered_sessions]

# Filter by script_impro
filtered_script_impro = df['script_impro'].str.contains('|'.join(script_impro))
df = df[filtered_script_impro]

# Filter by gender
filtered_genders = df['gender'].isin(genders)
df = df[filtered_genders]

# Reset indices
df = df.reset_index()

# Map emotion labels to numeric values
df['emotion'] = df['emotion'].map(_emotions).astype(np.float32)

# Map file to correct path w.r.t to root
df['file'] = [os.path.join('Session' + file[4], 'sentences', 'wav', file[:-5], file + _ext_audio) for file in df['file']]

print(df.head())
df.to_parquet('./working_data/iemocap3.pq')
print('Successfully saved the DF to a parquet in ./working_data')
