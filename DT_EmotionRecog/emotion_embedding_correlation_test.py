"""
Goal of this program:
Run experiment on one subject to find 
intravariation between emotional samples 
and neutral samples, and how to best
extract these for a classifier
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cosine

SUBJECT = 1004

# open embeddings
data = pd.read_pickle('./working_data/1000_extracted_embeddings.pk')

# Data filtering. Remove discards
data_filtered = []
for row, v in data.iterrows():
    # MAPPING {'disgust':0,'happy':1,'sad':2,'neutral':3,'fear':4,'angry':5}
    if v.labels != 'discard': #and (v.labels == 1 or v.labels == 2 or v.labels==5 or v.labels==3): # this uncomment is for 4 emotions
        data_filtered.append(v)
print(len(data_filtered)/len(data), '% valid data ---', len(data_filtered), 'samples')
embeddings = []
subjects = []
labels = []
for em in data_filtered:
    embeddings.append(em['dt'])
    subjects.append(em['subject'])
    labels.append(em['labels'])
        

X = np.array(embeddings)
X_sub = np.array(subjects)
y = np.array(labels)
#X = np.asarray(data_filtered['dt'])
#X_sub = np.asarray(data_filtered['subject'])
#y = np.asarray(data_filtered["labels"])

min_length = int(np.mean([len(i) for i in X]))
    
X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=min_length, dtype='float32')
print(X.shape)

to_investigate_x = []
to_investigate_y = []
for i in range(X.shape[0]):#7371):
    if data['subject'][i] == SUBJECT:
        to_investigate_x.append(X[i])
        to_investigate_y.append(y[i])

to_investigate_x = np.array(to_investigate_x)
nsamples, nx, ny = to_investigate_x.shape
to_investigate_x = to_investigate_x.reshape((nsamples,nx*ny))

to_sort = []
for i, embed in enumerate(to_investigate_x):
    to_sort.append((1 - cosine(embed, to_investigate_x[i-1]), to_investigate_y[i], to_investigate_y[i-1]))

# MAPPING {'disgust':0,'happy':1,'sad':2,'neutral':3,'fear':4,'angry':5}

to_sort.sort(reverse=True)
for i in to_sort:
    print('emo', i[1], i[2],'embed', i[0])
