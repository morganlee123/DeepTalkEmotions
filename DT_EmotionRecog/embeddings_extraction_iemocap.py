# Author: Morgan Sandler (sandle20@msu.edu)
#
# Extracts DeepTalk Embeddings from IEMOCAP and stores them as a pickle in
# the specified below directory as iemocap_embeds.pk
#

from pandas import read_parquet
import numpy as np
import pandas as pd

# Run this code with deeptalk environment
# SECTION 2 -------------- Feature extraction

iemocap_df = read_parquet('./working_data/iemocap3.pq')


print(iemocap_df.head(10))

import sys
sys.path.append("..") # Adds higher directory to python modules path.
import demo_functions
SAMPLE_RATE = 16000

root_dir = '/research/iprobe/datastore/datasets/speech/usc-iemocap-speech/'

data = {
        "subject": [],
        "labels": [],
        "dt": []
    }


for i, row in iemocap_df.iterrows():
    data['subject'].append(row['file'])
    data['labels'].append(row['emotion'])
    try:
        print('opening', row['file'])
        embeddings = demo_functions.run_DeepTalk_demo(ref_audio_path=root_dir+row['file'])
    except:
        print('bad embedding')
        embeddings = None
        data['labels'][i] = 'discard'

    data["dt"].append(np.asarray(embeddings))
    if i%500==0:
        print(i)
    
"""
Crema_df.replace({'Emotions':labels},inplace=True)

# dt_func has helper functon to get DT embeddings
import sys
sys.path.append("..") # Adds higher directory to python modules path.
import demo_functions
SAMPLE_RATE = 16000

data = {
        "subject": [],
        "labels": [],
        "dt": []
    }

for i in range(7442):
    print(i)
    data['subject'].append(Crema_df.iloc[i,2])
    data['labels'].append(Crema_df.iloc[i,0])
    #signal, sample_rate = librosa.load(Crema_df.iloc[i,1], sr=SAMPLE_RATE)
    
    # Compute DeepTalk embeddings
    try:
        print('opening', Crema_df.iloc[i, 1])
        embeddings = demo_functions.run_DeepTalk_demo(ref_audio_path=Crema_df.iloc[i, 1])
    except:
        print('bad embedding')
        embeddings = None
        data['labels'][i] = 'discard'




    #if embeddings. == None:
    #    continue
    #print(embeddings)    
    #embeddings = embeddings.T
    data["dt"].append(np.asarray(embeddings))
    if i%500==0:
        print(i)

"""
data_df = pd.DataFrame(data)
data_df.to_pickle('./working_data/iemocap_extracted_embeddings_test.pk')
