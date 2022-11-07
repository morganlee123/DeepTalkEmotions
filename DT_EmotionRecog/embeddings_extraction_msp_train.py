# Author: Morgan Sandler (sandle20@msu.edu)
#
# Extracts train DeepTalk Embeddings from MSP-Pod and stores them as a pickle in
# the specified below directory
#
# Experiment 2 - Step 2
#

from pandas import read_parquet
import numpy as np
import pandas as pd

# Run this code with deeptalk environment
# SECTION 2 -------------- Feature extraction

msp_train = read_parquet('./working_data/msp_train.pq')
#msp_train = read_parquet('./working_data/msp_train.pq')

labels = {'H':1,'S':2,'N':3,'A':5}
msp_train.replace({'EmoClass':labels},inplace=True)
#msp_train.replace({'EmoClass':labels},inplace=True)


# dt_func has helper functon to get DT embeddings
import sys
sys.path.append("..") # Adds higher directory to python modules path.
import demo_functions
SAMPLE_RATE = 22050

data = {
        "subject": [],
        "labels": [],
        "dt": []
    }

print('Will only extract ', int(len(msp_train) / 4),'embeddings')
for i in range(int(len(msp_train) / 4)):
    data['subject'].append(msp_train.iloc[i,2])
    data['labels'].append(msp_train.iloc[i,1])
    #signal, sample_rate = librosa.load(Crema_df.iloc[i,1], sr=SAMPLE_RATE)
    
    # Compute DeepTalk embeddings
    # TODO: Insert your personalized path to MSP dataset here
    msp_prefix='/research/iprobe-sandle20/sandle20/MSP-Podcast/www.lab-msp.com/MSP-PODCAST-Publish-1.8/AudioFiles/Audios_wav/'
    try:
        print('opening', msp_prefix + msp_train.iloc[i, 0])
        embeddings = demo_functions.run_DeepTalk_demo(ref_audio_path=(msp_prefix +msp_train.iloc[i, 0]))
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


data_df = pd.DataFrame(data)
data_df.to_pickle('./working_data/msp_train_extracted_embeddings_quarter.pk')