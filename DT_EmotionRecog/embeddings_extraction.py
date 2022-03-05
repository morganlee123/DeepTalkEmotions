from pandas import read_parquet
import numpy as np
import pandas as pd

# Run this code with deeptalk environment
# SECTION 2 -------------- Feature extraction

Crema_df = read_parquet('./working_data/crema.pq')

labels = {'disgust':0,'happy':1,'sad':2,'neutral':3,'fear':4,'angry':5}
Crema_df.replace({'Emotions':labels},inplace=True)

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

for i in range(1000):#7442):
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


data_df = pd.DataFrame(data)
data_df.to_pickle('/research/iprobe-sandle20/sandle20/SpeechEmotionRecognitionExperiments/DeepTalk/DT_EmotionRecog/working_data/1000_extracted_embeddings.pk')