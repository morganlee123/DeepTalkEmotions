# Author: Morgan Sandler (sandle20@msu.edu)
#
# Generates a MSP-Podcast pandas DataFrame that prepares data for
# embedding extraction
#
# Experiment 2 - Step 1
#
import numpy as np
import pandas as pd
import os

# SECTION 1 -------------- Load MSP Data Labels TODO: Add your own path here
msp_labels = "/research/iprobe-sandle20/sandle20/MSP-Podcast/www.lab-msp.com/MSP-PODCAST-Publish-1.8/Labels/labels_concensus.csv"

msp_dir = pd.read_csv(msp_labels)
test_set = msp_dir[msp_dir['Split_Set'] == 'Test1']
#print(test_set.groupby('EmoClass').count())
#print(test_set.head())
test_set = test_set.drop(columns=['EmoAct', 'EmoVal', 'EmoDom', 'Gender', 'Split_Set'])
#print('Unique speakers', len(test_set.SpkrID.unique()))

print(test_set.head())

train_set = msp_dir[msp_dir['Split_Set'] == 'Train']
train_set = train_set.drop(columns=['EmoAct', 'EmoVal', 'EmoDom', 'Gender', 'Split_Set'])
print(train_set.head())


test_set.to_parquet('./working_data/msp_test.pq')
train_set.to_parquet('./working_data/msp_train.pq')
print('Successfully saved the DFs to a parquet in ./working_data')

