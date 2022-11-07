# Author: Morgan Sandler (sandle20@msu.edu)
#
# Runs the experiment 3 from the paper. This program
# implements the hierarchical classifier with improved performance
# numbers. At the end of the program there are confusion matrix generators and
# then f-score reports are printed
#
# Dataset used: CREMA-D
#
# Experiment 3 - Step 3
#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from sklearn.model_selection import train_test_split
import joblib

# open embeddings
data = pd.read_pickle('./working_data/extracted_embeddings.pk')

# Data filtering. Remove discards
data_filtered_sad = []
for row, v in data.iterrows():
    # MAPPING {'disgust':0,'happy':1,'sad':2,'neutral':3,'fear':4,'angry':5}
    if v.labels != 'discard' and (v.labels == 1 or v.labels == 2 or v.labels==5 or v.labels==3):
        if (v.labels == 1 or v.labels == 5 or v.labels == 3):
            d = v
            d.labels = 0
            data_filtered_sad.append(d)
        else:
            data_filtered_sad.append(v)

data_filtered_other = []
for row, v in data.iterrows():
    # MAPPING {'disgust':0,'happy':1,'sad':2,'neutral':3,'fear':4,'angry':5}
    if v.labels != 'discard' and (v.labels == 1 or v.labels == 2 or v.labels==5 or v.labels==3):
        if (v.labels == 1 or v.labels == 5 or v.labels == 3):
            data_filtered_other.append(v)


data_filtered_full = []
for row, v in data.iterrows():
    # MAPPING {'disgust':0,'happy':1,'sad':2,'neutral':3,'fear':4,'angry':5}
    if v.labels != 'discard' and (v.labels == 1 or v.labels == 2 or v.labels==5 or v.labels==3):
        data_filtered_full.append(v)

embeddings_sad = []
subjects_sad = []
labels_sad = []
for em in data_filtered_sad:
    embeddings_sad.append(em['dt'])
    subjects_sad.append(em['subject'])
    labels_sad.append(em['labels'])

# TODO: Make these samples balanced for the 4 classes
X_sad = np.array(embeddings_sad)
X_sub_sad = np.array(subjects_sad)
y_sad = np.array(labels_sad)


X_prep_sad = []
for e in X_sad:
    X_prep_sad.append(np.mean(e,axis=0))
X_sad = np.array(X_prep_sad)



################# OTHER
embeddings_other = []
subjects_other = []
labels_other = []
for em in data_filtered_other:
    embeddings_other.append(em['dt'])
    subjects_other.append(em['subject'])
    labels_other.append(em['labels'])

# TODO: Make these samples balanced for the 4 classes
X_other = np.array(embeddings_other)
X_sub_other = np.array(subjects_other)
y_other = np.array(labels_other)


X_prep_other = []
for e in X_other:
    X_prep_other.append(np.mean(e,axis=0))
X_other = np.array(X_prep_other)
#################
embeddings_full = []
subjects_full = []
labels_full = []
for em in data_filtered_full:
    embeddings_full.append(em['dt'])
    subjects_full.append(em['subject'])
    labels_full.append(em['labels'])

X_full = np.array(embeddings_full)
X_sub_full = np.array(subjects_full)
y_full = np.array(labels_full)


X_prep_full = []
for e in X_full:
    X_prep_full.append(np.mean(e,axis=0))
X_full = np.array(X_prep_full)

print(X_sad.shape)
print(X_other.shape)
print(X_full.shape)

# Subject disjointness between train/validation/test

possible_subjects = list(range(1001, 1092))

TRAIN_SIZE = 0.7 # Training sample of the data
TEST_SIZE = 0.3 # Used for final testing of performance

# Determine which subjects belong to which set
RANDOM_STATE = 600
X_train_subs, X_test_subs = train_test_split(possible_subjects, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# There are now 2 sets of subjects X_train_subs | X_test_subs
# AKA INDEPENDENCE WOO

X_train_sad = []
y_train_sad = []
X_test_sad = []
y_test_sad = []
X_train_other = []
y_train_other = []
X_test_other = []
y_test_other = []
X_train_full = []
y_train_full = []
X_test_full = []
y_test_full = []


for i in range(X_sad.shape[0]):#7371):
    if data['subject'][i] in X_train_subs:
        X_train_sad.append(X_sad[i])
        y_train_sad.append(y_sad[i])
    elif data['subject'][i] in X_test_subs:
        X_test_sad.append(X_sad[i])
        y_test_sad.append(y_sad[i])

for i in range(X_other.shape[0]):#7371):
    if data['subject'][i] in X_train_subs:
        X_train_other.append(X_other[i])
        y_train_other.append(y_other[i])
    elif data['subject'][i] in X_test_subs:
        X_test_other.append(X_other[i])
        y_test_other.append(y_other[i])

for i in range(X_full.shape[0]):#7371):
    if data['subject'][i] in X_train_subs:
        X_train_full.append(X_full[i])
        y_train_full.append(y_full[i])
    elif data['subject'][i] in X_test_subs:
        X_test_full.append(X_full[i])
        y_test_full.append(y_full[i])

X_train_sad = np.asarray(X_train_sad)
X_test_sad = np.asarray(X_test_sad)

y_train_sad = np.asarray(y_train_sad)
y_test_sad = np.asarray(y_test_sad)


X_train_other = np.asarray(X_train_other)
X_test_other = np.asarray(X_test_other)

y_train_other = np.asarray(y_train_other)
y_test_other = np.asarray(y_test_other)

X_train_full = np.asarray(X_train_full)
X_test_full = np.asarray(X_test_full)

y_train_full = np.asarray(y_train_full)
y_test_full = np.asarray(y_test_full)

# TODO: fix this balance
from imblearn.datasets import make_imbalance
# 0 - others 2 - sad
X_train_sad, y_train_sad = make_imbalance(X_train_sad, y_train_sad, sampling_strategy={0: 720, 2: 720},random_state=RANDOM_STATE)
X_test_sad, y_test_sad = make_imbalance(X_test_sad, y_test_sad, sampling_strategy={0: 320, 2: 320},random_state=RANDOM_STATE)
print('Balanced emotion classes')

print(y_train_other)
X_train_other, y_train_other = make_imbalance(X_train_other, y_train_other, sampling_strategy={1:720, 3: 720, 5:720},random_state=RANDOM_STATE)
X_test_other, y_test_other = make_imbalance(X_test_other, y_test_other, sampling_strategy={1: 300, 3: 300, 5:300},random_state=RANDOM_STATE)
print('Balanced emotion classes')


X_train_full, y_train_full = make_imbalance(X_train_full, y_train_full, sampling_strategy={1:720,2:720, 3: 720, 5:720},random_state=RANDOM_STATE)
X_test_full, y_test_full = make_imbalance(X_test_full, y_test_full, sampling_strategy={1: 320, 2:320, 3: 320, 5:320},random_state=RANDOM_STATE)
print('Balanced emotion classes')

# SIZES OF TRAIN, VAL, TEST
print(X_train_sad.shape,y_train_sad.shape,X_test_sad.shape,y_test_sad.shape)
print(X_train_other.shape,y_train_other.shape,X_test_other.shape,y_test_other.shape)
print(X_train_full.shape,y_train_full.shape,X_test_full.shape,y_test_full.shape)

# LR Train
#model = LogisticRegression(random_state=10, multi_class='multinomial', penalty='l2', solver='newton-cg').fit(X_train, y_train)
#model = LogisticRegression(random_state=5, multi_class='multinomial', penalty='l2', solver='newton-cg').fit(X_train_maxpooled, y_train)

# SVM Train
sad_model = SVC(random_state=99999, kernel='rbf', gamma=0.1, C=1000, probability=True).fit(X_train_sad, y_train_sad)
other_model = SVC(random_state=99999, kernel='rbf', gamma=0.1, C=1000, probability=True).fit(X_train_other, y_train_other)

# Trained model to disriminate sad to other emotions
# 1 - others, 2 - sad
from sklearn.metrics import f1_score
sad_model_predicts = sad_model.predict(X_test_sad)
print('weighted f-score', f1_score(y_test_sad, sad_model_predicts, average='weighted'))
print('Accuracy', sad_model.score(X_test_sad, y_test_sad))
conf_mat = confusion_matrix(y_test_sad, sad_model_predicts)
print(conf_mat)

# Trained model to discriminate the other emotions
# 1 - Happy 2 - Neutral 3 - Angry
other_model_predicts = other_model.predict(X_test_other)
print('other weighted f-score', f1_score(y_test_other, other_model_predicts, average='weighted'))
print('other Accuracy', other_model.score(X_test_other, y_test_other))
conf_mat_other = confusion_matrix(y_test_other, other_model_predicts)
print(conf_mat_other)

full_model_predicts_first_stage = sad_model.predict(X_test_full)
to_run_other = []
second_stage_labels = []
for pred in range(len(full_model_predicts_first_stage)):
    if full_model_predicts_first_stage[pred] == 0:
        # it's other
        to_run_other.append(X_test_full[pred])
        second_stage_labels.append(y_test_full[pred])

to_run_other = np.array(to_run_other)
to_run_other_y = np.array(second_stage_labels)

full_model_predicts_second_stage = other_model.predict(to_run_other)

hap, neu, ang = 0, 0, 0
for i in range(len(full_model_predicts_first_stage)):
    if full_model_predicts_first_stage[i] == 2:
        if y_test_full[i] == 1:
            hap+=1
        if y_test_full[i] == 3:
            neu+=1
        if y_test_full[i] == 5:
            ang+=1

# first stage
for i in range(len(y_test_full)):
    if y_test_full[i] == 1 or y_test_full[i] == 3 or y_test_full[i] == 5:
        y_test_full[i] = 0


true_full = np.concatenate([y_test_full, to_run_other_y])
predicts_full = np.concatenate([full_model_predicts_first_stage, full_model_predicts_second_stage])
#print('FIRST STAGE weighted f-score', f1_score(y_test_full, full_model_predicts_first_stage, average='weighted'))
#print('SECOND STAGE weighted f-score', f1_score(to_run_other_y, full_model_predicts_second_stage, average='weighted'))
print('Total weighted f-score', f1_score(true_full, predicts_full, average='weighted'))

conf_mat_fs = confusion_matrix(y_test_full, full_model_predicts_first_stage)
conf_mat_ss = confusion_matrix(to_run_other_y, full_model_predicts_second_stage)

print(conf_mat_fs)
print(conf_mat_ss)
print(hap, neu, ang)
print(classification_report(y_test_full, full_model_predicts_first_stage, digits=4))
print(classification_report(to_run_other_y, full_model_predicts_second_stage, digits=4))

"""
plt.figure()
import seaborn as sns
sns.heatmap(conf_mat, annot=True)
plt.show()
"""






# SAVING CODE
# Save to file in the current working directory

joblib_file = "./models/CREMA-D/4emo/DualSVM_SadStage.pkl"
joblib.dump(sad_model, joblib_file)
print('saved model', joblib_file)
joblib_file = "./models/CREMA-D/4emo/DualSVM_OtherStage.pkl"
joblib.dump(other_model, joblib_file)
print('saved model', joblib_file)

# SAVING TEST CASES

#np.save('./tests/4emo/all_test_x.npy', X_test)
#np.save('./tests/4emo/all_test_y.npy', y_test)
#np.save('./tests/4emo/all_train_x.npy', X_train)
#np.save('./tests/4emo/all_train_y.npy', y_train)
