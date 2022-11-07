# Author: Morgan Sandler (sandle20@msu.edu)
#
# Runs the experiment 2 from the paper. This program
# implements the single SVM classifier. At the end of the program
# there is a conf matrix generated
#
# Experiment 2 - Step 3
#

from re import I
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from sklearn.model_selection import train_test_split
import joblib

# TODO: FIX THE TOTAL DATASET OF THIS SO ITS NOT JUST TEST EMBEDDINGS BUT TRAIN EMBEDDINGS TOO. MAKE SURE IT ALL BALANCED

# open embeddings
train_data = pd.read_pickle('./working_data/msp_train_extracted_embeddings.pk')
test_data = pd.read_pickle('./working_data/msp_test_extracted_embeddings.pk')

#print(data.groupby('labels').count())

# Data filtering. Remove discards
train_data_filtered = []
#print(train_data.groupby('labels').count())
#print(test_data.groupby('labels').count())
for row, v in train_data.iterrows():
    # MAPPING {'disgust':0,'happy':1,'sad':2,'neutral':3,'fear':4,'angry':5}
    if v.labels != 'discard' and (v.labels == 1 or v.labels == 2 or v.labels==5 or v.labels==3):
        train_data_filtered.append(v)
print(len(train_data_filtered)/len(train_data), '% train valid data ---', len(train_data_filtered), 'samples')

test_data_filtered = []
for row, v in test_data.iterrows():
    # MAPPING {'disgust':0,'happy':1,'sad':2,'neutral':3,'fear':4,'angry':5}
    if v.labels != 'discard' and (v.labels == 1 or v.labels == 2 or v.labels==5 or v.labels==3):
        test_data_filtered.append(v)
print(len(test_data_filtered)/len(test_data), '% train valid data ---', len(test_data_filtered), 'samples')


train_embeddings = []
train_subjects = []
train_labels = []
for em in train_data_filtered:
    train_embeddings.append(em['dt'])
    train_subjects.append(em['subject'])
    train_labels.append(em['labels'])

# TODO: Make these samples balanced for the 4 classes
Train_X = np.array(train_embeddings)
Train_X_sub = np.array(train_subjects)
Train_y = np.array(train_labels)

test_embeddings = []
test_subjects = []
test_labels = []
for em in test_data_filtered:
    test_embeddings.append(em['dt'])
    test_subjects.append(em['subject'])
    test_labels.append(em['labels'])

# TODO: Make these samples balanced for the 4 classes
Test_X = np.array(test_embeddings)
Test_X_sub = np.array(test_subjects)
Test_y = np.array(test_labels)


X_train_prep = []
for e in Train_X:
    X_train_prep.append(np.mean(e,axis=0))
Train_X = np.array(X_train_prep)

X_test_prep = []
for e in Test_X:
    X_test_prep.append(np.mean(e,axis=0))
Test_X = np.array(X_test_prep)


print(Test_X.shape, Test_y.shape, Train_X.shape, Train_y.shape)

# TODO: fix this balance
from imblearn.datasets import make_imbalance
X_train, y_train = make_imbalance(Train_X, Train_y, sampling_strategy={1: 1600, 2: 1600, 3: 1600, 5: 1600},random_state=42)
X_test, y_test = make_imbalance(Test_X, Test_y, sampling_strategy={1: 450, 2: 450, 3: 450, 5: 450},random_state=42)
#print('Balanced emotion classes')

# SIZES OF TRAIN, VAL, TEST
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

# LR Train
#model = LogisticRegression(random_state=10, multi_class='multinomial', penalty='l2', solver='newton-cg').fit(X_train, y_train)
#model = LogisticRegression(random_state=5, multi_class='multinomial', penalty='l2', solver='newton-cg').fit(X_train_maxpooled, y_train)

# SVM Train
#from sklearn.model_selection import GridSearchCV
 
# defining parameter range
#param_grid = {'C': [0.1, 1, 10, 100, 1000],
#              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#              'kernel': ['rbf']}
 
#grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
 
# fitting the model for grid search
#grid.fit(X_train, y_train) # C=10, 1
model = SVC(random_state=99999, kernel='rbf', gamma=.1, C=100, probability=True).fit(X_train, y_train)
#model = SVC(random_state=10, kernel='poly', degree=3, probability=True).fit(X_train, y_train)
#model = SVC(random_state=10, kernel='linear').fit(X_train, y_train)

#from sklearn.neural_network import MLPClassifier
#model = MLPClassifier(random_state=10, max_iter=300).fit(X_train, y_train)


# LR Test
#print(y_test, model.predict_proba(X_test))
#print(model.predict(reshaped_X_test))
#print(model.score(reshaped_X_test, y_test))
#print(confusion_matrix(y_test, model.predict(reshaped_X_test)))

from sklearn.metrics import f1_score
print('weighted f-score', f1_score(y_test, model.predict(X_test), average='weighted'))
print('Accuracy', model.score(X_test, y_test))
print(confusion_matrix(y_test, model.predict(X_test)))


# Poly SVM Test
#print(poly.predict_proba(reshaped_X_test))
#print(poly.predict(reshaped_X_test))
#print(poly.score(reshaped_X_test, y_test))
#print(confusion_matrix(y_test, poly.predict(reshaped_X_test)))
# RBF SVM Test
#print(rbf.predict_proba(reshaped_X_test))
#print(rbf.predict(reshaped_X_test))
#print(rbf.score(reshaped_X_test, y_test))
#print(confusion_matrix(y_test, rbf.predict(reshaped_X_test)))

# SAVING CODE
# Save to file in the current working directory

joblib_file = "./models/MSP/4emo/SVM_AllEmbeddings.pkl"
joblib.dump(model, joblib_file)
print('saved model', joblib_file)

# SAVING TEST CASES

#np.save('./tests/4emo/all_test_x_msp.npy', X_test)
#np.save('./tests/4emo/all_test_y_msp.npy', y_test)
#np.save('./tests/4emo/all_train_x_msp.npy', X_train)
#np.save('./tests/4emo/all_train_y_msp.npy', y_train)
