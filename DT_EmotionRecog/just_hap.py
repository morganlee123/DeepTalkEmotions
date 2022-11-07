# Author: Morgan Sandler (sandle20@msu.edu)
#
# Tests a binary SVM classifier for Happy vs others
#
# Experiment 3 - Auxiliary
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

# open embeddings
data = pd.read_pickle('./working_data/extracted_embeddings.pk')

# Data filtering. Remove discards
data_filtered = []
for row, v in data.iterrows():
    # MAPPING {'disgust':0,'happy':1,'sad':2,'neutral':3,'fear':4,'angry':5}
    if v.labels != 'discard' and (v.labels == 1 or v.labels == 2 or v.labels==5 or v.labels==3):

        if (v.labels == 2 or v.labels == 5 or v.labels == 3):
            d = v
            d.labels = 0
            data_filtered.append(d)
        else:
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


X_prep = []
for e in X:
    X_prep.append(np.mean(e,axis=0))
X = np.array(X_prep)


print(X.shape)

# Subject disjointness between train/validation/test

possible_subjects = list(range(1001, 1092))

TRAIN_SIZE = 0.7 # Training sample of the data
TEST_SIZE = 0.3 # Used for final testing of performance

# Determine which subjects belong to which set
RANDOM_STATE = 600
X_train_subs, X_test_subs = train_test_split(possible_subjects, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# There are now 2 sets of subjects X_train_subs | X_test_subs
# AKA INDEPENDENCE WOO

X_train = []
y_train = []
X_test = []
y_test = []

for i in range(X.shape[0]):#7371):
    if data['subject'][i] in X_train_subs:
        X_train.append(X[i])
        y_train.append(y[i])
    elif data['subject'][i] in X_test_subs:
        X_test.append(X[i])
        y_test.append(y[i])


X_train = np.asarray(X_train)
X_test = np.asarray(X_test)

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)


'''
happy, sad, neutral, angry = 0, 0, 0, 0
for i in y_test:
    if i==1:
        happy+=1
    elif i==2:
        sad+=1
    elif i==3:
        neutral+=1
    elif i==5:
        angry+=1

print(sad, happy, neutral, angry)
'''

from imblearn.datasets import make_imbalance
X_train, y_train = make_imbalance(X_train, y_train, sampling_strategy={0: 720, 1: 720},random_state=RANDOM_STATE)
X_test, y_test = make_imbalance(X_test, y_test, sampling_strategy={0: 320, 1: 320},random_state=RANDOM_STATE)
print('Balanced emotion classes')

# SIZES OF TRAIN, VAL, TEST
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)


# LR Train
#model = LogisticRegression(random_state=10, multi_class='multinomial', penalty='l2', solver='newton-cg').fit(X_train, y_train)
#model = LogisticRegression(random_state=5, multi_class='multinomial', penalty='l2', solver='newton-cg').fit(X_train_maxpooled, y_train)

# SVM Train 
model = SVC(random_state=99999, kernel='rbf', gamma=0.1, C=1000, probability=True).fit(X_train, y_train)
#model = SVC(random_state=10, kernel='poly', degree=3, probability=True).fit(X_train, y_train)
#model = SVC(random_state=10, kernel='linear').fit(X_train, y_train)

#from sklearn.neural_network import MLPClassifier
#model = MLPClassifier(random_state=RANDOM_STATE, max_iter=300, hidden_layer_sizes=(100,100, 100,)).fit(X_train, y_train)

#from sklearn.model_selection import GridSearchCV
 
# defining parameter range
#param_grid = {'C': [0.1, 1, 10, 100, 1000],
#              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#              'kernel': ['rbf']}
 
#grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
 
# fitting the model for grid search
#grid.fit(X_train, y_train)

# LR Test
#print(y_test, model.predict_proba(X_test))
#print(model.predict(reshaped_X_test))
#print(model.score(reshaped_X_test, y_test))
#print(confusion_matrix(y_test, model.predict(reshaped_X_test)))

from sklearn.metrics import f1_score
print('weighted f-score', f1_score(y_test, model.predict(X_test), average='weighted'))
print('Accuracy', model.score(X_test, y_test))
conf_mat = confusion_matrix(y_test, model.predict(X_test))
print(conf_mat)

plt.figure()
import seaborn as sns
sns.heatmap(conf_mat, annot=True)
plt.show()


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

#joblib_file = "./models/CREMA-D/4emo/SVM_AllEmbeddings.pkl"
#joblib.dump(grid, joblib_file)
#print('saved model', joblib_file)

# SAVING TEST CASES

#np.save('./tests/4emo/all_test_x.npy', X_test)
#np.save('./tests/4emo/all_test_y.npy', y_test)
#np.save('./tests/4emo/all_train_x.npy', X_train)
#np.save('./tests/4emo/all_train_y.npy', y_train)
