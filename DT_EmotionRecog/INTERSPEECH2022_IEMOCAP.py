# Author: Morgan Sandler (sandle20@msu.edu)
#
# Runs the Experiment 1 from the paper. Loads the extracted embeddings,
# splits the test/train sets, and then generates the single SVM model.
# At the end of the program there are confusion matrix generators and
# then f-score reports are printed
# 
# Experiment 1 - Step 3
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

# open embeddings
data = pd.read_pickle('./working_data/iemocap_extracted_embeddings.pk')
data_test = pd.read_pickle('./working_data/iemocap_extracted_embeddings_test.pk')

# Data filtering. Remove discards
data_filtered = []
for row, v in data.iterrows():
    # MAPPING {'disgust':0,'happy':1,'sad':2,'neutral':3,'fear':4,'angry':5}
    if v.labels != 'discard' and (v.labels == 1 or v.labels == 3 or v.labels==0 or v.labels==7):
        data_filtered.append(v)
print(len(data_filtered)/len(data), '% valid data ---', len(data_filtered), 'samples')

embeddings = []
subjects = []
labels = []
for em in data_filtered:
    embeddings.append(em['dt'])
    subjects.append(em['subject'])
    labels.append(em['labels'])

# TODO: Make these samples balanced for the 4 classes
X_train = np.array(embeddings)
X_sub_train = np.array(subjects)
y_train = np.array(labels)

print(np.unique(y_train))
X_prep = []
for e in X_train:
    X_prep.append(np.mean(e,axis=0))
X_train = np.array(X_prep)


print(X_train.shape)




# Data filtering. Remove discards
data_filtered = []
for row, v in data_test.iterrows():
    # MAPPING {'disgust':0,'happy':1,'sad':2,'neutral':3,'fear':4,'angry':5}
    if v.labels != 'discard' and (v.labels == 1 or v.labels == 3 or v.labels==0 or v.labels==7):
        data_filtered.append(v)
print(len(data_filtered)/len(data), '% valid data ---', len(data_filtered), 'samples')

embeddings = []
subjects = []
labels = []
for em in data_filtered:
    embeddings.append(em['dt'])
    subjects.append(em['subject'])
    labels.append(em['labels'])

# TODO: Make these samples balanced for the 4 classes
X_test = np.array(embeddings)
X_sub_test = np.array(subjects)
y_test = np.array(labels)

print(np.unique(y_test))
X_prep = []
for e in X_test:
    X_prep.append(np.mean(e,axis=0))
X_test = np.array(X_prep)


print(X_test.shape)



RANDOM_STATE = 500
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=RANDOM_STATE)

# SIZES OF TRAIN, VAL, TEST
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
from collections import Counter
print(Counter(y_train))
print(Counter(y_test))


from imblearn.datasets import make_imbalance
X_train, y_train = make_imbalance(X_train, y_train, sampling_strategy={0: 455, 1: 455, 3: 455, 7: 455},random_state=RANDOM_STATE)
X_test, y_test = make_imbalance(X_test, y_test, sampling_strategy={0: 133, 1: 133, 3: 133, 7: 133},random_state=RANDOM_STATE)
print('Balanced emotion classes')
print(Counter(y_train))
print(Counter(y_test))

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
plt.savefig('./iemocap_svm.png')

#X_test_maxpooled = np.array(X_test_maxpooled)
#print(X_test_maxpooled.shape)
#print(model.score(X_test_maxpooled, y_test))
#print(confusion_matrix(y_test, model.predict(X_test_maxpooled)))

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

#joblib_file = "./models/IEMOCAP/4emo/SVM_AllEmbeddings.pkl"
#joblib.dump(grid, joblib_file)
#print('saved model', joblib_file)

# SAVING TEST CASES

#np.save('./tests/4emo/all_test_x.npy', X_test)
#np.save('./tests/4emo/all_test_y.npy', y_test)
#np.save('./tests/4emo/all_train_x.npy', X_train)
#np.save('./tests/4emo/all_train_y.npy', y_train)
