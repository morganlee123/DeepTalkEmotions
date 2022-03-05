import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from sklearn.model_selection import train_test_split
import joblib

def balanced_subsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys

# open embeddings
data = pd.read_pickle('./working_data/extracted_embeddings.pk')

# Data filtering. Remove discards
data_filtered = []
for row, v in data.iterrows():
    # MAPPING {'disgust':0,'happy':1,'sad':2,'neutral':3,'fear':4,'angry':5}
    if v.labels != 'discard' and (v.labels == 1 or v.labels == 2 or v.labels==5 or v.labels==3):
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
X = np.array(embeddings)
X_sub = np.array(subjects)
y = np.array(labels)

X_prep = []
for e in X:
    X_prep.append(np.mean(e,axis=0))
X = np.array(X_prep)

# Old preprocessing step
#X = tf.keras.preprocessing.sequence.pad_sequences(X, dtype='float32')
print(X.shape)

# Subject disjointness between train/validation/test

possible_subjects = list(range(1001, 1092))

TRAIN_SIZE = 0.7 # Training sample of the data
TEST_SIZE = 0.3 # Used for final testing of performance

# Determine which subjects belong to which set
X_train_subs, X_test_subs = train_test_split(possible_subjects, test_size=TEST_SIZE)

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

# SIZES OF TRAIN, VAL, TEST
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

#nsamples, nx, ny = X_train.shape
#reshaped_X_train = X_train.reshape((nsamples,nx*ny))
#print(reshaped_X_train.shape)

# TODO: INTERIM MAX POOLING TEST
#from skimage.measure import block_reduce
#X_train_maxpooled = []
#for i in reshaped_X_train:
#    curr = block_reduce(i, block_size=(4,), func=np.mean)
#    X_train_maxpooled.append(curr)
#X_train_maxpooled = np.array(X_train_maxpooled)
#print(X_train_maxpooled.shape)

# LR Train
#model = LogisticRegression(random_state=10, multi_class='multinomial', penalty='l2', solver='newton-cg').fit(X_train, y_train)
#model = LogisticRegression(random_state=5, multi_class='multinomial', penalty='l2', solver='newton-cg').fit(X_train_maxpooled, y_train)

# SVM Train
model = SVC(random_state=10, kernel='rbf', gamma=0.8, C=1, probability=True).fit(X_train, y_train)
#model = SVC(random_state=10, kernel='poly', degree=3, probability=True).fit(X_train, y_train)
#model = SVC(random_state=10, kernel='linear').fit(X_train, y_train)

#nsamples, nx, ny = X_test.shape
#reshaped_X_test = X_test.reshape((nsamples,nx*ny))

# TODO: INTERIM MAX POOLING TEST
#from skimage.measure import block_reduce
#X_test_maxpooled = []
#for i in reshaped_X_test:
#    curr = block_reduce(i, block_size=(4,), func=np.mean)
#    X_test_maxpooled.append(curr)


# LR Test
#print(y_test, model.predict_proba(X_test))
#print(model.predict(reshaped_X_test))
#print(model.score(reshaped_X_test, y_test))
#print(confusion_matrix(y_test, model.predict(reshaped_X_test)))

from sklearn.metrics import f1_score
print('weighted f-score', f1_score(y_test, model.predict(X_test), average='weighted'))
print(model.score(X_test, y_test))
print(confusion_matrix(y_test, model.predict(X_test)))

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

joblib_file = "./models/CREMA-D/4emo/LR_300_4emo.pkl"
#joblib.dump(model, joblib_file)
print('saved model')

# SAVING TEST CASES

#np.save('./tests/4emo/300_test_x.npy', reshaped_X_test)
#np.save('./tests/4emo/300_test_y.npy', y_test)
#np.save('./tests/4emo/300_train_x.npy', reshaped_X_train)
#np.save('./tests/4emo/300_train_y.npy', y_train)
