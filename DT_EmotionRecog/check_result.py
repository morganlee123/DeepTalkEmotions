# Author: Morgan Sandler (sandle20@msu.edu)
#
# Can use this file to check stored test results if needed
#
# Auxiliary File
#

import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import tensorflow as tf


# Load from file
model = joblib.load('./models/CREMA-D/4emo/LR_300_4emo.pkl')

# Load test data
X_train = np.load('./tests/4emo/300_1train_x.npy')
y_train = np.load('./tests/4emo/300_1train_y.npy')
X_test = np.load('./tests/4emo/300_test_x.npy')
y_test = np.load('./tests/4emo/300_test_y.npy')


# Comparison
print('Train score', model.score(X_train, y_train))
print('Test score', model.score(X_test, y_test))
print(confusion_matrix(y_test, model.predict(X_test)))
