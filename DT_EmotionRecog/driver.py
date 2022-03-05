#import tensorflow as tf
import tensorflow.keras.layers as nn
from tensorflow.keras import Model
import numpy as np
import pandas as pd
from tensor2tensor.layers import area_attention
import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import Dense, LSTM, Dropout
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt    


class MACNN(Model):
    def __init__(self, attention_heads=6, attention_size=32, out_size=6):
        super(MACNN, self).__init__()
        self.conv1a = nn.Conv2D(16, (10, 2), padding='same', data_format='channels_last',)# activation='relu')
        self.conv1b = nn.Conv2D(16, (2, 8), padding='same', data_format='channels_last',)# activation='relu')
        self.conv2 = nn.Conv2D(32, (3, 3), padding='same', data_format='channels_last', )#activation='relu')
        self.conv3 = nn.Conv2D(48, (3, 3), padding='same', data_format='channels_last',)# activation='relu')
        self.conv4 = nn.Conv2D(64, (3, 3), padding='same', data_format='channels_last',)# activation='relu')
        self.conv5 = nn.Conv2D(80, (3, 3), padding='same', data_format='channels_last', )#activation='relu')
        self.maxp = nn.MaxPool2D((2, 2))
        self.bn1a = nn.BatchNormalization(3)
        self.bn1b = nn.BatchNormalization(3)
        self.bn2 = nn.BatchNormalization(3)
        self.bn3 = nn.BatchNormalization(3)
        self.bn4 = nn.BatchNormalization(3)
        self.bn5 = nn.BatchNormalization(3)
        self.gap = nn.GlobalAveragePooling2D(data_format='channels_last')
        self.flatten = nn.Flatten(data_format='channels_last')
        self.fc = nn.Dense(out_size, activation='softmax')
        self.attention_query = []
        self.attention_key = []
        self.attention_value = []
        self.attention_heads = attention_heads
        self.attention_size = attention_size
        for i in range(self.attention_heads):
            self.attention_query.append(nn.Conv2D(self.attention_size, 1, padding='same', data_format='channels_last'))
            self.attention_key.append(nn.Conv2D(self.attention_size, 1, padding='same', data_format='channels_last'))
            self.attention_value.append(nn.Conv2D(self.attention_size, 1, padding='same', data_format='channels_last'))

    def call(self, *input):
        x = input[0]
        xa = self.conv1a(x)
        xa = self.bn1a(xa)
        xa=tf.nn.relu(xa)
        xb = self.conv1b(x)
        xb = self.bn1b(xb)
        xb = tf.nn.relu(xb)
        x = tf.concat([xa, xb], 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x=tf.nn.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = tf.nn.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = tf.nn.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = tf.nn.relu(x)

        attn = None
        for i in range(self.attention_heads):
            # Q = self.attention_query[i](x)
            # Q = tf.transpose(Q, perm=[0, 3, 1, 2])
            # K = self.attention_key[i](x)
            # K = tf.transpose(K, perm=[0, 3, 2, 1])
            # V = self.attention_value[i](x)
            # V = tf.transpose(V, perm=[0, 3, 1, 2])
            # attention = tf.nn.softmax(tf.matmul(Q, K))
            # attention = tf.matmul(attention, V)
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = tf.nn.softmax(tf.multiply(Q, K))
            attention = tf.multiply(attention, V)
            if (attn is None):
                attn = attention
            else:
                attn = tf.concat([attn, attention], 2)
        x = tf.transpose(attn, perm=[0, 2, 3, 1])
        x = tf.nn.relu(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class AACNN(Model):
    def __init__(self, height=3,width=3,out_size=6):
        super(AACNN, self).__init__()
        self.height=height
        self.width=width
        self.conv1 = nn.Conv2D(32, (3,3), padding='same', data_format='channels_last',)

        self.conv1a = nn.Conv2D(16, (10, 2), padding='same', data_format='channels_last',)# activation='relu')
        self.conv1b = nn.Conv2D(16, (2, 8), padding='same', data_format='channels_last',)# activation='relu')
        self.conv2 = nn.Conv2D(32, (3, 3), padding='same', data_format='channels_last', )#activation='relu')
        self.conv3 = nn.Conv2D(48, (3, 3), padding='same', data_format='channels_last',)# activation='relu')
        self.conv4 = nn.Conv2D(64, (3, 3), padding='same', data_format='channels_last',)# activation='relu')
        self.conv5 = nn.Conv2D(80, (3, 3), padding='same', data_format='channels_last', )#activation='relu')
        self.conv6 = nn.Conv2D(128, (3, 3), padding='same', data_format='channels_last', )#
        self.maxp = nn.MaxPool2D((2, 2))
        self.bn1a = nn.BatchNormalization(3)
        self.bn1b = nn.BatchNormalization(3)
        self.bn2 = nn.BatchNormalization(3)
        self.bn3 = nn.BatchNormalization(3)
        self.bn4 = nn.BatchNormalization(3)
        self.bn5 = nn.BatchNormalization(3)
        self.bn6 = nn.BatchNormalization(3)
        self.gap = nn.GlobalAveragePooling2D(data_format='channels_last')
        self.flatten = nn.Flatten(data_format='channels_last')
        self.fc = nn.Dense(out_size, activation='softmax')
        self.query = nn.Dense(20)
        self.key = nn.Dense(20)
        self.value = nn.Dense(20)
        
    def call(self, *input):
        x = input[0]
        xa = self.conv1a(x)
        xa = self.bn1a(xa)
        xa=tf.nn.relu(xa)
        xb = self.conv1b(x)
        xb = self.bn1b(xb)
        xb = tf.nn.relu(xb)
        x = tf.concat([xa, xb], 1)

        #x=input[0]
        #x=self.bn1a(x)
        #x=self.conv1(x)
        #x=tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x=tf.nn.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = tf.nn.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = tf.nn.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = tf.nn.relu(x)
        
        #x=self.conv6(x)
        #x=self.bn6(x)
        #x=tf.nn.relu(x)

        q=x
        k=x
        v=x
        bias=None
        dropout_rate=0.5

        x=area_attention.dot_product_area_attention(
            q, k, v, bias, dropout_rate, None,
            save_weights_to=None,
            dropout_broadcast_dims=None,
            max_area_width=self.width,
            max_area_height=self.height,
            area_key_mode='mean',
            area_value_mode='sum',
            training=True)

        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == '__main__': # run this with dt_emorecog
    test = np.random.random((4, 40, 40,1)).astype(np.float32)
    test = tf.convert_to_tensor(test)
    macnn = MACNN()
    #s=tf.compat.v1.Session()



    # open embeddings
    data = pd.read_pickle('./working_data/extracted_embeddings.pk')


    # Data filtering. Remove discards
    data_filtered = []
    for row, v in data.iterrows():
        # MAPPING {'disgust':0,'happy':1,'sad':2,'neutral':3,'fear':4,'angry':5}
        if v.labels != 'discard' :#and (v.labels == 1 or v.labels == 2 or v.labels==5 or v.labels==3):
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
    #X = np.asarray(data_filtered['dt'])
    #X_sub = np.asarray(data_filtered['subject'])
    #y = np.asarray(data_filtered["labels"])


    X = tf.keras.preprocessing.sequence.pad_sequences(X, dtype='float32')
    print(X.shape)

    # Subject disjointness between train/validation/test

    possible_subjects = list(range(1001, 1092))

    TRAIN_SIZE = 0.7 # Training sample of the data
    TEST_SIZE = 0.1 # Used for final testing of performance for extra rigor
    VALIDATION_SIZE = 0.2 # Used in the training of the model for validation data

    # Determine which subjects belong to which set
    X_train_subs, X_test_subs = train_test_split(possible_subjects, test_size=TEST_SIZE)
    X_train_subs, X_val_subs = train_test_split(X_train_subs, test_size=VALIDATION_SIZE)

    # There are now 3 sets of subjects X_train_subs | X_test_subs | X_val_subs


    X_train = []
    y_train = []
    X_test = []
    y_test = []
    X_validation = []
    y_validation = []

    for i in range(X.shape[0]):#7371):
        if data['subject'][i] in X_train_subs:
            X_train.append(X[i])
            y_train.append(y[i])
        elif data['subject'][i] in X_test_subs:
            X_test.append(X[i])
            y_test.append(y[i])
        elif data['subject'][i] in X_val_subs:
            X_validation.append(X[i])
            y_validation.append(y[i])

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    X_validation = np.asarray(X_validation)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    y_validation = np.asarray(y_validation)

    # SIZES OF TRAIN, VAL, TEST
    print(X_train.shape,y_train.shape,X_validation.shape,y_validation.shape,X_test.shape,y_test.shape)


    # TODO: FEBRUARY 23, 2022 - Write the model code below for a CNN/LSTM/Attention program, LSTM, SVM.
    # Add a switch to go between those models. Refer ser-lstm jupyter notebook in 'Buid keras model' for previous stuff
    ''' LSTM ONLY '''
    def build_model(input_shape):
        model = tf.keras.Sequential()
        
        cnn = Sequential()

        # Layer of LSTM
        model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
        model.add(LSTM(64))
        
        model.add(Dense(32, activation='relu'))
        
        # Dropout layer to combat overfitting
        model.add(Dropout(0.3))

        # Softmax function to convert to probabilities for six emotional classes
        model.add(Dense(6, activation='softmax'))

        return model

    # create network
    input_shape = (None, 256)
    model = build_model(input_shape)

    # compile model
    optimiser = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimiser,
                    loss='sparse_categorical_crossentropy', # TODO: Try another loss function???
                    metrics=['accuracy'])

    model.summary()
    

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, min_delta=0.01)
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=10, epochs=100, callbacks=[callback])
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print("Test Accuracy: ", test_acc)
    
    test_val_loss, test_val_acc = model.evaluate(X_validation, y_validation, verbose=0)
    print("Test Val Accuracy: ", test_val_acc)
    
    
    # SEABORN CONFUSION MATRIX
 
    y_predicted = model.predict(X_test)
    matrix = confusion_matrix(y_test, y_predicted.argmax(axis=1)) 

    sns.heatmap(matrix, annot=True, fmt='g') #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    #ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    #ax.set_title('LSTM-Based Confusion Matrix'); 
    #ax.xaxis.set_ticklabels(['disgust','happy','sad','neutral','fear','angry']); ax.yaxis.set_ticklabels(['disgust','happy','sad','neutral','fear','angry']);
    plt.savefig('confusion.png')
