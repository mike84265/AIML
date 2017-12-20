# AIML Random Forest 

import sys
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt

###################### Define Function ########################

def standardize(x):
    s = (x - np.mean(x)) / np.std(x)
    return s

###################### Define Function ########################

###################### Define Function ########################

x_train = pd.read_csv(sys.argv[1])
y_train = x_train.pop('Y')
x_test  = pd.read_csv(sys.argv[2])
result  = open (sys.argv[3], 'w')
print ('Rank_ID', file = result)

######################## Format Data #########################

x_train = x_train.astype(float)

######################## Standardize ########################

x_train = x_train.drop(['Train_ID'], axis = 1)
x_train['LIMIT_BAL'] = standardize(x_train['LIMIT_BAL'])
x_train['AGE'] = standardize(x_train['AGE'])
#x_train['SEX'] = standardize(x_train['SEX'])
#x_train['EDUCATION'] = standardize(x_train['EDUCATION'])
#x_train['MARRIAGE'] = standardize(x_train['MARRIAGE'])
x_train['PAY_1'] = standardize(x_train['PAY_1'])
x_train['PAY_2'] = standardize(x_train['PAY_2'])
x_train['PAY_3'] = standardize(x_train['PAY_3'])
x_train['PAY_4'] = standardize(x_train['PAY_4'])
x_train['PAY_5'] = standardize(x_train['PAY_5'])
x_train['PAY_6'] = standardize(x_train['PAY_6'])
x_train['BILL_AMT1'] = standardize(x_train['BILL_AMT1'])
x_train['BILL_AMT2'] = standardize(x_train['BILL_AMT2'])
x_train['BILL_AMT3'] = standardize(x_train['BILL_AMT3'])
x_train['BILL_AMT4'] = standardize(x_train['BILL_AMT4'])
x_train['BILL_AMT5'] = standardize(x_train['BILL_AMT5'])
x_train['BILL_AMT6'] = standardize(x_train['BILL_AMT6'])
x_train['PAY_AMT1'] = standardize(x_train['PAY_AMT1'])
x_train['PAY_AMT2'] = standardize(x_train['PAY_AMT2'])
x_train['PAY_AMT3'] = standardize(x_train['PAY_AMT3'])
x_train['PAY_AMT4'] = standardize(x_train['PAY_AMT4'])
x_train['PAY_AMT5'] = standardize(x_train['PAY_AMT5'])
x_train['PAY_AMT6'] = standardize(x_train['PAY_AMT6'])

x_train['AGE_2ND'] = standardize(x_train['AGE'] ** 2)
x_train['PAY_5_2ND'] = standardize(x_train['PAY_5'] ** 2)
x_train['PAY_6_2ND'] = standardize(x_train['PAY_6'] ** 2)
x_train['BILL_AMT5_2ND'] = standardize(x_train['BILL_AMT5'] ** 2)
x_train['BILL_AMT6_2ND'] = standardize(x_train['BILL_AMT6'] ** 2)
x_train['PAY_AMT5_2ND'] = standardize(x_train['PAY_AMT5'] ** 2)
x_train['PAY_AMT6_2ND'] = standardize(x_train['PAY_AMT6'] ** 2)

x_train = np.array(x_train)
######################## Standardize ########################

y_train = y_train.astype(int)
y_train = np.array(y_train)
x_test = x_test.astype(float)

######################## Standardize ########################

x_test = x_test.drop(['Test_ID'], axis = 1)
x_test['LIMIT_BAL'] = standardize(x_test['LIMIT_BAL'])
x_test['AGE'] = standardize(x_test['AGE'])
#x_test['SEX'] = standardize(x_test['SEX'])
#x_test['EDUCATION'] = standardize(x_test['EDUCATION'])
#x_test['MARRIAGE'] = standardize(x_test['MARRIAGE'])
x_test['PAY_1'] = standardize(x_test['PAY_1'])
x_test['PAY_2'] = standardize(x_test['PAY_2'])
x_test['PAY_3'] = standardize(x_test['PAY_3'])
x_test['PAY_4'] = standardize(x_test['PAY_4'])
x_test['PAY_5'] = standardize(x_test['PAY_5'])
x_test['PAY_6'] = standardize(x_test['PAY_6'])
x_test['BILL_AMT1'] = standardize(x_test['BILL_AMT1'])
x_test['BILL_AMT2'] = standardize(x_test['BILL_AMT2'])
x_test['BILL_AMT3'] = standardize(x_test['BILL_AMT3'])
x_test['BILL_AMT4'] = standardize(x_test['BILL_AMT4'])
x_test['BILL_AMT5'] = standardize(x_test['BILL_AMT5'])
x_test['BILL_AMT6'] = standardize(x_test['BILL_AMT6'])
x_test['PAY_AMT1'] = standardize(x_test['PAY_AMT1'])
x_test['PAY_AMT2'] = standardize(x_test['PAY_AMT2'])
x_test['PAY_AMT3'] = standardize(x_test['PAY_AMT3'])
x_test['PAY_AMT4'] = standardize(x_test['PAY_AMT4'])
x_test['PAY_AMT5'] = standardize(x_test['PAY_AMT5'])
x_test['PAY_AMT6'] = standardize(x_test['PAY_AMT6'])

x_test['AGE_2ND'] = standardize(x_test['AGE'] ** 2)
x_test['PAY_5_2ND'] = standardize(x_test['PAY_5'] ** 2)
x_test['PAY_6_2ND'] = standardize(x_test['PAY_6'] ** 2)
x_test['BILL_AMT5_2ND'] = standardize(x_test['BILL_AMT5'] ** 2)
x_test['BILL_AMT6_2ND'] = standardize(x_test['BILL_AMT6'] ** 2)
x_test['PAY_AMT5_2ND'] = standardize(x_test['PAY_AMT5'] ** 2)
x_test['PAY_AMT6_2ND'] = standardize(x_test['PAY_AMT6'] ** 2)

x_test = np.array(x_test)
######################## Standardize ########################
if (sys.argv[4] == 'train'):
    model = Sequential()
    model.add(Dense(128, input_shape = (30,), activation='relu', kernel_initializer='truncated_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu', kernel_initializer='truncated_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu', kernel_initializer='truncated_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='truncated_normal'))

    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.summary()
    model.fit(x_train, y_train, batch_size = 64, epochs = 20, validation_split = 0.2)

    model.save('model.h5')
    
elif (sys.argv[4] == 'test'):
    model = load_model(sys.argv[5])
    y_test = model.predict(x_test)

    rank = dict(zip(range(len(y_test)), y_test))
    rank = sorted(rank.items(), key=lambda d:d[1], reverse=True)
    for i in range(len(rank)):
        print(rank[i][0]+1, file = result)
    result.close()


