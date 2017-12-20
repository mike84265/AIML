import sys
from pandas import read_csv
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization

if len(sys.argv) < 3:
    print('Usage: {0} <Train_file> <Model_name>'.format(sys.argv[0]))
    exit(1)

x_train = read_csv(sys.argv[1])
y_train = x_train.pop('Y')

def standardize(x):
    return (x - np.mean(x)) / np.std(x)

x_train.pop('Train_ID')
x_train.astype(float)
NO_STD = ['SEX', 'EDUCATION', 'MARRIAGE']
indices = x_train.axes[1]

for index in indices:
    if index not in NO_STD:
        x_train[index] = standardize(x_train[index])

x_train['AGE_2'] = standardize(x_train['AGE'] ** 2)
x_train['PAY_5_2'] = standardize(x_train['PAY_5'] ** 2)
x_train['PAY_6_2'] = standardize(x_train['PAY_6'] ** 2)
x_train['BILL_AMT5_2'] = standardize(x_train['BILL_AMT5'] ** 2)
x_train['BILL_AMT6_2'] = standardize(x_train['BILL_AMT6'] ** 2)
x_train['PAY_AMT5_2'] = standardize(x_train['PAY_AMT5'] ** 2)
x_train['PAY_AMT6_2'] = standardize(x_train['PAY_AMT6'] ** 2)

'''
for index in indices:
    x_train[index + '_2'] = standardize(x_train[index] ** 2)

for index in indices:
    x_train[index + '_3'] = standardize(x_train[index] ** 3)
'''

x_train = np.array(x_train)
y_train = np.array(y_train.astype(int))

model = Sequential()
model.add(Dense(128, input_shape = (np.size(x_train,1),), activation='relu', kernel_initializer='truncated_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu', kernel_initializer='truncated_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu', kernel_initializer='truncated_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu', kernel_initializer='truncated_normal'))
# model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid', kernel_initializer='truncated_normal'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# model.compile(loss = 'mae', optimizer = 'adam', metrics = ['accuracy'])
model.summary()
model.fit(x_train, y_train, batch_size = 64, epochs = 20, validation_split = 0.2)
model.save(sys.argv[2])
