import sys
from pandas import read_csv
import numpy as np
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam

if len(sys.argv) < 4:
    print('Usage: {0} <Test_file> <Model_name> <result_name> <ID_start>'.format(sys.argv[0]), file=sys.stderr)
    exit(1)

x_test = read_csv(sys.argv[1])
model = load_model(sys.argv[2])
result = open(sys.argv[3], 'w')
print('Rank_ID', file = result)

def standardize(x):
    return (x - np.mean(x)) / np.std(x)

x_test.pop('Test_ID')
x_test.astype(float)

NO_STD = ['SEX', 'EDUCATION', 'MARRIAGE']
SQUARE = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 
    'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
indices = x_test.axes[1]

for index in indices:
    if index not in NO_STD:
        x_test[index] = standardize(x_test[index])
for index in SQUARE:
    x_test[index + '_2'] = standardize(x_test[index] ** 2)

'''
x_test['AGE_2'] = standardize(x_test['AGE'] ** 2)
x_test['PAY_5_2'] = standardize(x_test['PAY_5'] ** 2)
x_test['PAY_6_2'] = standardize(x_test['PAY_6'] ** 2)
x_test['BILL_AMT4_2'] = standardize(x_test['BILL_AMT4'] ** 2)
x_test['BILL_AMT5_2'] = standardize(x_test['BILL_AMT5'] ** 2)
x_test['BILL_AMT6_2'] = standardize(x_test['BILL_AMT6'] ** 2)
x_test['PAY_AMT4_2'] = standardize(x_test['PAY_AMT4'] ** 2)
x_test['PAY_AMT5_2'] = standardize(x_test['PAY_AMT5'] ** 2)
x_test['PAY_AMT6_2'] = standardize(x_test['PAY_AMT6'] ** 2)

for index in indices:
    x_test[index + '_2'] = standardize(x_test[index] ** 2)

for index in indices:
    x_test[index + '_3'] = standardize(x_test[index] ** 3)

'''

x_test = np.array(x_test)

y_test = model.predict(x_test)
y_test = np.reshape(y_test, len(y_test))

rank = (-y_test).argsort() + int(sys.argv[4])
for r in rank:
    print(r, file=result)

result.close()
