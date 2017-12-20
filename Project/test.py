import sys
from pandas import read_csv
import numpy as np
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam

if len(sys.argv) < 3:
    print('Usage: {0} <Test_file> <Model_name> <result_name>'.format(sys.argv[0]), file=sys.stderr)
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
indices = x_test.axes[1]

for index in indices:
    if index not in NO_STD:
        x_test[index] = standardize(x_test[index])

x_test['AGE_2'] = standardize(x_test['AGE'] ** 2)
x_test['PAY_5_2'] = standardize(x_test['PAY_5'] ** 2)
x_test['PAY_6_2'] = standardize(x_test['PAY_6'] ** 2)
x_test['BILL_AMT5_2'] = standardize(x_test['BILL_AMT5'] ** 2)
x_test['BILL_AMT6_2'] = standardize(x_test['BILL_AMT6'] ** 2)
x_test['PAY_AMT5_2'] = standardize(x_test['PAY_AMT5'] ** 2)
x_test['PAY_AMT6_2'] = standardize(x_test['PAY_AMT6'] ** 2)

'''
for index in indices:
    x_test[index + '_2'] = standardize(x_test[index] ** 2)

for index in indices:
    x_test[index + '_3'] = standardize(x_test[index] ** 3)

'''

x_test = np.array(x_test)

y_test = model.predict(x_test)
y_test = np.reshape(y_test, len(y_test))

rank = (-y_test).argsort() + 1
for r in rank:
    print(r, file=result)

result.close()
