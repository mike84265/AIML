import sys
from pandas import read_csv

x_train = read_csv(sys.argv[1])
y_train = x_train.pop('Y')
