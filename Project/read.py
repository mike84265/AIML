import csv
rawdata = []
with open('Train.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        rawdata.append(row)

data = []
for row in rawdata[1:]:
    obj = dict()
    for index in range(len(row)):
        obj[rawdata[0][index]] = int(row[index])
    data.append(obj)
