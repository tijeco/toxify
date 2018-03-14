import pandas as pd
import io
import math
import numpy as np
import sys

def getOptionValue(option):
    optionPos = [i for i, j in enumerate(sys.argv) if j == option][0]
    optionValue = sys.argv[optionPos + 1]
    return optionValue


train_data = "train_dataing.csv"
test_data = "test_data.csv"

if "-train" in sys.argv:
    train_data = getOptionValue("-train")
else:
    print("Supply training data using -train")
    sys.exit()

if "-test" in sys.argv:
    test_data = getOptionValue("-test")
else:
    print("Supply testing data using -test")
    sys.exit()

training_data = pd.read_csv(train_data, skiprows=[0], header=None)
training_headers = pd.read_csv(train_data, skiprows=[0], header=None)

test_data = pd.read_csv(test_data, skiprows=[0], header=None)
test_headers = pd.read_csv(test_data, skiprows=[0], header=None)
num_cols = test_data.shape[1]-1

num_classes =  len(training_data[num_cols].unique())
mean_array = np.zeros((num_classes,num_cols))
var_array = np.zeros((num_classes,num_cols))


for i in range(num_classes):
    subset = training_data.loc[training_data[num_cols] == i]
    mean_array[i] = subset.iloc[:,0:num_cols].mean()
    var_array[i] = subset.iloc[:,0:num_cols].var()




for index, row in test_data.iterrows():
    blank_array = np.zeros((num_classes,num_cols))
    test_value = row[:num_cols]
    evidence = 0
    for i in range(num_classes):
        exponent = -(test_value-mean_array[i])**2/(2*var_array[i])
        base = 1.0/((2.0*var_array[i]*math.pi)**0.5)
        blank_array[i] = base*math.e**exponent
        evidence += np.prod(blank_array[i]) * 0.33
    line2write = ""
    # print("Actual value:",row[num_cols])
    for i in range(num_classes):
        # print(i, (np.prod(blank_array[i])*0.33)/evidence)
        line2write+=str((np.prod(blank_array[i])*0.33)/evidence)+","
        # if i == row[num_cols]:
        #     print(i,(np.prod(blank_array[i])*0.33)/evidence)
    line2write+=str(row[num_cols])
    print(line2write)
