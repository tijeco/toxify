import pandas as pd
import io
import math
import numpy as np


iris_train = "iris_training.csv"
iris_test = "iris_test.csv"


print(pd.read_csv(iris_test, skiprows=[0], header=None))
print(pd.read_csv(iris_test, nrows=1, header=None))

training_data = pd.read_csv(iris_train, skiprows=[0], header=None)
training_headers = pd.read_csv(iris_train, skiprows=[0], header=None)

test_data = pd.read_csv(iris_test, skiprows=[0], header=None)
test_headers = pd.read_csv(iris_test, skiprows=[0], header=None)
num_cols = test_data.shape[1]-1
print(training_data)

print(len(training_data[num_cols].unique()))
num_classes =  len(training_data[num_cols].unique())
mean_array = np.zeros((num_classes,num_cols))
var_array = np.zeros((num_classes,num_cols))
print(mean_array)

for i in range(num_classes):
    print(i)
    subset = training_data.loc[training_data[num_cols] == i]
    print(subset)
    print(subset.iloc[:,0:num_cols].mean())
    mean_array[i] = subset.iloc[:,0:num_cols].mean()
    var_array[i] = subset.iloc[:,0:num_cols].var()

print(mean_array,"\n\n\n")
print(var_array)


print("#################")

for index, row in test_data.iterrows():
    blank_array = np.zeros((num_classes,num_cols))
    # print(blank_array)
    test_value = row[:num_cols]
    # print(test_value)
    evidence = 0
    for i in range(num_classes):
        # print(mean_array[i]*test_value)
        exponent = -(test_value-mean_array[i])**2/(2*var_array[i])
        base = 1.0/((2.0*var_array[i]*math.pi)**0.5)
        blank_array[i] = base*math.e**exponent
        # calculateProbability(mean_array[i],var_array[i],test_value)
        evidence += np.prod(blank_array[i]) * 0.33

    print("Actual value:",row[num_cols])
    for i in range(num_classes):
        print(i, (np.prod(blank_array[i])*0.33)/evidence)
