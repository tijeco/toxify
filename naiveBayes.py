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


# print(training_data.describe().index['mean'])

# print(test_data.iloc[:,0:num_cols])
#
# print(test_data.loc[test_data[4] == 0])

# print(train_var,train_mean)
# print(training_data.iloc[:,0:num_cols].mean())
# pop_mean = training_data.iloc[:,0:num_cols].mean()
# # print(training_data.iloc[:,0:num_cols].var())
# pop_var = training_data.iloc[:,0:num_cols].var()
# print(pop_var)
# print(math.sqrt(25),math.pi)
# # print(1/(math.sqrt(2*math.pi*pop_var)))
# print(1/((2*pop_var*math.pi)**0.5))
# pop_base = 1/((2*pop_var*math.pi)**0.5)
# # print(training_data[:2].mean())
# # print(training_data.var())
# print(pop_mean[0])


print("#################")
# for i in test_data:
#     print(i)
#     print(test_data.iloc[i])

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
        blank_array[i] = base**exponent
        evidence += np.prod(blank_array[i]) * 0.33
    # print(blank_array)
    print("Actual value:",row[num_cols])
    for i in range(num_classes):
        print(i, (np.prod(blank_array[i])*0.33)/evidence)

    # evidence = 0
    # for i in range(num_cols):
    #     exponent = -(test_value-pop_mean[i])**2/(2*pop_var[i])
    #     pop_base = 1/((2*pop_var[i]*math.pi)**0.5)
    #     blank_array[i] =  pop_base**exponent
    #     evidence+=np.prod(blank_array[i])*0.33
    #     # print(i,np.prod(blank_array[i]))
    #     # print(i)
    # print(blank_array)
    # for i in blank_array:
    #     print(np.prod(i)/evidence)
    # print("evidence:",evidence)
    # print(test_data.shape[1]-1)
    0

    # print(-(test_value-pop_mean)**2/(2*pop_var))
    # exponent = -(test_value-pop_mean)**2/(2*pop_var)
    # print(pop_base**exponent)
    # print(np.prod(pop_base**exponent))
    # print(test_value)
# print(np.zeros((4,4)))
# blank_array = np.zeros((4,4))
# blank_array[0] = 0
# blank_array[1] = 1
#
# print(blank_array)
