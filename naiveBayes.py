import pandas as pd
import io
import math
import numpy as np

def calculateProbability(x,mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*stdev)))
	return (1.0 / (math.sqrt((2.0*math.pi) * stdev))) * exponent
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
        blank_array[i] = base*2.71828**exponent
        # calculateProbability(mean_array[i],var_array[i],test_value)
        evidence += np.prod(blank_array[i]) * 0.33

    print("Actual value:",row[num_cols])
    for i in range(num_classes):
        print(i, (np.prod(blank_array[i])*0.33)/evidence)
def getProb(test,mean,var):
    exponent = -(test-mean)**2/(2*var)
    base = 1/((2*var*math.pi)**0.5)
    print("Answer:",base*2.71828**exponent)

#setosa f1-4

getProb(5.9,4.9952381,0.12387921)
getProb(3.0,3.3952381,0.14192799)
getProb(4.2,1.45238095,0.02499419)
getProb(1.5,0.25,0.01085366)
print(0.6610380878866131*0.9689694474022094*1.9501164314264408e-61*1.0633848358756781e-42*0.33)
print("p:",0.042937519780444626*0.5723656795462498*1.8698423604236422e-55*1.4566187345893539e-30*0.33)
#4.3833031887638774e-104
#versicolor f1-4
getProb(5.9,5.93055556,0.32389683)
getProb(3.0,2.76111111,0.0955873)
getProb(4.2,4.26388889,0.25837302)
getProb(1.5,1.32222222,0.04234921)
print(1.0005121742072234*0.9267268626172791*1.0019154744133636*0.7811330463210072*0.33)
print("p:",0.8091930582373515*0.9595695455664152*0.8517033261116418*1.361053018995677*0.33)
#0.2394661681453941
# virginica f1-4
getProb(5.9,6.62142857,0.42562718)
getProb(3.0,2.9952381,0.10826945)
getProb(4.2,5.54285714286,0.300408163265) #the fucky one
getProb(1.5,2.03571429,0.07503484)
print(1.3508229413461246*0.9999798284750201*2.863161926047021*0.4872476945796461*0.33)
print("p:",0.34577231055453905*1.2438872253964925*0.03618934152729695*0.22795389619084247*0.33)
#0.6218695762283288
setosa = 0.6610380878866131*0.9689694474022094*1.9501164314264408e-61*1.0633848358756781e-42*0.33
versicolor = 1.0005121742072234*0.9267268626172791*1.0019154744133636*0.7811330463210072*0.33
virginica = 1.3508229413461246*0.9999798284750201*2.863161926047021*0.4872476945796461*0.33

denom = setosa + versicolor + virginica
print(setosa, versicolor, virginica)
print(setosa/denom, versicolor/denom, virginica/denom)

print("-----")

print(calculateProbability(5.54285714286,0.300408163265,4.2))

print(getProb(4.2,5.54285714286,0.300408163265))
