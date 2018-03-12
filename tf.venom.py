#!/usr/bin/env python

# conda execute
# env:
#  - python >=3
#  - numpy
#  - tensorflow



import os
import urllib.request
import sys
import numpy as np
import tensorflow as tf
# training_data = "venom.binary.train.csv"
# test_data = "venom.binary.test.csv"

def getOptionValue(option):
    optionPos = [i for i, j in enumerate(sys.argv) if j == option][0]
    optionValue = sys.argv[optionPos + 1]
    return optionValue


if "-train" in sys.argv:
    training_data = getOptionValue("-train")
else:
    print("please provide training data with -train")
    sys.exit()

if "-test" in sys.argv:
    test_data = getOptionValue("-test")
else:
    print("please provide test data with -test")
    sys.exit()


training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=training_data,
    target_dtype=np.int,
    features_dtype=np.float32)
with open(training_data) as f:
    for line in f:
        row = line.strip().split(",")
        if len(row) ==2:
            data_shape = int(row[1])
        else:
            break
print("NUM FEATURES:",data_shape)
# data_shape = training_set.shape[1]
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=test_data,
    target_dtype=np.int,
    features_dtype=np.float32)
feature_columns = [tf.feature_column.numeric_column("x", shape=[data_shape])]
print(feature_columns)
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                  hidden_units=[500,500,500],
                                      n_classes=2,
                                      # dropout=0.02,
                                      model_dir="tmp/venom_model",
                                      optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.001, l1_regularization_strength=0.001)
                                      )
                                      #Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD'

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(training_set.data)},
    y=np.array(training_set.target),
    num_epochs=3,
    shuffle=True)
classifier.train(input_fn=train_input_fn, steps=1000)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(test_set.data)},
    y=np.array(test_set.target),
    num_epochs=1,
    shuffle=False)
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


"""
49044 venom.train.csv
21020 venom.test.csv
120,4,setosa,versicolor,virginica


98089,6,pos,f1,f2,f3,f4,f5
42040,6,pos,f1,f2,f3,f4,f5

growforest -train train.fm -rfpred forest.sf -target B:FeatureName -oob -nCores 16 -nTrees 1000 -leafSize 8

"""
