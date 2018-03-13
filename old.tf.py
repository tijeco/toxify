from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys


tf.logging.set_verbosity(tf.logging.INFO)

# Data sets
# training_data = "humantrain3.csv"
# test_data = "humantest3.csv"

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

with open(training_data) as f:
    for line in f:
        row = line.strip().split(",")
        if len(row) ==2:
            data_shape = int(row[1])
        else:
            break
print("NUM FEATURES:",data_shape)

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=training_data,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=test_data,
    target_dtype=np.int,
    features_dtype=np.float32)

#Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=data_shape)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    test_set.data,
    test_set.target,
    every_n_steps=50)
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=2,
                                            model_dir="/tmp/H.uman4.25.1",
                                            config=tf.contrib.learn.RunConfig(
                                                save_checkpoints_secs=1))

# # Fit model.

classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=2000)

# # Evaluate accuracy.
accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

# # Classify two new flower samples.
# new_samples = np.array(
#     [[255,0,123,None,None,None,255,81,0,None,None,None,117,12,0,None,None,None
#     ,255,33,0,205,27,0,255,48,0,255,120,0,255,21,0,255,20,0,211,39,0,255,84,0,255,54,0,255,39,0,None,None,None,255,45,0,194,0,41,194,0,41,180,18,0,180,18,0,196,\
#     24,0,196,24,0,149,0,45,149,0,45,255,0,200,None,None,None],
#        [255,0,123,None,None,None,255,81,0,None,None,None,117,12,0,None,None,None,255,33,0,205,27,0,255,48,0,255,120,0,255,21,0,255,20,0,211,39,0,255,84,0,255,54,0,255,39,0,
#        None,None,None,255,45,0,194,0,41,194,0,41,180,18,0,180,18,0,196,24,0,196,24,0,149,0,45,149,0,45,255,0,200,None,None,None]],
#        dtype=float)
# y = list(classifier.predict(new_samples, as_iterable=True))
# print('Predictions: {}'.format(str(y)))
