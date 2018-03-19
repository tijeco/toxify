import pandas as pd
import io
import math
import numpy as np
import sys
from sklearn.naive_bayes import GaussianNB

def getOptionValue(option):
    optionPos = [i for i, j in enumerate(sys.argv) if j == option][0]
    optionValue = sys.argv[optionPos + 1]
    return optionValue




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

testing_data = pd.read_csv(test_data, skiprows=[0], header=None)

num_cols = testing_data.shape[1]-1

num_classes =  len(training_data[num_cols].unique())

gnb = GaussianNB()


train_df = training_data.drop([num_cols], axis=1)
train_labels  = training_data[num_cols]

test_df = testing_data.drop([num_cols], axis=1)
test_labels  = testing_data[num_cols]

y_fit = gnb.fit(train_df,train_labels)
y_pred = gnb.predict_proba(test_df)
print(y_pred[0])

df = pd.DataFrame(y_pred)
df[3] = test_labels
test_data+"_predictions.nb.csv"
print(df)

df.to_csv(test_data+"_predictions.nb.csv", index=False, header=False)
