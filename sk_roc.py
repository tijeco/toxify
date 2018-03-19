import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import cycle
import pandas as pd

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target


# y = label_binarize(y, classes=[0, 1, 2])
# print(y.flatten())
# n_classes = y.shape[1]

# Add noisy features to make the problem harder
# random_state = np.random.RandomState(0)
# n_samples, n_features = X.shape
# X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    # random_state=0)

# Learn to predict each class against the other
# classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
# y_fit = classifier.fit(X_train, y_train)
# y_score = y_fit.predict_proba(X_test)

# print(y_score)
# print(X)



df = pd.read_csv(sys.argv[1],header=None)
# print()
num_colums = df.shape[1] -1
# print(df)
# print(df[num_colums])
df_labels = df[num_colums].as_matrix()
df_labels = label_binarize(df_labels, classes=[0, 1])
# print(df.drop([num_colums], axis=1))
df_values = df.drop([num_colums], axis=1).as_matrix()

print(df_labels)
print(df_labels.shape)

print(df_values.shape)

# print(df_labels)
# print(df_values)
#
# print(y_test.flatten())
# print(y_test.shape)
# print(y_test.ravel())
# print(y_score)
# print(y_score.ravel())
# print(y_score.flatten())
# print(y_score.ravel().shape)


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

# print(y_test)
# print(df_labels)

for i in range(num_colums):
    fpr[i], tpr[i], _ = roc_curve(df_labels[:, i], df_values[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

#y_test is labels, y_score is probabilities
fpr["micro"], tpr["micro"], _ = roc_curve(df_labels.flatten(), df_values.flatten())


roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# plt.figure()
lw = 2
# plt.plot(fpr[2], tpr[2], color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()


# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_colums)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_colums):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= num_colums

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(num_colums), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
# plt.show()

plt.savefig(sys.argv[1]+".png")
