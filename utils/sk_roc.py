import sys
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import cycle
import pandas as pd

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

df = pd.read_csv(sys.argv[1],header=None)
# print()
num_colums = df.shape[1] -1
# print(df)
# print(df[num_colums])
df_labels = df[num_colums].as_matrix()
# print(np.unique(df_labels))
num_classes = len(np.unique(df_labels))
def oneHot(mat,n):
    new_mat = np.zeros((len(mat), n) ,dtype=np.int)
    for row in range(len(mat)):
        # print(mat[row])
        for col in range(n):
            if mat[row] == col:
                new_mat[row][col] = 1
            else:
                new_mat[row][col] = 0
    return new_mat
# print(df_labels)

# df_labels = label_binarize(df_labels, classes=[0,1, 2])
df_labels = oneHot(df_labels,num_classes)
# print(df.drop([num_colums], axis=1))
df_values = df.drop([num_colums], axis=1).as_matrix()

# print(df_labels)
# print(df_labels.shape)
# print(df_values.shape)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

# print(y_test)
# print(df_labels)
print(df_labels[:,1])
print(df_labels)
print(df_values)

for i in range(num_colums):
    fpr[i], tpr[i], _ = roc_curve(df_labels[:, i], df_values[:, i])
    # print(fpr[i],tpr[i])
    roc_auc[i] = auc(fpr[i], tpr[i])
print(fpr)
new_labels = df[num_colums]
new_values = df[num_colums-1]
print(roc_curve(new_labels,new_values))
new_labels = df[num_colums]
new_values = df[num_colums-2]
new_fpr, new_tpr, thresholds = roc_curve(new_labels,new_values)
print(thresholds)
for i in thresholds:
    print(i)
# plt.figure()
# plt.plot(new_fpr,new_tpr)
# plt.show()
#
# print(auc(new_fpr,new_tpr))

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
