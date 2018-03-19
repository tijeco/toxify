import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
# plt.figure(1)                # the first figure
# plt.subplot(211)             # the first subplot in the first figure
# plt.plot([0],[0])


roc_data = pd.read_csv(sys.argv[1],delim_whitespace=True,header=None)
# print(roc_data)

y_values = roc_data[1]
x_values = roc_data[2]

plt.plot([0]+x_values.tolist()+[1], [0]+y_values.tolist()+[1])
plt.plot([0]+[1], [0]+[1])
plt.axis([0, 1, 0, 1])
# plt.show()
plt.savefig(sys.argv[1])
