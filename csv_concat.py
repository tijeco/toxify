import sys
import pandas as pd
import numpy as np

def getOptionValue(option):
    optionPos = [i for i, j in enumerate(sys.argv) if j == option][0]
    optionValue = sys.argv[optionPos + 1]
    return optionValue

if "-pos" in sys.argv:
    pos_path = getOptionValue("-pos")
else:
    print("\nplease specify input file name using -pos <file_name> \n")
    sys.exit()

if "-neg" in sys.argv:
    neg_path = getOptionValue("-neg")
else:
    print("\nplease specify input file name using -neg <file_name> \n")
    sys.exit()

df_dict = {}

def path_file(file,opt):
    with open(file) as f:
        for line in f:

            currentFile = line.strip()
            print(currentFile)
            df_name = currentFile.split("/")[-2]+"_"+currentFile.split("/")[-3]
            currentData = pd.read_csv(currentFile, header=None)
            currentHeaders = list(currentData)
            newHeaders = ["N:feature_" + str(header) for header in currentHeaders]
            currentData.columns = newHeaders
            # print(currentHeaders)
            # print(newHeaders)
            currentData['C:venom'] = opt
            df_dict[df_name] = currentData
            # print(list(currentData))
            # print(currentData.shape)

path_file(pos_path,1)
path_file(neg_path,0)

all_combined = pd.concat(df_dict)
print(all_combined.shape)

all_combined.to_csv('all_data.csv')
msk = np.random.rand(len(all_combined)) < 0.7

train = df[msk]

test = df[~msk]

train.to_csv('train.csv')
test.to_csv('test.csv')
