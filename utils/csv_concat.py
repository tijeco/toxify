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
# print(df_dict)

all_combined.to_csv('all_data.csv')
msk = np.random.rand(len(all_combined)) < 0.7

train = all_combined[msk]
train.insert(0, '.', range(1, 1 + len(train)))

test = all_combined[~msk]
test.insert(0, '.', range(1, 1 + len(test)))


train.to_csv(pos_path.replace(".txt",".train.csv"),na_rep='nan', sep='\t', index=False,quoting=3)
test.to_csv(pos_path.replace(".txt",".test.csv"),na_rep='nan', sep='\t', index=False,quoting=3)
