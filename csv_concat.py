import sys
import pandas as pd

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
            print(currentFile.split("/")[-3])
            currentData = pd.read_csv(currentFile, header=None)
            currentHeaders = list(currentData)
            newHeaders = ["N:feature_" + str(header) for header in currentHeaders]
            currentData.columns = newHeaders
            # print(currentHeaders)
            # print(newHeaders)
            currentData['C:venom'] = opt
            # print(list(currentData))
            # print(currentData.shape)

path_file(pos_path,1)
path_file(neg_path,0)
