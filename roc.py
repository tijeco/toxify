import pandas as pd
import sys

df = pd.read_csv(sys.argv[1],header=None)
# print(list(df))
# print(df[2]
# print(df.loc[df[2] == 0])
neg_df = df.loc[df[2] == 0]
# print(df.shape)
for i in range(1,1000):
    TP=neg_df.loc[neg_df[0] > i/1000.0].shape[0]/neg_df.shape[0]
    FP=neg_df.loc[neg_df[1] > i/1000.0].shape[0]/neg_df.shape[0]
    # print(i,TP,FP)

pos_df = df.loc[df[2] == 1]
# print(df.shape)
for i in range(1,1000):
    FP=pos_df.loc[pos_df[0] > i/1000.0].shape[0]/pos_df.shape[0]
    TP=pos_df.loc[pos_df[1] > i/1000.0].shape[0]/pos_df.shape[0]
    print(i,TP,FP)

# print(pos_df.describe())
# print(neg_df.describe())
