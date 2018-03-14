import pandas as pd
import sys

df = pd.read_csv(sys.argv[1],header=None)
# print(list(df))
# print(df[2]
# print(df.loc[df[2] == 0])
pos_df = df.loc[df[2] == 0]
for i in range(501,1000):
    TP=pos_df.loc[pos_df[0] > i/1000.0].shape[0]
    FP=pos_df.loc[pos_df[1] > 0.9].shape[0]
    print(i,TP,FP)
# print(pos_df.loc[pos_df[0] > 0.9])
# print(pos_df.loc[pos_df[0] > 0.9].shape)
# print(pos_df.loc[pos_df[1] > 0.9])
# print(pos_df.loc[pos_df[1] > 0.9].shape)
