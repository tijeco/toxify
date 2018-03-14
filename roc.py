import pandas as pd
import sys

df = pd.read_csv(sys.argv[1],header=None)
print(list(df))
# print(df[2]
print(df.loc[df[2] == 0])
pos_df = df.loc[df[2] == 0]
print(pos_df.loc[pos_df[0] > 0.9])
print(pos_df.loc[pos_df[1] > 0.9])
