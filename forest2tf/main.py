import pandas as pd
import sys
try:
    fpath = sys.argv[1]
except:
    fpath = "forest2tf/forestfires.trans.fm"
outpath = fpath.replace(".fm",".csv")
df1 = pd.read_csv(fpath, sep='\t',header='infer')

# my_cols = set(df1.columns)
df1 = df1.drop(['.'], axis=1)
# df2 = df1[my_cols]

print(df1.shape)
with open(outpath, 'w') as out:
    out.write(str(df1.shape[0])+","+str(df1.shape[1])+"\n")


with open(outpath, 'a') as f:
    df1.to_csv(f, header=False)
