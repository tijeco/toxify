import pandas as pd
import sys

df = pd.read_csv(sys.argv[1],header=None)
print(list(df))
