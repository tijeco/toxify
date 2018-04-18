import pandas as pd
import numpy as np

def splitTrain(pos_list,neg_list):
    pos_df = labelDF(pos_list,1)
    neg_df = labelDF(neg_list,0)
    print("----------------NOW concatING NEG AND POS ---------------------\n\n\n\n\n\n")
    print("neg shape:",neg_df.shape)
    print("pos shape:",pos_df.shape)
    print("neg headers:",list(neg_df))
    print("pos headers:",list(pos_df))
    print("----------------NOW concatING NEG AND POS ---------------------\n\n\n\n\n\n")
    neg_and_pos = pd.concat([neg_df,pos_df])
    msk = np.random.rand(len(neg_and_pos)) < 0.7

    train = neg_and_pos[msk]
    test = neg_and_pos[~msk]
    return (train,test)

def labelDF(df_list,label):
    all_combined = pd.concat(df_list)
    currentHeaders = list(all_combined)
    newHeaders = ["N:feature_" + str(header) for header in currentHeaders]
    all_combined.columns = newHeaders
    all_combined['C:venom'] = label
    return all_combined

def df2tf(df):

    return tf_df
def df2fm(df):

    return fm_df
