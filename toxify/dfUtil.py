import pandas as pd
import numpy as np

drop_these = {'N:feature_118': True, 'N:feature_121': True, 'N:feature_127': True, 'N:feature_149': True, 'N:feature_174': True, 'N:feature_177': True, 
'N:feature_180': True, 'N:feature_183': True, 'N:feature_199': True, 'N:feature_202': True, 'N:feature_205': True, 'N:feature_208': True, 'N:feature_211': True,
'N:feature_214': True, 'N:feature_227': True, 'N:feature_230': True, 'N:feature_233': True, 'N:feature_236': True, 'N:feature_239': True, 'N:feature_245': True,
'N:feature_255': True, 'N:feature_261': True, 'N:feature_264': True, 'N:feature_267': True, 'N:feature_270': True, 'N:feature_273': True, 'N:feature_309': True,
'N:feature_311': True, 'N:feature_314': True, 'N:feature_317': True, 'N:feature_318': True, 'N:feature_319': True, 'N:feature_320': True, 'N:feature_322': True,
'N:feature_323': True, 'N:feature_326': True, 'N:feature_329': True, 'N:feature_414': True}

def splitTrain(pos_list,neg_list):
    pos_df = labelDF(pos_list,1)
    neg_df = labelDF(neg_list,0)
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
