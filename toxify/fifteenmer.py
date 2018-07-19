from itertools import groupby
import numpy as np
import pandas as pd
def fasta_iter(fasta_name):
    with open(fasta_name) as fh:
        faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))
        for header in faiter:
            headerStr = header.__next__()[1:].strip().split()[0]#Entire line, add .split[0] for just first column
            seq = "".join(s.strip() for s in faiter.__next__())
            yield (headerStr, seq)


import sys
from itertools import groupby

def fasta_iter(fasta_name):
    with open(fasta_name) as fh:
        faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))
        for header in faiter:
            headerStr = header.__next__()[1:].strip().split()[0]#Entire line, add .split[0] for just first column
            seq = "".join(s.strip() for s in faiter.__next__())
            yield (headerStr, seq)
class ProteinWindows(object):
    def __init__(self,  protein_fasta ):#,max_value=1000): n_samples=1000
        self.data = []

        window_size = 15
        max_seq_len = 100
        protein_iter = fasta_iter(protein_fasta)
        predictions_dir = protein_fasta +"_toxify_predictions"
        with open(predictions_dir+"/used_proteins.fa","w") as out:
            for ff in protein_iter:
                headerStr, seq = ff
                seq=seq.strip("*")
                if len(seq) <= max_seq_len:
                    if len(seq) >= window_size:
                        out.write(">"+headerStr+"\n")
                        out.write(seq+"\n")
                    for i in range(len(seq)):
                        if len(seq) - window_size >= i:

                            self.data.append([headerStr,"kmer_"+str(i),seq[i:i+window_size]])
            self.data = np.array(self.data)


def regenerate(df_15mer):
    print(df_15mer.header.unique())
    unique_seqs = df_15mer.header.unique()
    outDict = {'header':[], 'n_15mers':[], 'mean_venom_probability':[] ,'median_venom_probability':[], 'sd_venom_probability':[] }

    for h in unique_seqs:

        # print("header:",h)
        # print(df_15mer.loc[df_15mer['header'] == h])
        subset = df_15mer.loc[df_15mer['header'] == h]
        n_15mer = len(subset)

        mean_venom_probability = pd.to_numeric(subset['venom_probability'],errors='coerce').mean()
        median_venom_probability = pd.to_numeric(subset['venom_probability'],errors='coerce').median()
        sd_venom_probability = pd.to_numeric(subset['venom_probability'],errors='coerce').std()
        # print(mean_venom_probability,median_venom_probability,sd_venom_probability)
        outDict['header'].append(h)
        outDict['n_15mers'].append(n_15mer)
        outDict['mean_venom_probability'].append(mean_venom_probability)
        outDict['median_venom_probability'].append(median_venom_probability)
        outDict['sd_venom_probability'].append(sd_venom_probability)

    return pd.DataFrame(outDict)
        # for i in subset['sequence']

def seqs2train(pos,neg):
    mat=[]
    for i in pos[0]:
        pos_iter = fasta_iter(i)
        for ff in pos_iter:
            headerStr, seq = ff
            if len(seq) < 101 and len(seq) > 14:
                mat.append([headerStr,seq,1])
    for i in neg[0]:
        neg_iter = fasta_iter(i)
        for ff in neg_iter:
            headerStr, seq = ff
            if len(seq) < 101 and len(seq) > 14:
                mat.append([headerStr,seq,0])
    # print(mat)
    mat_np = np.array(mat)
    indices = np.random.permutation(mat_np.shape[0])
    training_idx, test_idx = indices[:int(mat_np.shape[0]*0.8)], indices[int(mat_np.shape[0]*0.8):]
    training_np, test_np = mat_np[training_idx,:], mat_np[test_idx,:]
    return (training_np,test_np)

        # print(subset['venom_probability'].mean())
        # print(subset.describe())

def joke():
    return ('joke')
