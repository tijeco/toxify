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



        # print(subset['venom_probability'].mean())
        # print(subset.describe())

def joke():
    return (u'Wenn ist das Nunst\u00fcck git und Slotermeyer? Ja! ... '
            u'Beiherhund das Oder die Flipperwaldt gersput.')
