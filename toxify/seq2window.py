import sys
from itertools import groupby
import pandas as pd
import numpy as np


window_size = 15
max_seq_len = 100

factorDict ={
"A":[-0.59,-1.30,-0.73,1.57,-0.15],
"C":[-1.34,0.47,-0.86,-1.02,-0.26],
"D":[1.05,0.30,-3.66,-0.26,-3.24],
"E":[1.36,-1.45,1.48,0.11,-0.84],
"F":[-1.01,-0.59,1.89,-0.40,0.41],
"G":[-0.38,1.65,1.33,1.05,2.06],
"H":[0.34,-0.42,-1.67,-1.47,-0.08],
"I":[-1.24,-0.55,2.13,0.39,0.82],
"K":[1.83,-0.56,0.53,-0.28,1.65],
"L":[-1.02,-0.99,-1.51,1.27,-0.91],
"M":[-0.66,-1.52,2.22,-1.01,1.21],
"N":[0.95,0.83,1.30,-0.17,0.93],
"P":[0.19,2.08,-1.63,0.42,-1.39],
"Q":[0.93,-0.18,-3.01,-0.50,-1.85],
"R":[1.54,-0.06,1.50,0.44,2.90],
"S":[-0.23,1.40,-4.76,0.67,-2.65],
"T":[-0.03,0.33,2.21,0.91,1.31],
"V":[-1.34,-0.28,-0.54,1.24,-1.26],
"W":[-0.60,0.01,0.67,-2.13,-0.18],
"Y":[0.26,0.83,3.10,-0.84,1.51],
"B":[1,0.565,-1.18,-0.215,-1.155],
"Z":[1.145,-0.815,-0.765,-0.195,-1.345],
"J":[-1.13,-0.77,0.31,0.83,-0.045],
"U":[-0.13,-0.12,0.6,-0.03,-0.115],
"O":[-0.13,-0.12,0.6,-0.03,-0.115],
"X":[-0.13,-0.12,0.6,-0.03,-0.115]
}
def fasta_iter(fasta_name):
    with open(fasta_name) as fh:
        faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))
        for header in faiter:
            headerStr = header.__next__()[1:].strip().split()[0]#Entire line, add .split[0] for just first column
            seq = "".join(s.strip() for s in faiter.__next__())
            yield (headerStr, seq)
# function that converts fasta file to np

def fa2pd(fastaFile,window,maxLen):

    seqDict = {"headers":[],"sequences":[]}
    print("IN:",fastaFile)
    seq_iter = fasta_iter(fastaFile)

    for ff in seq_iter:
        headerStr, seq = ff
        seq=seq.strip("*")
        if len(seq) <= maxLen and len(seq) >= window:
            seqDict["headers"].append(headerStr)
            seqDict["sequences"].append(seq)

    seq_pd = pd.DataFrame.from_dict(seqDict)
    return(seq_pd)




def seq15mer(  protein_panda ,window,maxLen):#,max_value=1000): n_samples=1000
    data = []
    for index, row in protein_panda.iterrows():
        headerStr, seq = row['headers'],row['sequences']
        seq=seq.strip("*")
        if len(seq) <= maxLen:
            for i in range(len(seq)):
                if len(seq) - window >= i:
                    data.append([headerStr,"kmer_"+str(i),seq[i:i+window]])
    data = np.array(data)
    return data
# print(fa2pd(sys.argv[1]))
# fastaPD = fa2pd(sys.argv[1])
# print(seq15mer(fastaPD))


def seqs2train(pos,neg,window,maxLen):
    pos_mat = []
    neg_mat = []
    for pos_file in pos[0]:
        # print("POS:",pos_file)
        pos_pd = fa2pd(pos_file,window,maxLen)
        if window:
            pos_mat.append(seq15mer(pos_pd,window,maxLen))
        else:
            print("POS PD:")
            pos_mat.append(pos_pd)

    for neg_file in neg[0]:
        # print("NEG:",neg_file)
        neg_pd = fa2pd(neg_file,window,maxLen)
        # print(neg_pd.shape)
        if window:
            print("window_size: ",window)
            neg_mat.append(seq15mer(neg_pd,window,maxLen))
        else:
            print("NEG PD:")
            neg_mat.append(neg_pd)
    #NOTE: struggling to combine fa_pds from multiple files!!!
    # print(pos_mat)
    pos_np = np.vstack(pos_mat)
    neg_np = np.vstack(neg_mat)
    # print(pos_np.shape[0])

    pos_ones = np.ones((pos_np.shape[0],1))
    # pos_ones = np.full((pos_np.shape[0], 2), np.array([0,1]))
    pos_labeled = np.append(pos_np,pos_ones, axis=1)
    # print("POS NP:")
    # print(pos_labeled)
    neg_zeros = np.zeros((neg_np.shape[0],1))
    # neg_zeros = np.full((neg_np.shape[0], 2), np.array([0,1]))
    # print(neg_zeros)
    neg_labeled = np.append(neg_np,neg_zeros, axis=1)
    # print(neg_labeled)

    mat_np = np.vstack([pos_labeled,neg_labeled])
    indices = np.random.permutation(mat_np.shape[0])
    training_idx, test_idx = indices[:int(mat_np.shape[0]*0.8)], indices[int(mat_np.shape[0]*0.8):]
    training_np, test_np = mat_np[training_idx,:], mat_np[test_idx,:]
    return (training_np,test_np)





def seq2atchley(s,window,maxLen):
    seqList = []
    # print("WINDOW: ",window,window == True)
    if window:
        for i in range(len(s)):
            aa = s[i]
            seqList.append([])
            for factor in factorDict[aa]:
                seqList[i].append(factor)
    else:
        for i in range(maxLen):
            try:
                aa = s[i]
                seqList.append([])
                for factor in factorDict[aa]:
                    seqList[i].append(factor)
            except:
                seqList.append([])
                for factor in factorDict["X"]:
                    seqList[i].append(0.0)
                    #NOTE, alternatively you could append factor as iff you were adding a bunch of Xs to the end



        # print("here will go zero-padding")
    print(np.transpose(np.array(seqList)))
    return np.transpose(np.array(seqList))
