import sys
from itertools import groupby
import numpy as np

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


def atchley(s,h):
    line2write = ""
    for aa in s:
        for factor in factorDict[aa]:
            line2write += str(factor) +","
    # padding = "NA,"*(500-(len(s)*5))
    return h+","+line2write
def seq2np(s):
    mat = np.zeros(shape=(5,15))
    for factors in range(5):
        for i in range(len(s)):
            aa = s[i]
            # print(factors,i,aa)
            mat[factors][i] = factorDict[aa][factors]
    return mat
if False:
    seq_list = []
    seq_iter = fasta_iter("training_data/reviewed_subset.NegEuk.fa.window")
    for ff in seq_iter:
        headerStr,seq = ff
        seq_list.append(seq2np(seq))
    seq_np = np.array(seq_list)

    seq_labels = np.full((seq_np.shape[0], 2), np.array([0,1]))
    np.save('training_data/reviewed_subset.NegEuk.fa.window.npy', seq_np)
    np.save('training_data/reviewed_subset.NegEuk.fa.window.labels.npy', seq_labels)


    seq_list = []
    seq_iter = fasta_iter("training_data/reviewed_subset.negMet.fa.window")
    for ff in seq_iter:
        headerStr,seq = ff
        seq_list.append(seq2np(seq))
    seq_np = np.array(seq_list)

    seq_labels = np.full((seq_np.shape[0], 2), np.array([0,1]))
    np.save('training_data/reviewed_subset.negMet.fa.window.npy', seq_np)
    np.save('training_data/reviewed_subset.negMet.fa.window.labels.npy', seq_labels)


    seq_list = []
    seq_iter = fasta_iter("training_data/databases.met.hard.fa.window")
    for ff in seq_iter:
        headerStr,seq = ff
        seq_list.append(seq2np(seq))
    seq_np = np.array(seq_list)

    seq_labels = np.full((seq_np.shape[0], 2), np.array([0,1]))
    np.save('training_data/databases.met.hard.fa.window.npy', seq_np)
    np.save('training_data/databases.met.hard.fa.window.labels.npy', seq_labels)


    seq_list = []
    seq_iter = fasta_iter("training_data/allPos.fa.window")
    for ff in seq_iter:
        headerStr,seq = ff
        seq_list.append(seq2np(seq))
    seq_np = np.array(seq_list)

    seq_labels = np.full((seq_np.shape[0], 2), np.array([1,0]))
    np.save('training_data/allPos.fa.window.npy', seq_np)
    np.save('training_data/allPos.fa.window.labels.npy', seq_labels)
if False:
    negEuk_np = np.load('training_data/reviewed_subset.NegEuk.fa.window.npy')
    negEuk_labels = np.load('training_data/reviewed_subset.NegEuk.fa.window.labels.npy')
    negMet_np = np.load('training_data/reviewed_subset.negMet.fa.window.npy')
    negMet_labels = np.load('training_data/reviewed_subset.negMet.fa.window.labels.npy')
    met_hard_np = np.load('training_data/databases.met.hard.fa.window.npy')
    met_hard_labels = np.load('training_data/databases.met.hard.fa.window.labels.npy')
    allPos_np = np.load('training_data/allPos.fa.window.npy')
    allPos_labels = np.load('training_data/allPos.fa.window.labels.npy')

    all_np = np.concatenate([negEuk_np,negMet_np,met_hard_np,allPos_np])
    all_labels = np.concatenate([negEuk_labels,negMet_labels,met_hard_labels,allPos_labels])
    np.save("training_data/all_np.npy",all_np)
    np.save("training_data/all_labels.npy",all_labels)

all_np = np.load("training_data/all_np.npy")
all_labels = np.load("training_data/all_labels.npy")

indices = np.random.permutation(all_np.shape[0])
training_idx, test_idx = indices[:int(all_np.shape[0]*0.8)], indices[int(all_np.shape[0]*0.8):]

training_np, test_np = all_np[training_idx,:], all_np[test_idx,:]
training_labels, test_labels = all_labels[training_idx,:], all_labels[test_idx,:]

np.save("training_data/training_np.npy",training_np)
np.save("training_data/test_np.npy",test_np)
np.save("training_data/training_labels.npy",training_labels)
np.save("training_data/test_labels.npy",test_labels)


if False:
    with open(sys.argv[2],"w") as out:
        seq_iter = fasta_iter(sys.argv[1])
        for ff in seq_iter:
            headerStr,seq = ff
            # seq=seq.strip("*")
            # if len(seq) <= 100:
            out.write(atchley(seq,headerStr)[:-1]+"\n")
