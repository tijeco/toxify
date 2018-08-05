import sys
from itertools import groupby

def fasta_iter(fasta_name):
    with open(fasta_name) as fh:
        faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))
        for header in faiter:
            headerStr = header.__next__()[1:].strip().split()[0]#Entire line, add .split[0] for just first column
            seq = "".join(s.strip() for s in faiter.__next__())
            yield (headerStr, seq)

# seq = "thebrownfoxjumpsoverthelazydog"
window_size = 15
seq_iter = fasta_iter(sys.argv[1])
with open(sys.argv[1]+".window","w") as out:
    for ff in seq_iter:
        headerStr, seq = ff
        seq=seq.strip("*")
        if len(seq) <= 100:

            for i in range(len(seq)):
                if len(seq) - window_size >= i:
                    out.write(">"+headerStr+"_window"+str(i)+"\n")
                    out.write(seq[i:i+window_size]+"\n")
