import sys
import os

from itertools import groupby
import sys


def fasta_iter(fasta_name):
    fh = open(fasta_name)
    faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))
    for header in faiter:
        headerStr = header.__next__()[1:].strip().split()[0]#Entire line, add .split[0] for just first column
        seq = "".join(s.strip() for s in faiter.__next__())
        yield (headerStr, seq)
# line1 = True
used_dict = {}

if not os.path.exists(sys.argv[1].strip("/").split("/")[-1]+".seq_dir"):
    os.makedirs(sys.argv[1].strip("/").split("/")[-1]+".seq_dir")
for ff in fasta_iter(sys.argv[1]):
    headerStr, seq = ff
    with open(sys.argv[1]+".seq_dir"+headerStr.replace("|","_"),"w") as out:
        out.write(headerStr+"\n")
        out.write(seq+"\n")
