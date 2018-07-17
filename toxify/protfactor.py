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


def seq2np(s):
    mat = np.zeros(shape=(5,15))
    for factors in range(5):
        for i in range(len(s)):
            aa = s[i]
            # print(factors,i,aa)
            mat[factors][i] = factorDict[aa][factors]
    return mat

class ProteinVectors(object):
    def __init__(self,  sequences ):#,max_value=1000): n_samples=1000
        self.data = []
        self.labels = []
        self.seqlen = []
        for row in sequences:
            prot_seq = row[2]
            # Random sequence length
            self.data.append(seq2np(prot_seq))
            # for when there are actually labels
            try:
                if row[3] == '0':
                    self.labels.append([0.,1.])
                if row[3] =='1':
                    self.labels.append([1.,0.])
            except:
                None
        self.data = np.array(self.data)
        self.batch_id = 0
    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen
