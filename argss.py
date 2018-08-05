#!/Users/jeffcole/miniconda3/bin/python

import tensorflow as tf
import numpy as np
from random import shuffle
import random
import os
import sys
from itertools import groupby
import argparse

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

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="subparser_name")
train_parser = subparsers.add_parser('train')

train_parser.add_argument('-pos',action='append',nargs='*')
train_parser.add_argument('-neg',action='append',nargs='*')
args = parser.parse_args()
print(args.pos)
print(args.neg)

def fasta_iter(fasta_name):
    with open(fasta_name) as fh:
        faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))
        for header in faiter:
            headerStr = header.__next__()[1:].strip().split()[0]#Entire line, add .split[0] for just first column
            seq = "".join(s.strip() for s in faiter.__next__())
            yield (headerStr, seq)

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

print(seqs2train(args.pos,args.neg)[0][:,1])

def seq2atchley(s):
    seqList = []
    for aa in s:
        for factor in factorDict[aa]:
            seqList.append([factor])
    return seqList
print("MENDEL")
print(seq2atchley("MENDEL"))

training_seqs = seqs2train(args.pos,args.neg)[0]
test_seqs = seqs2train(args.pos,args.neg)[1]

class ProteinVectors(object):
    def __init__(self,  sequences ):#,max_value=1000): n_samples=1000
        self.data = []
        self.labels = []
        self.seqlen = []
        for row in sequences:
            prot_seq = row[1].strip("*")
            # Random sequence length
            num_AA = len(prot_seq)
            # Monitor sequence seq_length for TensorFlow dynamic calculation
            self.seqlen.append(num_AA*5)
            s = seq2atchley(prot_seq)
            s += [[0.] for i in range(500 - num_AA*5)]
            self.data.append(s)
            if row[2] == '0':
                self.labels.append([0.,1.])
            if row[2] =='1':
                self.labels.append([1.,0.])
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

trainset = ProteinVectors(training_seqs)
testset = ProteinVectors(test_seqs)
# print(trainset.data)
print("20 Batch")
print(trainset.next(1))

# Parameters
training_epochs = 25
learning_rate = 0.01
training_steps = 10000
batch_size = 100
display_step = 200

# Network Parameters
n_hidden = 64 # hidden layer num of features
n_classes = 2 # linear sequence or not

# tf Graph input
x = tf.placeholder("float", [None, 500, 1])
y = tf.placeholder("float", [None, n_classes])

# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


if False:
    def  dynamicRNNdynamic(x, seqlen, weights, biases):

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, 500, 1)

        # Define a lstm cell with tensorflow
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

        # Get lstm cell output, providing 'sequence_length' will perform dynamic
        # calculation.
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                    sequence_length=seqlen)

        # When performing dynamic calculation, we must retrieve the last
        # dynamically computed output, i.e., if a sequence length is 10, we need
        # to retrieve the 10th output.
        # However TensorFlow doesn't support advanced indexing yet, so we build
        # a custom op that for each sample in batch size, get its length and
        # get the corresponding relevant output.

        # 'outputs' is a list of output at every timestep, we pack them in a Tensor
        # and change back dimension to [batch_size, n_step, n_input]
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])

        # Hack to build the indexing and retrieve the right output.
        batch_size = tf.shape(outputs)[0]
        # Start indices for each sample
        index = tf.range(0, batch_size) * 500 + (seqlen - 1)
        # Indexing
        outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

        # Linear activation, using outputs computed above
        return tf.matmul(outputs, weights['out']) + biases['out']

    pred =  dynamicRNNdynamic(x, seqlen, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()


    # Start training# Start
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(len(trainset)/ batch_size)
            # Loop over all batches
            for i in range(total_batch):
                # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
                # Run optimization op (backprop), cost op (to get loss value)
                # and summary nodes
                _, c, summary = sess.run([optimizer, cost, merged_summary_op],
                                         feed_dict={x: batch_xs, y: batch_ys})
                # Write logs at every iteration
                summary_writer.add_summary(summary, epoch * total_batch + i)
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if (epoch+1) % display_epoch == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
        for step in range(1, training_steps+1):
            batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,seqlen: batch_seqlen})
            if step % display_step == 0 or step == 1:
                # Calculate batch accuracy & loss
                acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y,
                                                    seqlen: batch_seqlen})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

        print("Optimization Finished!")

        # Calculate accuracy
        test_data = testset.data
        test_label = testset.labels
        test_seqlen = testset.seqlen
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                          seqlen: test_seqlen}))
if True:
    train_X = np.load('training_data/training_np.npy') #(7352, 128, 9)
    test_X  = np.load('training_data/test_np.npy')
    train_Y = np.load('training_data/training_labels.npy') #(7352, 6)
    test_Y  = np.load('training_data/test_labels.npy')


print(train_X.shape)
print(train_Y.shape)
# Parameters
n = train_X.shape[0]  # Number of training sequences
print(n) #7352
n_test = train_Y.shape[0]  # Number of test sequences
print(n_test) #7352
m = train_Y.shape[1]  # Output dimension
print(m) #6
d = train_X.shape[2]  # Input dimension
print(d) #9
T = train_X.shape[1]  # Sequence length
epochs = 500
# batch_size = 100

lr = 0.01  # Learning rate

# Placeholders
inputs = tf.placeholder(tf.float32, [None, None, d])
target = tf.placeholder(tf.float32, [None, m])

# Network architecture
N = 150
rnn_units = tf.nn.rnn_cell.GRUCell(N)
rnn_output, _ = tf.nn.dynamic_rnn(rnn_units, inputs, dtype=tf.float32)

# Ignore all but the last timesteps
last = tf.gather(rnn_output, T - 1, axis=1)

# Fully connected layer
logits = tf.layers.dense(last, m, activation=None)
# Output mapped to probabilities by softmax
prediction = tf.nn.softmax(logits)
# Error function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=target, logits=logits))
# 0-1 loss; compute most likely class and compare with target
accuracy = tf.equal(tf.argmax(logits, 1), tf.argmax(target, 1))
# Average 0-1 loss
accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
# Optimizer
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", accuracy)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# if True:
    # Create session and initialize variables
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    summary_writer = tf.summary.FileWriter("my_test_model", graph=tf.get_default_graph())
    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state("my_test_model")
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(ckpt.model_checkpoint_path)
        i_stopped = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
        print('No checkpoint file found!')
        i_stopped = 0
    sess.graph.finalize()
    # Do the learning
    for i in range(epochs):
        sess.run(train_step, feed_dict={inputs: train_X, target: train_Y})
        _, c, summary = sess.run([train_step, loss, merged_summary_op],feed_dict={inputs: train_X, target: train_Y})
        summary_writer.add_summary(summary, epochs)
        if (i + 1) % 10 == 0:
            tmp_loss, tmp_acc = sess.run([loss, accuracy], feed_dict={inputs: train_X, target: train_Y})
            tmp_acc_test = sess.run(accuracy, feed_dict={inputs: test_X, target: test_Y})
            print(i + 1, 'Loss:', tmp_loss, 'Accuracy, train:', tmp_acc, ' Accuracy, test:', tmp_acc_test)
            checkpoint_path = os.path.join("my_test_model", 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=i)

with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "my_test_model/model.ckpt")
  print("Model restored.")
  # Check the values of the variables
  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())
