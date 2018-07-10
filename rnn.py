#!/Users/jeffcole/miniconda3/bin/python

"""Time series classification example.

Supplementary code for:
D. Hafner and C. Igel. "Signal Processing with Recurrent Neural Networks in TensorFlow"

Data and problem definition taken from:
D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz. "A public domain dataset for human activity recognition using smartphones". In 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN), pages 437–442. i6doc.com, 2013.
"""

import tensorflow as tf
import numpy as np
from random import shuffle
import os

if False:
    inputLength = 16

    train_input = ['{0:016b}'.format(i) for i in range(2**inputLength)]
    inputSize = len(train_input)
    shuffle(train_input)
    print(train_input)
    subsetList = np.array(list(train_input[0])).reshape(4,4).astype(int)
    print(subsetList)
    num_zero = np.count_nonzero(subsetList==0)
    if num_zero%2!=0:
        print([0,1])
    else:
        print([1,0])
    print("JESUS",np.count_nonzero(subsetList==0))
    # print(np.sum(~subsetList.any(1)))

    print(len(train_input))
    # train_np = np.array(train_input)
    # print(train_np.shape)
    tmpList = []
    print()
    for i in range(len(train_input)):
        tmpList.append(np.array(list(train_input[i])).reshape(4,4).astype(int))
    print(np.array(tmpList).shape)
    train_np = np.array(tmpList)

    train_labels = []
    for i in train_np:
        num_zero = np.count_nonzero(subsetList==0)
        if num_zero%2!=0:
            train_labels.append(np.array([0,1]))
        else:
            train_labels.append(np.array([1,0]))
    train_labels = np.array(train_labels)
    print(train_labels.shape)

    train_output = train_labels
    train_input = train_np
    NUM_EXAMPLES = 10000
    test_input = train_input[NUM_EXAMPLES:]
    test_output = train_output[NUM_EXAMPLES:] #everything beyond 10,000

    train_input = train_input[:NUM_EXAMPLES]
    train_output = train_output[:NUM_EXAMPLES] #till 10,000

# Load data
if True:
    train_X = np.load('training_data/training_np.npy') #(7352, 128, 9)
    test_X  = np.load('training_data/test_np.npy')
    train_Y = np.load('training_data/training_labels.npy') #(7352, 6)
    test_Y  = np.load('training_data/test_labels.npy')
    print(train_Y)
# train_X = train_input
# test_X = test_input
# train_Y = train_output
# test_Y = test_output
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
if True:
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


graph = tf.Graph()
with restored_graph.as_default():
    with tf.Session as sess:
        tf.saved_model.loader.load(
            sess,
            [tag_constants.SERVING],
        'path/to/your/location/',
        )
        batch_size_placeholder = graph.get_tensor_by_name('batch_size_placeholder:0')
        features_placeholder = graph.get_tensor_by_name('features_placeholder:0')
        labels_placeholder = graph.get_tensor_by_name('labels_placeholder:0')
        prediction = restored_graph.get_tensor_by_name('dense/BiasAdd:0')

        sess.run(prediction, feed_dict={
            batch_size_placeholder: some_value,
            features_placeholder: some_other_value,
            labels_placeholder: another_value
        })
