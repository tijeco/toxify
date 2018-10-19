import tensorflow as tf
import numpy as np
import pandas as pd
from random import shuffle
import random
import os
import sys
from itertools import groupby
import argparse
import toxify
import toxify.fifteenmer as fm
import toxify.protfactor as pf
import toxify.seq2window as sw


model_dir = os.path.abspath(toxify.__file__).replace("__init__.py","models")
print(model_dir)
class ParseCommands(object):

    def __init__(self):

        parser = argparse.ArgumentParser(
            description='Pretends to be git',
            usage='''git <command> [<args>]

The most commonly used git commands are:
   commit     Record changes to the repository
   fetch      Download objects and refs from another repository''')

        parser.add_argument('command', help='Subcommand to run')
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        self.args = parser.parse_args(sys.argv[1:2])
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def train(self):
        parser = argparse.ArgumentParser(
            description='Record changes to the repository')
        # prefixing the argument with -- means it's optional
        parser.add_argument('-pos',action='append',nargs='*')
        parser.add_argument('-neg',action='append',nargs='*')
        parser.add_argument('-window',type = int,default = 15)
        parser.add_argument('-maxLen',type = int,default = 100)
        parser.add_argument('-units',type = int,default = 150)
        # now that we're inside a subcommand, ignore the first
        # TWO argvs, ie the command (git) and the subcommand (commit)
        args = parser.parse_args(sys.argv[2:])
        print('Running toxify train\n positive data:' , args.pos,'\n negative data:' , args.neg)
        self.args = args
        return(self.args)

    def predict(self):
        parser = argparse.ArgumentParser(
            description='Predicts venom probabilities')
        # NOT prefixing the argument with -- means it's not optional
        parser.add_argument('sequences')
        args = parser.parse_args(sys.argv[2:])
        print('Running toxify predict\n input data:' , args.sequences)
        self.args = args
        return(self.args)



def main():
    # print(fm.joke())
    # ParseCommands()
    tox_args = ParseCommands().args
    # print(tox_args)
    if hasattr(tox_args,"sequences"):
        # print(tox_args.sequences)
        """
        HERE needs to be a new way of converting fasta proteins to atchley factors, seq2window funcs
        """
        predictions_dir = tox_args.sequences +"_toxify_predictions"
        if not os.path.exists(predictions_dir):
            os.makedirs(predictions_dir)
        protein_pd = sw.fa2pd(tox_args.sequences,0,500)
        print("Number of input proteins:",protein_pd.shape)
        print(protein_pd["sequences"])
        # this will produce np array of fifteenmer seqs
        use15mer = False
        if use15mer:

            proteins = fm.ProteinWindows(tox_args.sequences)
            protein_15mer = proteins.data

            protein_vectors_np = pf.ProteinVectors(protein_15mer).data
            np.save(predictions_dir+"/protein_vectors.npy",protein_vectors_np)

            os.system("saved_model_cli run --dir "+model_dir+"  --tag_set serve --signature_def serving_default --inputs inputs="+predictions_dir+"/protein_vectors.npy  --outdir "+predictions_dir)

            prediction_np = np.load(predictions_dir+"/predictions.npy")
            prediction_15mer = np.hstack((protein_15mer,prediction_np))
            prediction_15mer_df = pd.DataFrame(prediction_15mer).drop(4,1)
            prediction_15mer_df.columns = [ 'header','15mer','sequence','venom_probability']
            columnsTitles=['header','15mer','venom_probability','sequence']
            prediction_15mer_df=prediction_15mer_df.reindex(columns=columnsTitles)
            prediction_15mer_outfile = predictions_dir+"/predictions_15mer.csv"
            prediction_15mer_df.to_csv(prediction_15mer_outfile,index=False)
            prediction_proteins = fm.regenerate(prediction_15mer_df)
            prediction_proteins_outfile = predictions_dir+"/predictions_proteins.csv"
            prediction_proteins.to_csv(prediction_proteins_outfile,index=False)


    # here we are given a list of positive fasta files and a list of negative fasta files
    elif hasattr(tox_args,"pos") and hasattr(tox_args,"neg"):
        max_seq_len = tox_args.maxLen
        window_size = tox_args.window
        N_units = tox_args.units
        # here we are given a list of positive fasta files and a list of negative fasta files
        print(tox_args.pos)
        print(tox_args.neg)
        (train_seqs,test_seqs) = sw.seqs2train(tox_args.pos,tox_args.neg,window_size,max_seq_len)
        print("TEST:")
        # print(test_seqs)

        training_dir = "training_data/max_len_" + str(max_seq_len) + "/window_"+str(window_size)+"/units_"+str(N_units)+"/"
        if not os.path.exists(training_dir):
            os.makedirs(training_dir)
            print("writing to: "+training_dir+"testSeqs.csv")
            test_seqs_pd = pd.DataFrame(test_seqs)
            if window_size:
                test_seqs_pd.columns = ['header', 'kmer','sequence','label']
            else:
                test_seqs_pd.columns = ['header', 'sequence','label']
            test_seqs_pd.to_csv(training_dir+"testSeqs.csv",index= False)

            test_mat = []
            test_label_mat = []
            for row in test_seqs:
                seq = row[-2]
                label = float(row[-1])
                # print(label)
                # print(bool(label),label,"row:",row)
                if label:
                    test_label_mat.append([1,0])
                else:
                    test_label_mat.append([0,1])
                test_mat.append(sw.seq2atchley(seq,window_size,max_seq_len))
            test_label_np = np.array(test_label_mat)
            test_np = np.array(test_mat)

            train_mat = []
            train_label_mat = []
            for row in train_seqs:
                seq = row[-2]
                train_mat.append(sw.seq2atchley(seq,window_size,max_seq_len))
                label = float(row[-1])
                # print(label)
                if label:

                    train_label_mat.append([1,0])
                else:
                    train_label_mat.append([0,1])
            train_label_np = np.array(train_label_mat)
            train_np = np.array(train_mat)

            np.save(training_dir+"testData.npy",test_np)
            np.save(training_dir+"testLabels.npy",test_label_np)
            np.save(training_dir+"trainData.npy",train_np)
            np.save(training_dir+"trainLabels.npy",train_label_np)
            # test_X = test_np
            # test_Y = test_label_np
            # train_X = train_np
            # train_Y = train_label_np
        # else:


        # train_X = np.load('/media/brewerlab/BigRAID/Jeffrey/toxify/sequence_data/training_data/training_np.npy') #(7352, 128, 9)
        # test_X  = np.load('/media/brewerlab/BigRAID/Jeffrey/toxify/sequence_data/training_data/test_np.npy')
        # train_Y = np.load('/media/brewerlab/BigRAID/Jeffrey/toxify/sequence_data/training_data/training_labels.npy') #(7352, 6)
        # test_Y  = np.load('/media/brewerlab/BigRAID/Jeffrey/toxify/sequence_data/training_data/test_labels.npy')

        test_X = np.load(training_dir+"testData.npy")
        test_Y = np.load(training_dir+"testLabels.npy")
        train_X = np.load(training_dir+"trainData.npy")
        train_Y = np.load(training_dir+"trainLabels.npy")

        print("train_X.shape:",train_X.shape)
        print("train_Y.shape:",train_Y.shape)
        # Parameters
        n = train_X.shape[0]  # Number of training sequences
        print(n) #7352
        n_test = train_Y.shape[0]  # Number of test sequences
        # print(n_test) #7352
        m = train_Y.shape[1]  # Output dimension
        print(m) #6
        d = train_X.shape[2]  # Input dimension
        print(d) #9
        T = train_X.shape[1]  # Sequence length
        epochs = 300
        # batch_size = 100

        lr = 0.01  # Learning rate

        # Placeholders
        inputs = tf.placeholder(tf.float32, [None, None, d])
        target = tf.placeholder(tf.float32, [None, m])

        # Network architecture

        rnn_units = tf.nn.rnn_cell.GRUCell(N_units)
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
        model_dir = training_dir+"models"
        # if True:
            # Create session and initialize variables
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            summary_writer = tf.summary.FileWriter(model_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver()

            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print(ckpt.model_checkpoint_path)
                i_stopped = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            else:
                print('No checkpoint file found!')
                i_stopped = 0

            #tf.saved_model.simple_save(sess, model_dir+"/saved_model/", inputs={"inputs":inputs,"target":target},outputs={"predictions":prediction})
            #sess.graph.finalize()

            # Do the learning
            for i in range(epochs):
                sess.run(train_step, feed_dict={inputs: train_X, target: train_Y})
                _, c, summary = sess.run([train_step, loss, merged_summary_op],feed_dict={inputs: train_X, target: train_Y})
                summary_writer.add_summary(summary, epochs)
                if (i + 1) % 10 == 0:
                    tmp_loss, tmp_acc = sess.run([loss, accuracy], feed_dict={inputs: train_X, target: train_Y})
                    tmp_acc_test = sess.run(accuracy, feed_dict={inputs: test_X, target: test_Y})
                    print(i + 1, 'Loss:', tmp_loss, 'Accuracy, train:', tmp_acc, ' Accuracy, test:', tmp_acc_test)
                    checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=i)
            tf.saved_model.simple_save(sess, model_dir+"/saved_model/", inputs={"inputs":inputs,"target":target},outputs={"predictions":prediction})
            sess.graph.finalize()

        if False:
            with tf.Session() as sess:
              # Restore variables from disk.
              saver.restore(sess, model_dir+"/model.ckpt")
              print("Model restored.")
              # Check the values of the variables
              print("v1 : %s" % v1.eval())
              print("v2 : %s" % v2.eval())






        # training_file = "sequence_data/training_np.npy"
        # test_file =  "sequence_data/test_np.npy"
        # if training_file.exists() and test_file.exists():
        #     training_np = np.load("sequence_data/training_np.npy")
        #     test_np =  np.load("sequence_data/test_np.npy")
        # else:
        #     print("Error, sequence_data/training_np.npy doesn't exist\nuse window.py and factor.py to generate them")

#             """
#             def fasta_iter(fasta_name):
#                 with open(fasta_name) as fh:
#                     faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))
#                     for header in faiter:
#                         headerStr = header.__next__()[1:].strip().split()[0]#Entire line, add .split[0] for just first column
#                         seq = "".join(s.strip() for s in faiter.__next__())
#                         yield (headerStr, seq)
#
#             def seqs2train(pos,neg):
#                 mat=[]
#                 for i in pos[0]:
#                     pos_iter = fasta_iter(i)
#                     for ff in pos_iter:
#                         headerStr, seq = ff
#                         if len(seq) < 101 and len(seq) > 14:
#                             mat.append([headerStr,seq,1])
#                 for i in neg[0]:
#                     neg_iter = fasta_iter(i)
#                     for ff in neg_iter:
#                         headerStr, seq = ff
#                         if len(seq) < 101 and len(seq) > 14:
#                             mat.append([headerStr,seq,0])
#                 # print(mat)
#                 mat_np = np.array(mat)
#                 indices = np.random.permutation(mat_np.shape[0])
#                 training_idx, test_idx = indices[:int(mat_np.shape[0]*0.8)], indices[int(mat_np.shape[0]*0.8):]
#                 training_np, test_np = mat_np[training_idx,:], mat_np[test_idx,:]
#                 return (training_np,test_np)
#             print(seqs2train(tox_args.pos,tox_args.neg)[0][:,1])
#
#             def seq2atchley(s):
#                 seqList = []
#                 for aa in s:
#                     for factor in factorDict[aa]:
#                         seqList.append([factor])
#                 return seqList
#             print("MENDEL")
#             print(seq2atchley("MENDEL"))
#
#
#
#
#             training_seqs = seqs2train(tox_args.pos,tox_args.neg)[0]
#             test_seqs = seqs2train(tox_args.pos,tox_args.neg)[1]
#             np.save(training_seqs,"sequence_data/training_np.npy")
#             np.save(test_seqs,"sequence_data/test_np.npy")
# """
        # returns np array with four columns
        # 1. header 2. kmerNum 3. sequence 4. label
