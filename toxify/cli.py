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
    print(fm.joke())
    # ParseCommands()
    tox_args = ParseCommands().args
    # print(tox_args)
    if hasattr(tox_args,"sequences"):
        print(tox_args.sequences)
        predictions_dir = tox_args.sequences +"_toxify_predictions"
        if not os.path.exists(predictions_dir):
            os.makedirs(predictions_dir)
        # execute fm.seq2np(sequences)
        # returns np array with three columns
        # 1. header 2. kmerNum 3. sequence

        proteins = fm.ProteinWindows(tox_args.sequences)
        protein_15mer = proteins.data
        print(protein_15mer)
        # import protfactor as pf
        # execute pf.ProteinVectors(protein_15mer)
        # returns np.array (N,5,15)
        print(pf.ProteinVectors(protein_15mer).data.shape)


        protein_vectors_np = pf.ProteinVectors(protein_15mer).data
        np.save(predictions_dir+"/protein_vectors.npy",protein_vectors_np)
        # os.system("pwd")
        os.system("saved_model_cli run --dir "+model_dir+"  --tag_set serve --signature_def serving_default --inputs inputs="+predictions_dir+"/protein_vectors.npy  --outdir "+predictions_dir)
        prediction_np = np.load(predictions_dir+"/predictions.npy")
        print("data:",protein_15mer.shape)
        print("output:",prediction_np.shape)
        # print(protein_15mer[0])
        prediction_15mer = np.hstack((protein_15mer,prediction_np))
        # print(prediction_15mer)
        prediction_15mer_df = pd.DataFrame(prediction_15mer).drop(4,1)

        prediction_15mer_df.columns = [ 'header','15mer','sequence','venom_probability']

        columnsTitles=['header','15mer','venom_probability','sequence']
        prediction_15mer_df=prediction_15mer_df.reindex(columns=columnsTitles)
        print(prediction_15mer_df)
        prediction_15mer_outfile = predictions_dir+"/predictions_15mer.csv"
        prediction_15mer_df.to_csv(prediction_15mer_outfile,index=False)
        prediction_proteins = fm.regenerate(prediction_15mer_df)
        prediction_proteins_outfile = predictions_dir+"/predictions_proteins.csv"
        prediction_proteins.to_csv(prediction_proteins_outfile,index=False)

        # now pass prediction_15mer_df to fm.regenerate(prediction_15mer_df)
        # output csv file with 6 columns
        # 1. header 2. n_15mers 3. mean_venom_probability 4. median_venom_probability 5. sd_venom_probability 6. sequence



        # run command_line_tf using os on pf.ProteinVectors(protein_15mer)
        # open the returned np.array, recombine with headers, create file with three columns
        # 1. header 2. 15mer 3. venom probability



    elif hasattr(tox_args,"pos") and hasattr(tox_args,"neg"):
        print(tox_args.pos)
        print(tox_args.neg)
        # returns np array with four columns
        # 1. header 2. kmerNum 3. sequence 4. label
