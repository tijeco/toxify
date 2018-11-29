# toxify
deep learning approach to classify animal toxins

# Installation

1. Get  the repository for toxify
```bash
git clone https://github.com/tijeco/toxify.git
```

2. Change into toxify repository
```bash
cd toxify
```

3. Create conda environment with dependencies needed to run toxify
```bash
conda env create -f requirements.yml
```
4. Activate toxify conda environment
```bash
source activate toxify_env
```
5. Install toxify to the toxify conda environment
```bash
python setup.py install
```
6. Run toxify
```bash
toxify predict <input.fasta>
```
7. view results in ```<input.fasta>_toxify_predictions/predictions_proteins.csv```
8. Deactivate toxify conda environment
```bash
source deactivate
```

# Custom training dataset

If you wish to train a different model with your own training data, that can be  accomplished with toxify using the ```train``` sub command.

There are a few variables that can be modified in terms of setting up the model.

* ```-pos``` can be followed by a list of protein sequence fasta files that you wish to constitute the positive dataset

* ```-neg``` can be followed by a list of protein sequence fasta files that you wish to constitute the negative dataset

* ```-maxLen``` is the maximum length (integer) of protein seqeunce to be included in the training set

* ```-units``` is the number (integer) of gated recurrent units to be used in the model

* ```-epochs``` is the number (integer) of training epochs to run for the model

* ```lr``` is the learning_rate (float, 0 < ```lr``` < 1)
