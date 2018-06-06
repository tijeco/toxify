# toxify
deep learning approach to classify animal toxins


# Getting started

Convert fasta protein files to csv files.


Set up toxify conda environment
```
conda env create -f requirements.yml
```
Activate the environment
```
source activate toxify
```
install toxify
```
cd toxify/
python setup.py install
```

create features
```
toxify --create-features dir_with_fastaFile/
```

prep csv

```
toxify --prep-csv dir_with_fastaFile/all.combined.csv
```

generate predictions

```
toxify tensorflow -predict dir_with_fastaFile/all.combined.csv.tf.csv -model path/to/model
```
