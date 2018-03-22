# toxify
deep learning approach to classify animal toxins


# Getting started

Convert fasta protein files to csv files.



```
snakemake --snakefile create_features/Snakefile -d dir/with/fastas/ --cores
```

make sure that fasta files end in .fa


```
python toxify.py -predict dir/with/fastas/all.combined.to_csv
```
