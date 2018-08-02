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
