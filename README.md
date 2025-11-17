# nucleate-hackathon-2025

Nucleate hackathon Novartis challenge repository.

## Scientific questions

- Can a model predict the effect of an
  unseen perturbation (compound,
  concentration) in a given cell line
- Can a model predict the effect of an seen
  perturbation in an unseen cell line?

## Structure

Overall project structure.

```bash
├── README.md
├── LICENSE
├── models
│   ├── mlp.ipynb
│   ├── train.ipynb
│   └── mlp_model.py
├── preprocessing
│   ├── dataset-1
│   │   ├── cas-to-smiles
│   │   │   ├── CAS_to_smiles.tsv
│   │   │   ├── cas-to-smiles-2.ipynb
│   │   │   ├── CAS-to-SMILES.ipynb
│   │   ├── cell-line-1
│   │   │   ├── dataset-1_10um.ipynb
│   │   │   └── read-data.ipynb
│   │   ├── cell-line-2
│   │   │   └── read-data.ipynb
│   │   ├── chemical-embeddings
│   │   │   ├── molformer-embeddings.tsv
│   │   │   └── molformer.ipynb
│   └── dataset-2
│       ├── drug4k_smiles-embeddings.tsv
│       ├── MoABox_compounds_metadata.txt
│       ├── molformer.ipynb
│       └── morgan-fingerprint.ipynb
├── presentation
│   └── novaice-results.pdf
```

## Datasets

- **dataset-1**: [HiMAPSeq](https://doi.org/10.1038/s41592-025-02781-5) of 13,221 compounds across 93,664 perturbations in 2 cell lines.
- **dataset-2**: [Novartis DrugSeq Data](https://github.com/Novartis/DRUG-seq/tree/main/data/Novartis_drugseq_U2OS_MoABox) across 4000 compounds in 1 cell line.

## Approach

We focused on two key scientific questions:

1. Predicting the effect of unseen compounds at a single concentration in one cell line.
2. Transferring the learnt effects in one cell line onto a second cell line while minimizing the screening effort in the second cell line. This application of virtuall screening would significantly reduce costs and increase throughput in screening efforts.

## Methods

Key steps we took:

### Preprocessing

1. Streamline drug identifiers in datasets to smiles strings
2. Convert datasets to [`anndata` objects](https://github.com/scverse/anndata.git) for streamlined processing

### Data preparation

1. Compute chemical embeddings with a chemical foundation model [molformer](https://github.com/IBM/molformer.git).
2. Streamline gene expression data to log1p normalized counts

### Model building

1. Create a scVI-inspired probabilistic model [novaice](https://github.com/lucas-diedrich/novaice.git) that takes chemical embeddings as input and predicts the parameters of the transcript count distribution. **This allows us to estimate model uncertainty for every transcript in each perturbation condition**.

### Training

1. **Perturbation effect prediction** Train various small-scale models (MLPs, MLPs with bottlenecks) to predict log1p gene expression. We build a tensorboard-viewer to quickly iterate through embeddings inputs, model types + hyperparameters
2. **Transfer learning approach** Train model on full data of cell line 1, freeze weights and only fine tune final linear layer on a small subset of data in cell line 2.

### Evaluation

1. Evaluation on **$R^2$ of measured and predicted log1p normalized counts**
2. Evaluation on **directionality of predicted and observed logfoldchanges**
3. Comparison of **prediction in train set, test set and against naive baseline** (mean gene expression in control [DMSO] condition)

## Results

Have a look at our [final presentation](./presentation/novaice-results.pdf)

## Contributors

@lucas-diedrich, @mlorenz49, @esemsc-3b60f913
