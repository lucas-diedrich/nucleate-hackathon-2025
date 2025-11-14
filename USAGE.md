# Drug Response Prediction using MLP

## Project Overview

This project implements a Multi-Layer Perceptron (MLP) to predict gene expression changes induced by drug treatments. The model takes molecular descriptors (fingerprints) of drugs as input and predicts the resulting gene expression profile.

## Model Architecture

The `SimpleMLP` class in `models/mlp_model.py` provides:

- **Input**: Molecular descriptors/fingerprints (e.g., 256-bit Morgan fingerprints)
- **Architecture**: 
  - 3 hidden layers with configurable dimensions (default: 512 → 256 → 128)
  - Batch normalization after each hidden layer
  - ReLU activations
  - Dropout for regularization (default: 0.3)
- **Output**: Predicted gene expression values for all genes

### Key Components

1. **SimpleMLP**: Main neural network model
2. **DrugResponseDataset**: PyTorch dataset class for handling drug-gene expression pairs
3. **Training utilities**: `train_epoch()`, `validate()`, `create_data_loaders()`

## Data Structure

### Current Data
- **Gene Expression**: `MCE_Bioactive_Compounds_HEK293T_10μM_Counts.csv`
  - Contains RNA-seq count data for genes (rows) across different drug treatments (columns)
- **Metadata**: `MCE_Bioactive_Compounds_HEK293T_10μM_MetaData.csv`
  - Contains sample information including drug IDs and treatment conditions

### Molecular Descriptors ✓
- **Source**: `unique_compounds.csv` with SMILES strings
- **Descriptor Type**: Morgan Fingerprints (ECFP4)
  - Radius: 2
  - Bits: 2048
  - Generated using RDKit from SMILES
- **Alternative options to explore**:
  - MACCS keys (166 structural keys)
  - RDKit molecular descriptors (physicochemical properties)
  - Different fingerprint parameters (radius, bit length)
  - Graph neural network embeddings

## Usage

### 1. Data Preparation

```python
# Load gene expression data
counts_df = pd.read_csv('./data/MCE_Bioactive_Compounds_HEK293T_10μM_Counts.csv', 
                        index_col=0, skiprows=1)
metadata_df = pd.read_csv('./data/MCE_Bioactive_Compounds_HEK293T_10μM_MetaData.csv', 
                          skiprows=1)

# Load compounds with SMILES data
compounds = pd.read_csv('./data/unique_compounds.csv')

# Generate Morgan fingerprints from SMILES
from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)

# Create descriptor dictionary
catalog_to_smiles = dict(zip(compounds['Catalog Number'], compounds['smiles']))
drug_descriptor_dict = {
    drug_id: smiles_to_fingerprint(catalog_to_smiles[drug_id])
    for drug_id in drug_treatments if drug_id in catalog_to_smiles
}
```

### 2. Model Initialization

```python
from mlp_model import SimpleMLP

model = SimpleMLP(
    input_dim=2048,     # Number of molecular features (Morgan FP bits)
    output_dim=3407,    # Number of genes
    hidden_dims=[512, 256, 128],
    dropout_rate=0.3
)
```

### 3. Training

```python
from mlp_model import create_data_loaders, train_epoch, validate

# Create data loaders
train_loader, val_loader, train_idx, val_idx = create_data_loaders(
    drug_features=drug_features,
    gene_expression=gene_expression_data,
    train_size=0.8,
    batch_size=32
)

# Training loop
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)
```

### 4. Prediction

```python
model.eval()
with torch.no_grad():
    predictions = model(drug_features_tensor)
```

## Next Steps

1. **Train the Model**:
   - Run the training loop in the notebook
   - Monitor training and validation losses
   - The model is now using real molecular descriptors! ✓

2. **Model Improvements**:
   - Experiment with different fingerprint types (MACCS, RDKit FP)
   - Try different fingerprint parameters (radius, bit length)
   - Add molecular descriptors (MW, LogP, TPSA, etc.)
   - Implement cross-validation
   - Add attention mechanisms or try graph neural networks

3. **Evaluation**:
   - Calculate R² scores for prediction quality
   - Analyze which genes are most predictable
   - Identify important molecular features
   - Perform ablation studies

4. **Deployment**:
   - Save trained model for inference
   - Create prediction pipeline for new drugs
   - Develop visualization tools for results

## File Structure

```
nucleate-hackathon-2025/
├── models/
│   └── mlp_model.py          # MLP architecture and training utilities
├── data/
│   ├── MCE_Bioactive_Compounds_HEK293T_10μM_Counts.csv
│   └── MCE_Bioactive_Compounds_HEK293T_10μM_MetaData.csv
├── mlp.ipynb                  # Main notebook with full pipeline
├── prep.ipynb                 # Data preprocessing notebook
└── USAGE.md                   # This file
```

## Dependencies

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib rdkit
```

## Model Performance Metrics

The model is evaluated using:
- **MSE Loss**: Mean Squared Error for regression
- **R² Score**: Coefficient of determination per gene
- **MAE**: Mean Absolute Error per gene

Current performance with placeholder data will not be meaningful. Real performance metrics should be calculated after replacing with actual molecular descriptors.

**Update**: The model now uses real Morgan fingerprints generated from SMILES! Training should produce meaningful results.
