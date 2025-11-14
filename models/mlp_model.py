import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """
    Multi-Layer Perceptron for drug response prediction (regression).
    
    This model takes molecular descriptors/fingerprints as input and predicts
    gene expression values as output.
    """
    
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256, 128], dropout_rate=0.3):
        """
        Initialize the MLP model for drug-induced gene expression prediction.

        Args:
            input_dim (int): Number of input features (molecular descriptors/fingerprints).
            output_dim (int): Number of output features (gene expression values to predict).
            hidden_dims (list): List of hidden layer dimensions. Default: [512, 256, 128].
            dropout_rate (float): Dropout probability for regularization. Default: 0.3.
        """
        super(SimpleMLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build the network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation for regression)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
                             containing molecular descriptors.
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
                         containing predicted gene expression values.
        """
        return self.network(x)


class DrugResponseDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for drug response prediction.
    
    This dataset handles drug molecular descriptors and corresponding gene expression data.
    """
    
    def __init__(self, drug_features, gene_expression, drug_ids=None):
        """
        Initialize the dataset.
        
        Args:
            drug_features (np.ndarray or torch.Tensor): Molecular descriptors/fingerprints 
                                                        of shape (n_samples, n_features).
            gene_expression (np.ndarray or torch.Tensor): Gene expression values 
                                                          of shape (n_samples, n_genes).
            drug_ids (list, optional): List of drug identifiers for tracking.
        """
        self.drug_features = torch.FloatTensor(drug_features) if not isinstance(drug_features, torch.Tensor) else drug_features
        self.gene_expression = torch.FloatTensor(gene_expression) if not isinstance(gene_expression, torch.Tensor) else gene_expression
        self.drug_ids = drug_ids
        
        assert len(self.drug_features) == len(self.gene_expression), \
            "Drug features and gene expression must have the same number of samples"
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.drug_features)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            tuple: (drug_features, gene_expression) or 
                   (drug_features, gene_expression, drug_id) if drug_ids provided.
        """
        if self.drug_ids is not None:
            return self.drug_features[idx], self.gene_expression[idx], self.drug_ids[idx]
        return self.drug_features[idx], self.gene_expression[idx]


def create_data_loaders(drug_features, gene_expression, drug_ids=None, 
                       train_size=0.8, batch_size=32, random_state=42):
    """
    Create train and validation data loaders.
    
    Args:
        drug_features (np.ndarray): Molecular descriptors of shape (n_samples, n_features).
        gene_expression (np.ndarray): Gene expression values of shape (n_samples, n_genes).
        drug_ids (list, optional): List of drug identifiers.
        train_size (float): Proportion of data to use for training. Default: 0.8.
        batch_size (int): Batch size for data loaders. Default: 32.
        random_state (int): Random seed for reproducibility. Default: 42.
    
    Returns:
        tuple: (train_loader, val_loader, train_indices, val_indices)
    """
    from sklearn.model_selection import train_test_split
    
    # Split indices
    indices = list(range(len(drug_features)))
    train_idx, val_idx = train_test_split(
        indices, 
        train_size=train_size, 
        random_state=random_state
    )
    
    # Create datasets
    train_dataset = DrugResponseDataset(
        drug_features[train_idx], 
        gene_expression[train_idx],
        [drug_ids[i] for i in train_idx] if drug_ids is not None else None
    )
    
    val_dataset = DrugResponseDataset(
        drug_features[val_idx], 
        gene_expression[val_idx],
        [drug_ids[i] for i in val_idx] if drug_ids is not None else None
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, train_idx, val_idx


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on (cuda/cpu).
    
    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    
    for batch_data in train_loader:
        # Handle both 2-tuple and 3-tuple returns from dataset
        if len(batch_data) == 3:
            batch_features, batch_targets, _ = batch_data
        else:
            batch_features, batch_targets = batch_data
            
        batch_features = batch_features.to(device)
        batch_targets = batch_targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(batch_features)
        loss = criterion(predictions, batch_targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: The PyTorch model to validate.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to validate on (cuda/cpu).
    
    Returns:
        float: Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_data in val_loader:
            # Handle both 2-tuple and 3-tuple returns from dataset
            if len(batch_data) == 3:
                batch_features, batch_targets, _ = batch_data
            else:
                batch_features, batch_targets = batch_data
                
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            predictions = model(batch_features)
            loss = criterion(predictions, batch_targets)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)
