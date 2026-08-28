import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.preprocessing import load_raw_data, DiabetesPreprocessor
from src.model import DiabetesANN

def set_seeds(seed: int = 42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(
    csv_path: str = "data/diabetes.csv",
    models_dir: str = "models",
    lr: float = 0.005,
    batch_size: int = 32,
    epochs: int = 1000,
    patience: int = 50,
    seed: int = 42
):
    # Set seeds
    set_seeds(seed)
    
    # Create directories
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load raw data
    print("Loading raw data...")
    X, y = load_raw_data(csv_path)
    
    # 2. Train/Validation/Test Split (70% / 15% / 15%)
    # First, split 85% train-val, 15% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=seed, stratify=y
    )
    # Then split train-val into 70% train, 15% val (which is 17.65% of 85%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, random_state=seed, stratify=y_train_val
    )
    
    print(f"Dataset split summary:")
    print(f" - Train: {X_train.shape[0]} samples (Positives: {y_train.sum()})")
    print(f" - Val:   {X_val.shape[0]} samples (Positives: {y_val.sum()})")
    print(f" - Test:  {X_test.shape[0]} samples (Positives: {y_test.sum()})")
    
    # 3. Preprocessing
    print("\nFitting preprocessor on training data...")
    preprocessor = DiabetesPreprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(X_test)
    
    # Save the preprocessor
    preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
    preprocessor.save(preprocessor_path)
    print(f"Saved preprocessor to {preprocessor_path}")
    
    # Save datasets for evaluation
    np.savez(
        os.path.join(models_dir, "split_data.npz"),
        X_train=X_train_scaled, y_train=y_train.values,
        X_val=X_val_scaled, y_val=y_val.values,
        X_test=X_test_scaled, y_test=y_test.values
    )
    print("Saved split datasets to models/split_data.npz")
    
    # 4. Prepare PyTorch DataLoaders
    # Convert numpy arrays to torch tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled), 
        torch.FloatTensor(y_train.values).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled), 
        torch.FloatTensor(y_val.values).unsqueeze(1)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 5. Initialize Model, Loss, Optimizer
    model = DiabetesANN(input_dim=8, hidden_dim1=32, hidden_dim2=16)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Training state
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0
    best_model_weights = None
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print("\nTraining Artificial Neural Network...")
    for epoch in range(epochs):
        # --- Training phase ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
            # Compute accuracy
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_x.size(0)
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_x.size(0)
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_x.size(0)
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        # Save history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Log progress periodically
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{epochs} | "
                  f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
                  f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
            
        # Check early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            best_epoch = epoch + 1
            best_model_weights = model.state_dict().copy()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}! "
                      f"Best epoch was {best_epoch} with Val Loss: {best_val_loss:.4f}")
                break
                
    # Restore best weights and save
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        
    model_path = os.path.join(models_dir, "best_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved best model weights to {model_path}")
    
    # Save training info (hyperparameters & history)
    meta = {
        'input_features': preprocessor.feature_names,
        'num_layers': 3,
        'layers_config': [
            {'type': 'Linear', 'in': 8, 'out': 32, 'activation': 'LeakyReLU', 'batch_norm': True},
            {'type': 'Linear', 'in': 32, 'out': 16, 'activation': 'LeakyReLU', 'batch_norm': True, 'dropout': 0.2},
            {'type': 'Linear', 'in': 16, 'out': 1, 'activation': 'None (logits)'}
        ],
        'optimizer': 'Adam',
        'learning_rate': lr,
        'batch_size': batch_size,
        'best_epoch': best_epoch,
        'epochs_trained': len(history['train_loss']),
        'loss_function': 'BCEWithLogitsLoss',
        'history': history
    }
    
    meta_path = os.path.join(models_dir, "model_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=4)
    print(f"Saved model metadata to {meta_path}")

if __name__ == "__main__":
    train_model()
