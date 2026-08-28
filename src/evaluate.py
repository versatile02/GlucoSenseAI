import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve
)

from src.model import DiabetesANN

def evaluate_pipeline(
    data_npz_path: str = "models/split_data.npz",
    model_weights_path: str = "models/best_model.pth",
    metadata_json_path: str = "models/model_metadata.json",
    raw_csv_path: str = "data/diabetes.csv",
    output_dir: str = "reports/figures",
    results_json_path: str = "models/evaluation_results.json"
):
    print("Starting evaluation...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load split datasets
    if not os.path.exists(data_npz_path):
        raise FileNotFoundError(f"Split data file {data_npz_path} not found. Run training first.")
        
    data = np.load(data_npz_path)
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # 2. Load model
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Model weights {model_weights_path} not found. Run training first.")
        
    model = DiabetesANN(input_dim=8, hidden_dim1=32, hidden_dim2=16)
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()
    
    # 3. Predict on Test Set
    X_test_tensor = torch.FloatTensor(X_test)
    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = torch.sigmoid(logits).numpy().flatten()
        preds = (probs >= 0.5).astype(int)
        
    # 4. Calculate metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    
    print("\n--- Test Set Metrics ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # Save metrics
    results = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'roc_auc': float(auc),
        'confusion_matrix': cm.tolist()
    }
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved metrics to {results_json_path}")
    
    # 5. Generate visualizations
    sns.set_theme(style="whitegrid")
    
    # Visual 1: Correlation Heatmap of raw features
    if os.path.exists(raw_csv_path):
        df_raw = pd.read_csv(raw_csv_path)
        plt.figure(figsize=(10, 8))
        corr = df_raw.corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, square=True)
        plt.title("Feature Correlation Heatmap", fontsize=14, pad=15)
        plt.tight_layout()
        corr_path = os.path.join(output_dir, "correlation_heatmap.png")
        plt.savefig(corr_path, dpi=300)
        plt.close()
        print(f"Saved feature correlation heatmap to {corr_path}")
        
        # Visual 1b: Class Distribution
        plt.figure(figsize=(6, 4))
        sns.countplot(x='Outcome', data=df_raw, hue='Outcome', palette="Set2", legend=False)
        plt.xticks([0, 1], ["Non-Diabetic (0)", "Diabetic (1)"])
        plt.xlabel("Class Outcome")
        plt.ylabel("Patient Count")
        plt.title("Class Distribution (Clinical Dataset)", fontsize=12, pad=10)
        plt.tight_layout()
        dist_path = os.path.join(output_dir, "class_distribution.png")
        plt.savefig(dist_path, dpi=300)
        plt.close()
        
    # Visual 2: Training vs Validation curves (Loss and Accuracy)
    if os.path.exists(metadata_json_path):
        with open(metadata_json_path, 'r') as f:
            meta = json.load(f)
        history = meta.get('history', {})
        
        epochs_range = range(1, len(history.get('train_loss', [])) + 1)
        
        # Plot Loss
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
        plt.plot(epochs_range, history['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2, linestyle='--')
        plt.xlabel('Epochs')
        plt.ylabel('BCE Loss')
        plt.title('Training & Validation Loss')
        plt.legend()
        
        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history['train_acc'], label='Train Acc', color='#2ca02c', linewidth=2)
        plt.plot(epochs_range, history['val_acc'], label='Val Acc', color='#d62728', linewidth=2, linestyle='--')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training & Validation Accuracy')
        plt.legend()
        
        plt.tight_layout()
        curves_path = os.path.join(output_dir, "learning_curves.png")
        plt.savefig(curves_path, dpi=300)
        plt.close()
        print(f"Saved learning curves to {curves_path}")
        
    # Visual 3: Confusion Matrix heatmap
    plt.figure(figsize=(6, 5))
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = [f"{value:0.0f}" for value in cm.flatten()]
    group_percentages = [f"{value:.2%}" for value in cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, 
                xticklabels=['Non-Diabetic', 'Diabetic'], 
                yticklabels=['Non-Diabetic', 'Diabetic'])
    plt.xlabel('Predicted Label', fontsize=11, labelpad=10)
    plt.ylabel('Actual Label', fontsize=11, labelpad=10)
    plt.title('Confusion Matrix on Test Set', fontsize=13, pad=15)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix heatmap to {cm_path}")
    
    # Visual 4: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#9467bd', label=f'ANN (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (FPR)', fontsize=11, labelpad=10)
    plt.ylabel('True Positive Rate (TPR)', fontsize=11, labelpad=10)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=13, pad=15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved ROC curve to {roc_path}")
    
    print("\nEvaluation successfully finished.")

if __name__ == "__main__":
    evaluate_pipeline()
