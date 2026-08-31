# GlucoSenseAI: Diabetes Prediction Using Artificial Neural Networks

GlucoSenseAI is an educational machine learning application that uses an Artificial Neural Network to estimate diabetes risk from selected clinical and physiological features. The system preprocesses the input data, applies the trained neural network, and presents the resulting prediction through an interactive interface. Model performance and evaluation metrics are provided to make the system easier to understand and reproduce.

---

> [!WARNING]
> **🛑 Clinical Disclaimer:**
> This system is developed strictly as an educational machine-learning demonstration and risk assessment prototype. It does **NOT** represent a clinical diagnostic tool or medical decision support system. The output represents a statistical probability and should not be used as a formal medical diagnosis. Consult a qualified, licensed healthcare professional for medical consults, diagnostics, or treatment plans.

---

## 🚀 Key Features

* **Advanced PyTorch Neural Network:** Built with Batch Normalization and Dropout layers to ensure robust generalizability and prevent overfitting.
* **Robust Preprocessing Pipeline:** Automatically detects and imputes medically implausible values (e.g., zero measurements in Blood Pressure, Glucose, BMI) using training set medians, followed by feature standardization.
* **Interactive Healthcare Web App:** A Streamlit interface tailored for medical parameter input, providing detailed risk scoring, clinical descriptions, and educational analytics.
* **Unit Tested Components:** Fully covered by PyTorch and preprocessor tests to verify shape alignment, imputation accuracy, and input bounds validation.

---

## 📁 Repository Structure

```
GlucoSenseAI/
├── data/
│   └── diabetes.csv           # Raw Clinical Diabetes Dataset
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Imputation and standardization pipeline
│   ├── model.py               # PyTorch neural network definition
│   ├── train.py               # Mini-batch training with early stopping
│   ├── evaluate.py            # Diagnostic metrics and plot generator
│   └── predict.py             # Inference API with input bounds validation
├── models/
│   ├── best_model.pth         # Trained ANN weights checkpoint
│   ├── preprocessor.pkl       # Saved scaler and imputation medians
│   ├── model_metadata.json    # Hyperparameters & train history
│   └── evaluation_results.json# Saved evaluation test set metrics
├── reports/
│   └── figures/               # Automatically generated plots (ROC, CM, Loss, Heatmap)
├── app/
│   └── app.py                 # Streamlit web application
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py       # PyTest automated tests
├── requirements.txt           # Pinned python packages
├── README.md                  # System documentation
└── .gitignore                 # Untracked files configuration
```

---

## 📊 Dataset Information

* **Dataset:** Pima Indians Diabetes Database
* **Records:** 768
* **Input Features:** 8
* **Target Variable:** Outcome
* **Task:** Binary Classification
* **Source:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)

### Input Features:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

### Target:
- Outcome (0 = No diabetes, 1 = Diabetes)

---

## 📊 Preprocessing & Model Configuration

### Data Preprocessing
Medically invalid values represented as `0` in columns `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are imputed using the **overall median** computed solely from the training set (stratified 70/15/15 split). Standard scaling is then fit on training inputs and applied to all inputs during train, validation, and inference to prevent target data leakage.

### ANN Architecture
The model is implemented in PyTorch (`src/model.py`) with the following layered architecture:
* **Input Layer:** 8 features.
* **Hidden Layer 1 (Dense):** 32 neurons, followed by **1D Batch Normalization** and a **LeakyReLU** activation function.
* **Hidden Layer 2 (Dense):** 16 neurons, followed by **1D Batch Normalization**, **LeakyReLU** activation, and **Dropout (p=0.2)** to improve regularization.
* **Output Layer (Dense):** 1 neuron outputting raw logits.
* **Loss Function:** BCEWithLogitsLoss.
* **Optimizer:** Adam (learning rate = 0.005, weight decay = 1e-4).

---

## 📈 Model Performance

Evaluation results computed on the hold-out test set (representing 15% of the total dataset) are as follows:

| Metric | Test Set Value |
| :--- | :--- |
| **Accuracy** | 74.14% |
| **Precision** | 64.71% |
| **Recall (Sensitivity)** | 55.00% |
| **F1-Score** | 59.46% |
| **ROC-AUC** | 0.8385 |

### Confusion Matrix
* **True Negatives (TN):** 64
* **False Positives (FP):** 12
* **False Negatives (FN):** 18
* **True Positives (TP):** 22

*Metrics plots, including the ROC Curve, Confusion Matrix, and Learning Curves, are saved in the [reports/figures/](reports/figures/) folder.*

---

## 🛠️ Setup & Execution Instructions

### 1. Environment Setup
Clone this repository and set up a Python virtual environment:
```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:
```powershell
pip install -r requirements.txt
```

### 2. Training the Model
To re-train the preprocessor and neural network model:
```powershell
python -m src.train
```

### 3. Evaluating the Model
To compute classification metrics and generate evaluation figures:
```powershell
python -m src.evaluate
```

### 4. Running the Web Application
Launch the Streamlit web dashboard:
```powershell
streamlit run app/app.py
```

### 5. Running Automated Tests
Run tests with `pytest`:
```powershell
python -m pytest tests/
```
