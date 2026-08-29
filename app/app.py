import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np

# Add project root to python path to resolve local src module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our prediction module
from src.predict import DiabetesPredictor

# Set page configuration
st.set_page_config(
    page_title="GlucoSenseAI | Diabetes Prediction Using Artificial Neural Networks",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean, premium healthcare + AI dashboard aesthetic
st.markdown("""
<style>
    /* Background colors */
    .stApp {
        background-color: #f5f8f9 !important;
    }
    
    /* Sidebar Navigation styling override */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b3d4f 0%, #062633 100%) !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        color: #e6eff2 !important;
    }
    
    /* Styling the radio navigation options */
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        color: #cfdee2 !important;
        background-color: transparent !important;
        padding: 8px 14px !important;
        border-radius: 8px !important;
        margin-bottom: 6px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: rgba(255, 255, 255, 0.08) !important;
        color: white !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
        background-color: #008080 !important;
        color: white !important;
        box-shadow: 0 3px 8px rgba(0, 128, 128, 0.25) !important;
    }
    
    /* Customize native st.container(border=True) into sleek cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: white !important;
        border: 1px solid #e1ebed !important;
        border-left: 5px solid #008080 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.02) !important;
        padding: 1.8rem !important;
        margin-bottom: 1.5rem !important;
    }
    
    /* Customize st.form to match the cards */
    div[data-testid="stForm"] {
        background-color: white !important;
        border: 1px solid #e1ebed !important;
        border-left: 5px solid #008080 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.02) !important;
        padding: 2rem !important;
    }
    
    /* Hero section formatting */
    .hero-container {
        background: linear-gradient(135deg, #0b3d4f 0%, #008080 100%);
        color: white;
        padding: 2.2rem;
        border-radius: 14px;
        box-shadow: 0 5px 20px rgba(0, 128, 128, 0.12);
        margin-bottom: 2rem;
    }
    .hero-container h1 {
        color: white !important;
        font-weight: 700;
        margin: 0 0 0.5rem 0 !important;
        font-size: 2.1rem;
    }
    .hero-container p {
        color: #d8edf2 !important;
        font-size: 1.05rem;
        line-height: 1.5;
        margin: 0 !important;
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        color: #008080 !important;
        font-weight: 700 !important;
    }
    
    /* Hide default Streamlit footer and options menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Show header to keep the sidebar toggle arrow visible and functional */
    header {background-color: transparent !important;}
</style>
""", unsafe_allow_html=True)

# ----------------- Sidebar Setup -----------------
st.sidebar.markdown("<h2 style='margin-bottom: 0px;'>🩺 GlucoSenseAI</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size: 0.82rem; opacity: 0.8; margin-top: 0px; margin-bottom: 20px;'>Diabetes Prediction Using ANN</p>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    [
        "Home / Overview",
        "Diabetes Prediction",
        "Dataset & Features",
        "ANN Architecture",
        "Model Performance",
        "About"
    ]
)

# Load predictor cached
@st.cache_resource
def get_predictor():
    try:
        return DiabetesPredictor()
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None

predictor = get_predictor()

# Common Medical Disclaimer
DISCLAIMER_TEXT = (
    "Educational Use Only: GlucoSenseAI is a student machine-learning project "
    "and is not a medical diagnostic system. Its predictions should not be used "
    "as a substitute for professional medical advice."
)

# ----------------- Page 1: Home / Overview -----------------
if page == "Home / Overview":
    # Hero Section
    st.markdown("""
<div class="hero-container">
    <h1>GlucoSenseAI: Diabetes Risk Prediction</h1>
    <p>An educational machine learning application that uses an Artificial Neural Network to estimate diabetes risk from selected clinical and physiological features.</p>
</div>
""", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.container(border=True):
            st.markdown("### Project Objective")
            st.write(
                "Diabetes is a major health condition that requires timely identification and appropriate medical care. "
                "GlucoSenseAI explores how Artificial Neural Networks can be used for binary diabetes-risk classification "
                "from structured health measurements."
            )
            st.write(
                "The system preprocesses the input data, scales relevant features, and passes the processed inputs "
                "through a trained neural network to generate a model-based prediction. Performance metrics and "
                "visual analytics are provided to make the model easier to understand and evaluate."
            )
        
        st.markdown("### Key Features")
        
        with st.container(border=True):
            st.markdown("🧠 **Artificial Neural Network**")
            st.write("Uses a multi-layer feed-forward Artificial Neural Network for binary diabetes-risk classification.")
            
        with st.container(border=True):
            st.markdown("⚙️ **Data Preprocessing**")
            st.write("Handles invalid or missing values where appropriate and standardizes numerical features before model inference.")
            
        with st.container(border=True):
            st.markdown("📊 **Interactive Prediction**")
            st.write("Allows users to enter health measurements and receive a model-based diabetes-risk prediction.")
            
        with st.container(border=True):
            st.markdown("📈 **Model Evaluation**")
            st.write("Provides accuracy, precision, recall, F1-score, ROC-AUC and confusion-matrix analysis.")
            
        with st.container(border=True):
            st.markdown("🔍 **Transparent Results**")
            st.write("Provides information about the model architecture, preprocessing workflow and evaluation results.")

    with col2:
        with st.container(border=True):
            st.markdown("### Important Information")
            st.markdown(
                "**Dataset:**\n"
                "Public diabetes classification dataset\n\n"
                "**Input:**\n"
                "8 clinical and physiological features\n\n"
                "**Model:**\n"
                "Artificial Neural Network (PyTorch)\n\n"
                "**Task:**\n"
                "Binary classification\n\n"
                "**Purpose:**\n"
                "Educational diabetes-risk prediction"
            )
            
        st.warning(DISCLAIMER_TEXT)

# ----------------- Page 2: Diabetes Prediction -----------------
elif page == "Diabetes Prediction":
    st.title("Diabetes-Risk Prediction Panel")
    st.markdown("Enter patient clinical parameters below to compute a model-based diabetes-risk prediction.")
    st.markdown("---")
    
    if predictor is None:
        st.error("Model assets not found! Please verify that training scripts have run successfully.")
    else:
        with st.form("prediction_form"):
            st.markdown("### Patient Health Measurements")
            
            # Form structure grouped logically
            st.markdown("#### 👤 Demographics")
            col_demo1, col_demo2 = st.columns(2)
            with col_demo1:
                age = st.number_input("Age (in years)", min_value=21, max_value=110, value=30, step=1)
            with col_demo2:
                pregnancies = st.number_input("Pregnancies (Number of times pregnant)", min_value=0, max_value=20, value=1, step=1)
                
            st.markdown("#### 🩺 Clinical Measurements")
            col_clin1, col_clin2, col_clin3 = st.columns(3)
            with col_clin1:
                glucose = st.number_input("Glucose (2-Hour plasma glucose concentration in mg/dL)", min_value=0.0, max_value=300.0, value=120.0, step=1.0, help="Enter 0 if missing (will be automatically imputed)")
            with col_clin2:
                bp = st.number_input("Blood Pressure (Diastolic blood pressure in mmHg)", min_value=0.0, max_value=200.0, value=70.0, step=1.0, help="Enter 0 if missing (will be automatically imputed)")
            with col_clin3:
                bmi = st.number_input("BMI (Body Mass Index - kg/m²)", min_value=0.0, max_value=70.0, value=28.5, step=0.1, help="Enter 0 if missing (will be automatically imputed)")
                
            st.markdown("#### 🧪 Other Measurements")
            col_aux1, col_aux2, col_aux3 = st.columns(3)
            with col_aux1:
                skin = st.number_input("Skin Thickness (Triceps skin fold thickness in mm)", min_value=0.0, max_value=100.0, value=20.0, step=1.0, help="Enter 0 if missing (will be automatically imputed)")
            with col_aux2:
                insulin = st.number_input("Insulin (2-Hour serum insulin in mu U/ml)", min_value=0.0, max_value=1200.0, value=80.0, step=1.0, help="Enter 0 if missing (will be automatically imputed)")
            with col_aux3:
                dpf = st.number_input("Diabetes Pedigree Function (Family history score)", min_value=0.0, max_value=3.0, value=0.47, step=0.01)
                
            submit_button = st.form_submit_button("Evaluate Diabetes-Risk", type="primary")
            
        if submit_button:
            input_dict = {
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': bp,
                'SkinThickness': skin,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': dpf,
                'Age': age
            }
            
            with st.spinner("Processing preprocessor transformations and evaluating network logs..."):
                try:
                    res = predictor.predict(input_dict)
                    prob = res['probability']
                    risk = res['risk_level']
                    
                    st.markdown("---")
                    
                    with st.container(border=True):
                        st.markdown("### Prediction Results Analysis")
                        
                        r_col1, r_col2 = st.columns([1, 2])
                        
                        with r_col1:
                            if risk == "Low Risk":
                                st.success(f"**Diabetes-Risk prediction:** {risk}")
                            elif risk == "Moderate Risk":
                                st.warning(f"**Diabetes-Risk prediction:** {risk}")
                            else:
                                st.error(f"**Diabetes-Risk prediction:** {risk}")
                                
                            st.metric(label="Model-based Probability", value=f"{prob:.1%}")
                            
                        with r_col2:
                            st.markdown("**Model Interpretation:**")
                            st.write(res['interpretation'])
                            
                            # Show imputed cols
                            imputed = [k for k, v in input_dict.items() if k in predictor.preprocessor.zero_impute_cols and v == 0]
                            if imputed:
                                st.caption(f"*Note: Zero values in the following inputs were automatically imputed using training dataset medians: {', '.join(imputed)}*")
                                
                        st.warning(DISCLAIMER_TEXT)
                        
                except Exception as e:
                    st.error(f"Error computing prediction metrics: {e}")

# ----------------- Page 3: Dataset & Features -----------------
elif page == "Dataset & Features":
    st.title("Dataset & Physiological Features")
    st.markdown("Technical details regarding the dataset origin, feature definitions, and preprocessing logic.")
    st.markdown("---")
    
    # Summary Metric Row
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.metric("Total Records", "768 Patients")
    with c2:
        with st.container(border=True):
            st.metric("Input Dimensions", "8 Health Features")
    with c3:
        with st.container(border=True):
            st.metric("Prediction Class Target", "Binary Outcome (0/1)")
            
    col1, col2 = st.columns([3, 2])
    
    with col1:
        with st.container(border=True):
            st.markdown("### Dataset Source & Provenance")
            st.write(
                "GlucoSenseAI utilizes the **Pima Indians Diabetes Database** compiled by the "
                "National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK). The dataset consists of physiological "
                "measurements from female patients of at least 21 years of age."
            )
            
            st.markdown("### Physiological Feature Glossary")
            glossary = pd.DataFrame({
                "Feature Name": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
                "Description": [
                    "Number of times pregnant",
                    "Plasma glucose concentration at 2 hours in an oral glucose tolerance test (mg/dL)",
                    "Diastolic blood pressure (mmHg)",
                    "Triceps skin fold thickness (mm)",
                    "2-Hour serum insulin (mu U/ml)",
                    "Body Mass Index (weight in kg / (height in m)²)",
                    "Diabetes pedigree function (family relative history score)",
                    "Patient age in years"
                ],
                "Zero Values Status": [
                    "Valid measurement",
                    "Implausible (Imputed with median)",
                    "Implausible (Imputed with median)",
                    "Implausible (Imputed with median)",
                    "Implausible (Imputed with median)",
                    "Implausible (Imputed with median)",
                    "Valid measurement",
                    "Valid measurement"
                ]
            })
            st.table(glossary)
            
            st.markdown("### Target Classification Variable")
            st.write(
                "**Outcome:** Class indicator denoting diabetes status (0 = No diabetes, 1 = Diabetes). "
                "The dataset exhibits a class split of 500 negative and 268 positive records."
            )
            
    with col2:
        with st.container(border=True):
            st.markdown("### Preprocessing Pipeline Steps")
            st.markdown(
                "1. **Missing Value Imputation:** Zero values in the clinical columns (`Glucose`, `BloodPressure`, "
                "`SkinThickness`, `Insulin`, `BMI`) represent missing entries rather than true zeros. These values are replaced "
                "using the computed overall median from the training split.\n\n"
                "2. **Feature Standardization:** Once imputed, all independent variables are standardized using a "
                "`StandardScaler` fit on the training split to produce zero-mean and unit-variance features before network pass."
            )
            
        dist_img = "reports/figures/class_distribution.png"
        if os.path.exists(dist_img):
            st.image(dist_img, caption="Dataset Class Balance", use_container_width=True)
            
    st.markdown("---")
    st.subheader("Feature Correlation Matrix")
    corr_img = "reports/figures/correlation_heatmap.png"
    if os.path.exists(corr_img):
        st.image(corr_img, caption="Pearson Correlation of physiological inputs", use_container_width=True)

# ----------------- Page 4: ANN Architecture -----------------
elif page == "ANN Architecture":
    st.title("Artificial Neural Network Architecture")
    st.markdown("Details of the PyTorch neural network configuration and training parameters.")
    st.markdown("---")
    
    with st.container(border=True):
        st.markdown("### Network Layer Flowchart")
        st.code("""
Input Features (8 dimensions: Age, BMI, Glucose, Pregnancies, etc.)
              │
              ▼
  Input Layer (8 features mapped directly)
              │
              ▼
Hidden Layer 1 (32 nodes + Batch Normalization + LeakyReLU)
              │
              ▼
Hidden Layer 2 (16 nodes + Batch Normalization + LeakyReLU + Dropout p=0.2)
              │
              ▼
 Output Layer (1 unit outputting logit classification value)
              │
              ▼
Diabetes-Risk Prediction (Sigmoid Probability Output)
        """, language="text")
        
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.markdown("### Model Structure")
            arch_data = pd.DataFrame({
                "Layer Name": ["Input Layer", "Hidden Layer 1", "Hidden Layer 2", "Output Layer"],
                "Neurons/Nodes": ["8", "32", "16", "1"],
                "Activation / Regularization": [
                    "None (Raw inputs)",
                    "LeakyReLU (slope=0.1) + Batch Normalization (1D)",
                    "LeakyReLU (slope=0.1) + Batch Normalization (1D) + Dropout (p=0.2)",
                    "None (Raw logits outputted)"
                ]
            })
            st.table(arch_data)
            
            st.markdown("**Regularization details:**")
            st.markdown(
                "* **Batch Normalization (1D):** Normalizes activations across batches to reduce internal covariate shift and accelerate convergence.\n"
                "* **Dropout (p=0.2):** Randomly drops 20% of activations in hidden layer 2 during training to reduce neuron co-dependency and prevent overfitting."
            )
            
    with col2:
        with st.container(border=True):
            st.markdown("### Training Hyperparameters")
            hyper_data = pd.DataFrame({
                "Hyperparameter": [
                    "Loss Function",
                    "Optimizer",
                    "Base Learning Rate",
                    "Weight Decay (L2 Regularization)",
                    "Batch Size",
                    "Training Split Ratio",
                    "Early Stopping Patience"
                ],
                "Value": [
                    "BCEWithLogitsLoss",
                    "Adam",
                    "0.005",
                    "0.0001 (1e-4)",
                    "32",
                    "70% Train, 15% Validation, 15% Test",
                    "50 Epochs"
                ]
            })
            st.table(hyper_data)
            
            # Load best epoch from metadata
            meta_json = "models/model_metadata.json"
            best_epoch = "5"
            if os.path.exists(meta_json):
                with open(meta_json, 'r') as f:
                    best_epoch = str(json.load(f).get('best_epoch', 5))
                    
            st.success(f"**Training Convergence:** Training terminated at epoch 55 using early stopping (restored best weights from epoch {best_epoch}).")

# ----------------- Page 5: Model Performance -----------------
elif page == "Model Performance":
    st.title("Model Performance & Diagnostics")
    st.markdown("Validation and testing metrics compiled from the final trained ANN model.")
    st.markdown("---")
    
    eval_json = "models/evaluation_results.json"
    if not os.path.exists(eval_json):
        st.error("Evaluation results file not found! Please run training and evaluation first.")
    else:
        with open(eval_json, 'r') as f:
            res = json.load(f)
            
        # Display Core Metrics in Metric Cards
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        with col_m1:
            with st.container(border=True):
                st.metric("Accuracy", f"{res['accuracy']:.2%}")
        with col_m2:
            with st.container(border=True):
                st.metric("Precision", f"{res['precision']:.2%}")
        with col_m3:
            with st.container(border=True):
                st.metric("Recall (Sensitivity)", f"{res['recall']:.2%}")
        with col_m4:
            with st.container(border=True):
                st.metric("F1-Score", f"{res['f1_score']:.2%}")
        with col_m5:
            with st.container(border=True):
                st.metric("ROC-AUC", f"{res['roc_auc']:.4f}")
                
        st.markdown("---")
        
        # Display charts
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container(border=True):
                cm_img = "reports/figures/confusion_matrix.png"
                if os.path.exists(cm_img):
                    st.image(cm_img, caption="Confusion Matrix on Test Dataset", use_container_width=True)
                else:
                    st.warning("Confusion matrix heatmap not found.")
                    
            with st.container(border=True):
                curves_img = "reports/figures/learning_curves.png"
                if os.path.exists(curves_img):
                    st.image(curves_img, caption="Loss & Accuracy curves (Train vs Validation)", use_container_width=True)
                else:
                    st.warning("Learning curves plot not found.")
                    
        with col2:
            with st.container(border=True):
                roc_img = "reports/figures/roc_curve.png"
                if os.path.exists(roc_img):
                    st.image(roc_img, caption="Receiver Operating Characteristic (ROC) Curve", use_container_width=True)
                else:
                    st.warning("ROC curve image not found.")
                    
            with st.container(border=True):
                st.markdown("### Metrics Interpretation Guide")
                st.markdown(
                    "* **Accuracy (74.14%):** Overall percentage of correct predictions on the test dataset.\n\n"
                    "* **Precision (64.71%):** The proportion of correctly predicted positive cases among all positive predictions. Reduces false positive risk.\n\n"
                    "* **Recall (55.00%):** The proportion of actual positives correctly identified by the network.\n\n"
                    "* **ROC-AUC (0.8385):** Evaluates the network's capacity to distinguish between outcomes across classification thresholds, indicating robust discriminative performance."
                )

# ----------------- Page 6: About -----------------
elif page == "About":
    st.title("About the Project")
    st.markdown("Information regarding the concept, technology stack, and metadata of GlucoSenseAI.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.subheader("System Overview")
            st.write(
                "GlucoSenseAI is an educational machine-learning application that uses an Artificial Neural Network "
                "to estimate diabetes risk from selected clinical and physiological features."
            )
            st.write(
                "The system includes data preprocessing, neural-network training, model evaluation and an "
                "interactive prediction interface. It is designed to demonstrate how deep-learning techniques "
                "can be applied to structured healthcare data."
            )
            
        with st.container(border=True):
            st.subheader("Technology Stack")
            st.markdown(
                "**Core Deep Learning:**\n"
                "PyTorch\n\n"
                "**Machine Learning & Preprocessing:**\n"
                "Scikit-learn\n\n"
                "**Data Processing:**\n"
                "Pandas, NumPy\n\n"
                "**Visualization:**\n"
                "Matplotlib, Seaborn\n\n"
                "**Application Framework:**\n"
                "Streamlit\n\n"
                "**Environment:**\n"
                "Python virtual environment"
            )
            
    with col2:
        with st.container(border=True):
            st.subheader("System Metadata")
            st.markdown(
                "**System Name:**\n"
                "GlucoSenseAI\n\n"
                "**Project Title:**\n"
                "Diabetes Prediction Using Artificial Neural Networks\n\n"
                "**Code Version:**\n"
                "v1.0.0 (Stable release)"
            )
            
        st.warning(DISCLAIMER_TEXT)
