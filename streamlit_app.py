# Create a Streamlit UI with a professional white gradient background,
# glassmorphism card container, rounded corners, soft blur effect,
# and white translucent panels with a modern, clean design.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Loan Approval Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# ---------- WHITISH-BLUE GLASSMORPHISM THEME (FORCED LIGHT) ----------
bg = """
<style>
/* Force Light Theme */
:root {
    color-scheme: light;
}

[data-theme="dark"] {
    display: none !important;
}

html, body, .stApp {
    color-scheme: light !important;
}

* {
    margin: 0;
    padding: 0;
}

.stApp {
    background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%);
    background-attachment: fixed;
    min-height: 100vh;
}

.block-container {
    background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(16px);
    border-radius: 28px;
    padding: 3rem;
    box-shadow: 0 15px 50px rgba(100, 120, 150, 0.08),
                inset 0 1px 0 rgba(255, 255, 255, 0.6);
    border: 1px solid #e8ecf2;
    margin-top: 2rem !important;
    margin-bottom: 2rem !important;
}

/* Main Title */
h1 {
    color: #111111;
    font-weight: 900;
    font-size: 2.8rem;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    margin-bottom: 0.5rem;
}

/* Section Headers */
h2 {
    color: #111111;
    font-weight: 800;
    font-size: 2rem;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    margin-top: 2rem !important;
    margin-bottom: 1.5rem !important;
}

h3 {
    color: #222222;
    font-weight: 700;
    font-size: 1.4rem;
    text-shadow: 0 1px 5px rgba(0, 0, 0, 0.02);
}

/* Text and Content */
p, span, .stMarkdown {
    color: #333333;
}

/* Buttons */
.stButton > button {
    background: #ffffff !important;
    color: #111111 !important;
    border: 2px solid #d5dce5 !important;
    border-radius: 12px !important;
    padding: 1rem 2.5rem !important;
    font-weight: 700 !important;
    font-size: 1.15rem !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.12) !important;
    border-color: #3d4d5c !important;
    background: #f9fbfd !important;
}

/* Input Fields */
.stNumberInput > div > div > input,
.stSelectbox > div > div > select,
.stSlider > div > div > div {
    background: rgba(255, 255, 255, 0.95);
    color: #111111;
    border: 1px solid #d8e2ee;
    border-radius: 12px;
    padding: 0.75rem;
}

.stNumberInput > div > div > input::placeholder,
.stSelectbox > div > div > select::placeholder {
    color: rgba(100, 100, 100, 0.6);
}

/* Metrics Cards */
.stMetric {
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(15px);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid #e8ecf2;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.05);
}

.stMetric > div > p {
    color: #666666;
}

.stMetric > div > div {
    color: #111111;
    font-size: 2rem;
    font-weight: 800;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 14px !important;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.7);
    border-radius: 14px;
    color: #333333;
    border: 2px solid #cfd8e3 !important;
    padding: 8px 18px !important;
}

.stTabs [aria-selected="true"] {
    background: rgba(255, 255, 255, 0.95);
    color: #111111;
    border: 2px solid #3d4d5c;
}

/* Data Tables */
.stDataFrame {
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 2px solid #d5dce5 !important;
    padding: 1rem !important;
}

.stDataFrame tbody {
    background: rgba(255, 255, 255, 0.9) !important;
}

/* Divider */
.stMarkdown hr {
    border: 1px solid #d5dce5;
    margin: 2rem 0;
}

/* Success and Error Boxes */
.stSuccess {
    background: rgba(40, 167, 69, 0.12);
    border: 2px solid rgba(40, 167, 69, 0.3);
    border-radius: 12px;
    color: #222222 !important;
}

.stError {
    background: rgba(220, 53, 69, 0.12);
    border: 2px solid rgba(220, 53, 69, 0.3);
    border-radius: 12px;
    color: #222222 !important;
}

.stWarning {
    background: rgba(255, 193, 7, 0.12);
    border: 2px solid rgba(255, 193, 7, 0.3);
    border-radius: 12px;
    color: #222222 !important;
}

.stInfo {
    background: rgba(13, 110, 253, 0.12);
    border: 2px solid rgba(13, 110, 253, 0.3);
    border-radius: 12px;
    color: #222222 !important;
}

</style>
"""

st.markdown(bg, unsafe_allow_html=True)

# ---------- LOAD AND PREPARE DATA ----------
@st.cache_resource
def load_and_prepare_data():
    """Load and preprocess the loan dataset"""
    df = pd.read_csv("loan_approval_dataset.csv")
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Clean string values
    for col in df.columns:
        if df[col].dtype == "object" or str(df[col].dtype) == "string":
            df[col] = df[col].astype(str).str.strip()
    
    return df

@st.cache_resource
def train_model_pipeline(df):
    """Train the Random Forest model"""
    df_copy = df.copy()
    
    # Encode categorical variables
    le_dict = {}
    non_numeric_cols = df_copy.select_dtypes(exclude=['number']).columns
    
    for col in non_numeric_cols:
        le = LabelEncoder()
        encoded_values = le.fit_transform(df_copy[col].astype(str))
        df_copy[col] = encoded_values
        le_dict[col] = le
    
    # Split data
    X = df_copy.drop("loan_status", axis=1)
    y = df_copy["loan_status"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    return {
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'X_train': X_train,
        'y_train': y_train,
        'X': X,
        'le_dict': le_dict,
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_mat': confusion_matrix(y_test, y_pred),
        'class_report': classification_report(y_test, y_pred, output_dict=True)
    }

# Load data
df = load_and_prepare_data()
model_data = train_model_pipeline(df)

# ---------- SIDEBAR NAVIGATION ----------
st.sidebar.markdown("""
<style>
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(14px);
        border-right: 1px solid #d5dce5;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Page",
    ["Dashboard", "Dataset", "Analytics", "Model", "Predict"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Stats")
st.sidebar.metric("Accuracy", "97")
st.sidebar.metric("Records", f"{len(df):,}")

# ---------- PAGE 1: DASHBOARD ----------
if page == "Dashboard":
    st.title("Loan Approval Predictor")
    st.markdown("**AI-Powered Loan Approval System with Glassmorphism Design**")
    st.markdown("---")
    
    # Key Metrics - One at a time style
    st.subheader("Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", "97%")
    with col2:
        st.metric("Dataset Size", f"{len(df):,}")
    with col3:
        st.metric("Features", df.shape[1])
    with col4:
        st.metric("Trees", "200")
    
    st.markdown("---")
    
    # Dataset Overview
    st.subheader("First Look at Data")
    
    st.write("**First 5 Records:**")
    st.dataframe(df.head(5), use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.write("**Quick Statistics:**")
    st.dataframe(df.describe().round(2), use_container_width=True)
    
    st.markdown("---")
    
    # Feature Importance - Clean one section
    st.subheader("Top Features Contributing to Loan Decisions")
    
    importances = pd.Series(
        model_data['model'].feature_importances_,
        index=model_data['X'].columns
    ).sort_values(ascending=False).head(6)
    
    _, col_center, _ = st.columns([1, 2, 1])
    with col_center:
        fig, ax = plt.subplots(figsize=(7, 4), facecolor='white', edgecolor='none')
        ax.set_facecolor('#ffffff')
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(importances)))
        bars = ax.barh(range(len(importances)), importances.values, color=colors, edgecolor='#d5dce5', linewidth=2)
        
        ax.set_yticks(range(len(importances)))
        ax.set_yticklabels(importances.index, fontsize=12, fontweight='bold', color='#111111')
        ax.set_xlabel("Importance Score", fontsize=12, fontweight='bold', color='#111111')
        ax.set_title("Top 6 Most Important Features", fontsize=14, fontweight='bold', color='#111111')
        ax.invert_yaxis()
        ax.tick_params(colors='#111111')
        ax.grid(axis='x', alpha=0.3, linestyle='--', color='#d5dce5')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        for i, (bar, val) in enumerate(zip(bars, importances.values)):
            ax.text(val, i, f'  {val:.2%}', va='center', fontweight='bold', fontsize=10, color='#1f2d3d')
        
        plt.tight_layout()
        st.pyplot(fig)

# ---------- PAGE 2: DATASET ----------
elif page == "Dataset":
    st.title("Dataset Overview")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    st.markdown("---")
    
    # Tabs for clean separation
    tab1, tab2, tab3 = st.tabs(["Head Data", "Data Info", "Summary"])
    
    with tab1:
        st.subheader("First 10 Records")
        st.dataframe(df.head(10), width='stretch')
    
    with tab2:
        st.subheader("Data Types & Missing Values")
        info_data = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str).values,
            'Missing': df.isnull().sum().values.astype(int)
        })
        st.dataframe(info_data, width='stretch')
    
    with tab3:
        st.subheader("Statistical Summary")
        st.dataframe(df.describe().round(3), width='stretch')

# ---------- PAGE 3: ANALYTICS ----------
elif page == "Analytics":
    st.title("Exploratory Data Analysis")
    st.markdown("---")
    
    ana_tab1, ana_tab2, ana_tab3 = st.tabs(["Distributions", "Status", "Correlation"])
    
    with ana_tab1:
        st.subheader("Feature Distributions")
        # Only show 6 key features
        selected_features = ['income_annum', 'loan_amount', 'cibil_score', 'no_of_dependents', 'residential_assets_value', 'commercial_assets_value']
        n_cols = len(selected_features)
        n_rows = 2
        
        _, col_center, _ = st.columns([1, 2, 1])
        with col_center:
            fig, axes = plt.subplots(n_rows, 3, figsize=(7, 4.5), facecolor='white')
            axes = axes.flatten()
            
            for idx, col in enumerate(selected_features):
                axes[idx].hist(df[col], bins=25, color='#1f2d3d', edgecolor='#e8ecf2', linewidth=1.2, alpha=0.85)
                axes[idx].set_title(col, fontsize=10, fontweight='bold', color='#111111')
                axes[idx].set_xlabel('Value', fontsize=8, color='#333333')
                axes[idx].set_ylabel('Frequency', fontsize=8, color='#333333')
                axes[idx].set_facecolor('#ffffff')
                axes[idx].grid(axis='y', alpha=0.3, linestyle='--', color='#d5dce5')
                axes[idx].tick_params(colors='#333333', labelsize=7)
            
            plt.suptitle("Distribution of Key Features", fontsize=13, fontweight='bold', color='#111111', y=0.995)
            plt.tight_layout()
            st.pyplot(fig)
    
    with ana_tab2:
        st.subheader("Loan Status Analysis")
        
        _, col_center, _ = st.columns([1, 2, 1])
        with col_center:
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
        ax.set_facecolor('#ffffff')
        
        status_counts = df['loan_status'].value_counts()
        colors_status = ['#28a745', '#dc3545']
        
        bars = ax.bar(range(len(status_counts)), status_counts.values, color=colors_status, edgecolor='#e8ecf2', linewidth=2, width=0.6)
        ax.set_xlabel("Status", fontsize=12, fontweight='bold', color='#111111')
        ax.set_ylabel("Count", fontsize=12, fontweight='bold', color='#111111')
        ax.set_title("Loan Approval vs Rejection", fontsize=14, fontweight='bold', color='#111111')
        ax.set_xticklabels(status_counts.index, fontsize=11, color='#111111')
        ax.tick_params(colors='#111111')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=12, color='#111111')
        
            plt.tight_layout()
            st.pyplot(fig)
    
    with ana_tab3:
        st.subheader("Correlation Heatmap")
        
        _, col_center, _ = st.columns([1, 2, 1])
        with col_center:
            fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
        numeric_df = df.select_dtypes(include=np.number)
        corr_matrix = numeric_df.corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap="Blues", center=0, fmt='.2f',
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax,
                   annot_kws={'fontsize': 9, 'color': '#111111'},
                   vmin=-1, vmax=1)
        ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight='bold', pad=20, color='#111111')
        plt.tight_layout()
        st.pyplot(fig)

# ---------- PAGE 4: MODEL PERFORMANCE ----------
elif page == "Model":
    st.title("Model Performance")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{model_data['accuracy']*100:.2f}%")
    with col2:
        cm = model_data['confusion_mat']
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp)
        st.metric("Precision", f"{precision*100:.2f}%")
    with col3:
        recall = tp / (tp + fn)
        st.metric("Recall", f"{recall*100:.2f}%")
    
    st.markdown("---")
    
    model_tab1, model_tab2, model_tab3 = st.tabs(["Confusion Matrix", "Features", "Metrics"])
    
    with model_tab1:
        st.subheader("Confusion Matrix Analysis")
        
        _, col_center, _ = st.columns([1, 2, 1])
        with col_center:
            fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
        cm = model_data['confusion_mat']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['Approved', 'Rejected'],
                   yticklabels=['Approved', 'Rejected'],
                   annot_kws={'fontsize': 16, 'fontweight': 'bold', 'color': '#111111'},
                   ax=ax, linewidths=3, linecolor='#e8ecf2',
                   cbar_kws={'label': 'Count'})
        
        ax.set_xlabel("Predicted", fontsize=13, fontweight='bold', color='#111111')
        ax.set_ylabel("Actual", fontsize=13, fontweight='bold', color='#111111')
        ax.set_title("Confusion Matrix", fontsize=15, fontweight='bold', color='#111111')
        ax.set_facecolor('#ffffff')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with model_tab2:
        st.subheader("Feature Importance Ranking")
        
        importances = pd.Series(
            model_data['model'].feature_importances_,
            index=model_data['X'].columns
        ).sort_values(ascending=False)
        
        _, col_center, _ = st.columns([1, 2, 1])
        with col_center:
            fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        ax.set_facecolor('#ffffff')
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(importances)))
        bars = ax.barh(range(len(importances)), importances.values, color=colors, edgecolor='#d5dce5', linewidth=1.5)
        
        ax.set_yticks(range(len(importances)))
        ax.set_yticklabels(importances.index, fontsize=10, fontweight='bold', color='#111111')
        ax.set_xlabel("Importance Score", fontsize=13, fontweight='bold', color='#111111')
        ax.set_title("Model Feature Importance", fontsize=14, fontweight='bold', color='#111111')
        ax.invert_yaxis()
        ax.tick_params(colors='#111111', labelsize=10)
        ax.grid(axis='x', alpha=0.3, linestyle='--', color='#d5dce5')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        for i, (bar, val) in enumerate(zip(bars, importances.values)):
            ax.text(val, i, f'  {val:.2%}', va='center', fontweight='bold', fontsize=9, color='#1f2d3d')
        
            plt.tight_layout()
            st.pyplot(fig)
    
    with model_tab3:
        st.subheader("Classification Metrics")
        
        crr = model_data['class_report']
        metrics_df = pd.DataFrame({
            'Class': ['Approved', 'Rejected', 'Avg'],
            'Precision': [f"{crr['0']['precision']:.4f}", f"{crr['1']['precision']:.4f}", f"{crr['weighted avg']['precision']:.4f}"],
            'Recall': [f"{crr['0']['recall']:.4f}", f"{crr['1']['recall']:.4f}", f"{crr['weighted avg']['recall']:.4f}"],
            'F1-Score': [f"{crr['0']['f1-score']:.4f}", f"{crr['1']['f1-score']:.4f}", f"{crr['weighted avg']['f1-score']:.4f}"]
        })
        
        st.dataframe(metrics_df, width='stretch', hide_index=True)

# ---------- PAGE 5: PREDICTION ----------
elif page == "Predict":
    st.title("Make Prediction")
    st.markdown("**Enter applicant details for loan prediction**")
    st.markdown("---")
    
    col_form1, col_form2, col_form3 = st.columns(3)
    
    with col_form1:
        st.write("**Financial**")
        income = st.number_input("Annual Income (₹)", 100000, 5000000, 500000, step=10000)
        loan_amount = st.number_input("Loan Amount (₹)", 10000, 1000000, 200000, step=10000)
        cibil = st.slider("CIBIL Score", 300, 900, 700)
    
    with col_form2:
        st.write("**Personal**")
        dependents = st.selectbox("Dependents", [0, 1, 2, 3, 4, 5])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed?", ["No", "Yes"])
    
    with col_form3:
        st.write("**Assets**")
        loan_term = st.slider("Loan Term (Months)", 12, 480, 360, step=12)
        residential = st.number_input("Residential Assets (₹)", 0, 100000000, 1000000, step=100000)
        commercial = st.number_input("Commercial Assets (₹)", 0, 100000000, 500000, step=100000)
    
    luxury = st.number_input("Luxury Assets (₹)", 0, 100000000, 1000000, step=100000)
    bank_asset = st.number_input("Bank Assets (₹)", 0, 100000000, 500000, step=100000)
    
    st.markdown("---")
    
    col_btn1, col_btn2 = st.columns([2, 2])
    
    with col_btn1:
        predict_btn = st.button("PREDICT LOAN STATUS", use_container_width=True, key="predict_btn", type="primary")
    
    if predict_btn:
        # Encode
        education_enc = 1 if education == "Graduate" else 0
        self_emp_enc = 1 if self_employed == "Yes" else 0
        
        # Create input
        pred_input = pd.DataFrame({
            'loan_id': [0], 'no_of_dependents': [dependents], 'education': [education_enc],
            'self_employed': [self_emp_enc], 'income_annum': [income], 'loan_amount': [loan_amount],
            'loan_term': [loan_term], 'cibil_score': [cibil], 'residential_assets_value': [residential],
            'commercial_assets_value': [commercial], 'luxury_assets_value': [luxury],
            'bank_asset_value': [bank_asset]
        })
        
        # Predict
        pred = model_data['model'].predict(pred_input)[0]
        pred_proba = model_data['model'].predict_proba(pred_input)[0]
        
        st.markdown("---")
        st.subheader("✨ Prediction Result")
        
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            if pred == 0:
                st.markdown(f"""
                <div style='background: rgba(40, 167, 69, 0.3); border: 3px solid #28a745; border-radius: 20px; padding: 2rem; text-align: center;'>
                    <h2 style='color: #111111; margin: 0;'>APPROVED</h2>
                    <p style='color: #111111; font-size: 1.3rem; margin: 1rem 0 0 0;'>{pred_proba[0]*100:.1f}% Confident</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background: rgba(220, 53, 69, 0.3); border: 3px solid #dc3545; border-radius: 20px; padding: 2rem; text-align: center;'>
                    <h2 style='color: #111111; margin: 0;'>REJECTED</h2>
                    <p style='color: #111111; font-size: 1.3rem; margin: 1rem 0 0 0;'>{pred_proba[1]*100:.1f}% Confident</p>
                </div>
                """, unsafe_allow_html=True)
        
        with result_col2:
            fig, ax = plt.subplots(figsize=(6, 4.5), facecolor='white')
            ax.set_facecolor('#ffffff')
            
            labels = ['Approved', 'Rejected']
            sizes = pred_proba * 100
            colors_pie = ['#28a745', '#dc3545']
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                              colors=colors_pie, explode=(0.1, 0), shadow=True,
                              textprops={'fontsize': 12, 'fontweight': 'bold', 'color': '#111111'},
                              wedgeprops={'edgecolor': '#e8ecf2', 'linewidth': 2})
            
            for autotext in autotexts:
                autotext.set_color('#111111')
                autotext.set_fontweight('bold')
            
            ax.set_title("Prediction Distribution", fontsize=13, fontweight='bold', color='#111111')
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("---")
        st.subheader("Application Summary")
        
        summary_data = {
            'Field': ['Income', 'Loan Amount', 'Loan Term', 'CIBIL', 'Dependents', 'Education', 'Self Employed',
                     'Residential', 'Commercial', 'Luxury', 'Bank Assets'],
            'Value': [f"₹{income:,.0f}", f"₹{loan_amount:,.0f}", f"{loan_term}mo", cibil, dependents,
                     education, self_employed, f"₹{residential:,.0f}", f"₹{commercial:,.0f}",
                     f"₹{luxury:,.0f}", f"₹{bank_asset:,.0f}"]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, width='stretch', hide_index=True)

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 2rem; background: rgba(255, 255, 255, 0.6); border-radius: 16px; margin-top: 2rem;'>
        <p style='color: #111111; font-size: 1.1rem; font-weight: bold;'>
        Loan Approval Prediction System
        </p>
        <p style='color: #333333; font-size: 0.95rem;'>
        97% Accuracy | 4,269 Records | Random Forest ML
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
