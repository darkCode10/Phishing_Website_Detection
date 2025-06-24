import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pickle
from elmModel import ELM
from sklearn.svm import SVC
import joblib

elm=joblib.load('elm_model.pkl')
#svm=joblib.load('svm_model.pkl')

with open('elm_evaluation.pkl','rb') as file:
    elm_evaluation=pickle.load(file)

with open('svm_evaluation.pkl','rb') as file:
    svm_evaluation=pickle.load(file)

with open('nb_evaluation.pkl','rb') as file:
    nb_evaluation=pickle.load(file)
                            
# Set Streamlit page configuration
st.set_page_config(
    page_title="Phishing Detection Model Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Model Data (Hardcoded) ---
model_data = {
    "ELM": {
        "accuracy": elm_evaluation['accuracy'],
        "precision": elm_evaluation['precision'],
        "recall": elm_evaluation['recall'],
        "f1Score": elm_evaluation['f1_score'],
        "classificationReport": elm_evaluation['classification'],
        "confusionMatrix": {'tn': int(elm_evaluation['confusion_matrix'][0][0]),
                            'fp': int(elm_evaluation['confusion_matrix'][0][1]), 
                            'fn': int(elm_evaluation['confusion_matrix'][1][0]), 
                            'tp': int(elm_evaluation['confusion_matrix'][1][1])}
    },
    "SVM": {
        "accuracy": svm_evaluation['accuracy'],
        "precision": svm_evaluation['precision'],
        "recall": svm_evaluation['recall'],
        "f1Score": svm_evaluation['f1_score'],
        "classificationReport": svm_evaluation['classification'],
        "confusionMatrix": {'tn': int(svm_evaluation['confusion_matrix'][0][0]),
                            'fp': int(svm_evaluation['confusion_matrix'][0][1]), 
                            'fn': int(svm_evaluation['confusion_matrix'][1][0]), 
                            'tp': int(svm_evaluation['confusion_matrix'][1][1])}
    },
    "Naive Bayes": {
        "accuracy": nb_evaluation['accuracy'],
        "precision": nb_evaluation['precision'],
        "recall": nb_evaluation['recall'],
        "f1Score": nb_evaluation['f1_score'],
        "classificationReport": nb_evaluation['classification'],
        "confusionMatrix": {'tn': int(nb_evaluation['confusion_matrix'][0][0]),
                            'fp': int(nb_evaluation['confusion_matrix'][0][1]), 
                            'fn': int(nb_evaluation['confusion_matrix'][1][0]), 
                            'tp': int(nb_evaluation['confusion_matrix'][1][1])}
    }
}

# --- Helper function for rendering Confusion Matrix using native Streamlit ---
def render_confusion_matrix(matrix_data):
    st.markdown("##### Confusion Matrix")
    
    # Using columns to simulate a grid layout for the matrix
    # Header row
    col_empty, col_actual_legit, col_actual_phish = st.columns([1, 2, 2])
    with col_actual_legit:
        st.write("**Actual**")
    with col_actual_phish:
        st.write("") # Placeholder for alignment, actual label above
    
    # Row for "Legitimate" (Actual) / Predicted labels
    st.write("---") # Visual separator
    col_y_predicted, col_actual_header, col_actual_header_2 = st.columns([1,2,2])
    with col_y_predicted:
        st.write("**Predicted**")
    with col_actual_header:
        st.write("Legitimate")
    with col_actual_header_2:
        st.write("Phishing")

    # Row 1: Predicted Legitimate
    col_pred_legit, col_tn, col_fp = st.columns([1,2,2])
    with col_pred_legit:
        st.write("Legitimate")
    with col_tn:
        st.info(f"TN: {matrix_data['tn']}") # Use Streamlit's info/success/warning for visual cues
    with col_fp:
        st.warning(f"FP: {matrix_data['fp']}")

    # Row 2: Predicted Phishing
    col_pred_phish, col_fn, col_tp = st.columns([1,2,2])
    with col_pred_phish:
        st.write("Phishing")
    with col_fn:
        st.warning(f"FN: {matrix_data['fn']}")
    with col_tp:
        st.info(f"TP: {matrix_data['tp']}")
    st.write("---") # Visual separator


# --- Header Section ---
st.title("Phishing Detection Model Dashboard")
st.write("An interactive analysis of Extreme Learning Machine (ELM), Support Vector Machine (SVM), and Naive Bayes models for identifying phishing websites.")

# --- Section 1: High-Level Comparison ---
st.header("At-a-Glance Performance")
st.write("Start with a high-level comparison of the three models. The chart below visualizes the primary performance metrics, providing a quick overview of which model performs best across the board.")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="ELM Accuracy", value=f"{float(model_data['ELM']['accuracy']*100):.2f}%")
with col2:
    st.metric(label="SVM Accuracy", value=f"{float(model_data['SVM']['accuracy']*100):.2f}%")
with col3:
    st.metric(label="Naive Bayes Accuracy", value=f"{float(model_data['Naive Bayes']['accuracy']*100):.2f}%")

st.subheader("Comparative Metrics")

metrics_df = pd.DataFrame({
    'Model': ['ELM', 'SVM', 'Naive Bayes'] * 3,
    'Metric': ['Accuracy'] * 3 + ['F1 Score (Macro)'] * 3 + ['Precision (Macro)'] * 3,
    'Value': [
        model_data['ELM']['accuracy'], model_data['SVM']['accuracy'], model_data['Naive Bayes']['accuracy'],
        model_data['ELM']['f1Score'], model_data['SVM']['f1Score'], model_data['Naive Bayes']['f1Score'],
        model_data['ELM']['precision'], model_data['SVM']['precision'], model_data['Naive Bayes']['precision']
    ]
})

fig_bar = px.bar(
    metrics_df,
    x='Model',
    y='Value',
    color='Metric',
    barmode='group',
    labels={'Value': 'Score', 'Model': 'Model Name'},
    color_discrete_map={
        'Accuracy': '#4682B4',  # SteelBlue
        'F1 Score (Macro)': '#87CEEB', # SkyBlue
        'Precision (Macro)': '#778899' # LightSlateGray
    },
    hover_data={'Value': ':.2f'}
)
fig_bar.update_layout(
    yaxis_range=[0, 1],
    font=dict(family="Inter", color="#2F4F4F"),
    legend_title_text='Metric',
    height=400
)
fig_bar.update_yaxes(tickformat=".0%")
st.plotly_chart(fig_bar, use_container_width=True)

# --- Section 2: Detailed Analysis ---
st.header("Detailed Model Analysis")
st.write("Dive deeper into each model's performance. Use the tabs to switch between models and inspect their detailed classification reports and confusion matrices. This reveals class-specific strengths and weaknesses.")

tab_elm, tab_svm, tab_nb = st.tabs(["ELM", "SVM", "Naive Bayes"])

with tab_elm:
    st.subheader("ELM Classification Report")
    report_df_elm = pd.DataFrame(model_data["ELM"]["classificationReport"]).T
    report_df_elm.index.name = "Class"
    st.dataframe(report_df_elm.style.format({'precision': "{:.2f}", 'recall': "{:.2f}", 'f1-score': "{:.2f}"}), use_container_width=True)
    render_confusion_matrix(model_data["ELM"]["confusionMatrix"])

with tab_svm:
    st.subheader("SVM Classification Report")
    report_df_svm = pd.DataFrame(model_data["SVM"]["classificationReport"]).T
    report_df_svm.index.name = "Class"
    st.dataframe(report_df_svm.style.format({'precision': "{:.2f}", 'recall': "{:.2f}", 'f1-score': "{:.2f}"}), use_container_width=True)
    render_confusion_matrix(model_data["SVM"]["confusionMatrix"])

with tab_nb:
    st.subheader("Naive Bayes Classification Report")
    report_df_nb = pd.DataFrame(model_data["Naive Bayes"]["classificationReport"]).T
    report_df_nb.index.name = "Class"
    st.dataframe(report_df_nb.style.format({'precision': "{:.2f}", 'recall': "{:.2f}", 'f1-score': "{:.2f}"}), use_container_width=True)
    render_confusion_matrix(model_data["Naive Bayes"]["confusionMatrix"])

# --- Section 3: Interactive Prediction Tool ---
st.header("Interactive Prediction Tool")
st.write("Test the model's logic. Based on its superior performance, this tool uses a simplified version of the ELM model's decision-making process. Input the features of a website below to get a simulated phishing prediction.")

with st.form("prediction_form"):
    features = [
        "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State", "Domain_registeration_length",
        "Request_URL", "URL_of_Anchor", "Links_in_tags", "SFH", "web_traffic",
        "Google_Index", "Links_pointing_to_page"
    ]
    
    input_values = {}
    cols = st.columns(3)

    for i, feature in enumerate(features):
        with cols[i % 3]:
            if feature=='Prefix_Suffix' or feature=='Domain_registeration_length' or feature=='Request_URL' or feature=='Google_Index':
                readable_label = feature.replace('_', ' ').capitalize()
                input_values[feature] = st.selectbox(
                    f"{readable_label}",
                    options=[-1, 1],
                    format_func=lambda x: {1: "Safe (1)", -1: "Phishing (-1)"}[x],
                    key=f"input_{feature}"
                )
            else:
                readable_label = feature.replace('_', ' ').capitalize()
                input_values[feature] = st.selectbox(
                    f"{readable_label}",
                    options=[-1, 0, 1],
                    format_func=lambda x: {1: "Safe (1)", 0: "Suspicious (0)", -1: "Phishing (-1)"}[x],
                    key=f"input_{feature}"
                )

    st.write("") # Add some vertical space
    submitted = st.form_submit_button("Predict Status")
    inputs=pd.DataFrame([input_values])
    
    pred=elm.predict(inputs)
    if submitted:
        if pred==1:
            st.success('Prediction: Legitimate Website') # Use st.success for "Legitimate"
        else:
            st.error('Prediction: Phishing Website') # Use st.error for "Phishing"
    
# --- Footer ---
st.markdown("---") # A simple line separator
st.write("Designed & Developed by an Expert Frontend Developer.")