"""
app.py — TruthLens.ai: Multimodal Fake News Detection & AI-Powered Media Analysis.
Main Streamlit Application Entrypoint & Centralized Navigation Router.
"""
from __future__ import annotations
import streamlit as st

from src.config import APP_NAME, APP_SUBTITLE
from src.utils import inject_css

# Global page configuration for TruthLens.ai
st.set_page_config(
    page_title=f"{APP_NAME} — {APP_SUBTITLE}",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# Centralized Navigation Declarations
p_dashboard = st.Page("pages/1_Dashboard.py", title="Executive Dashboard", icon="📊", default=True, url_path="Dashboard")
p_data_upload = st.Page("pages/2_Data_Upload.py", title="Data Ingestion Studio", icon="📥", url_path="Data_Upload")
p_text = st.Page("pages/3_Text_Analysis.py", title="Text Classification", icon="📝", url_path="Text_Analysis")
p_image = st.Page("pages/4_Image_Analysis.py", title="Image Forensics", icon="🖼️", url_path="Image_Analysis")
p_url = st.Page("pages/5_URL_Analysis.py", title="URL & Web Scanner", icon="🌐", url_path="URL_Analysis")
p_batch = st.Page("pages/6_Dataset_and_Batch.py", title="Batch & Dataset Scoring", icon="📦", url_path="Dataset_and_Batch")
p_model_eval = st.Page("pages/7_Model_Evaluation.py", title="Model Evaluation & ROC", icon="📈", url_path="Model_Evaluation")
p_explain = st.Page("pages/8_Explainability.py", title="Model Explainability", icon="🔍", url_path="Explainability")
p_early = st.Page("pages/9_Early_Detection.py", title="Early Detection", icon="⚡", url_path="Early_Detection")
p_reports = st.Page("pages/10_Research_Reports.py", title="Research Reports", icon="📑", url_path="Research_Reports")
p_history = st.Page("pages/11_Results_History.py", title="Results History", icon="🕒", url_path="Results_History")
p_settings = st.Page("pages/12_Settings.py", title="System Settings", icon="⚙️", url_path="Settings")
p_help = st.Page("pages/13_Help_and_Docs.py", title="Help & Documentation", icon="📚", url_path="Help_and_Docs")

# Navigation Hierarchy
navigation_sections = {
    "Overview": [
        p_dashboard,
    ],
    "Multimodal Studio": [
        p_text,
        p_image,
        p_url,
        p_batch,
    ],
    "Model Engineering & Evaluation": [
        p_data_upload,
        p_model_eval,
        p_explain,
        p_early,
        p_reports,
    ],
    "Administration & Docs": [
        p_history,
        p_settings,
        p_help,
    ],
}

pg = st.navigation(navigation_sections, position="sidebar")
pg.run()
