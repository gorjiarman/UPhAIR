import os
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

# ==============================================================================
# 📂 FILE AND DIRECTORY PATHS
# All paths are built dynamically to ensure the project works on any computer.
# ==============================================================================
# Base directory of the project (i.e., the 'UPHAIR_Project' folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Input Directories ---
DATA_DIR = os.path.join(BASE_DIR, 'data')
PAPER_DIR = os.path.join(DATA_DIR, 'papers') # For downloaded PDFs

# --- Output Directories ---
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# --- Data & Model Storage ---
# The main CSV file containing patient features and labels.
DATASET_PATH = os.path.join(DATA_DIR, "features.csv") # ⬅️ Make sure this filename is correct
# Path to save/load the pre-built FAISS index for the literature retriever.
FAISS_INDEX_PATH = os.path.join(DATA_DIR, 'faiss_literature_index.bin')
# Path to save/load the processed text from all research papers.
ALL_TEXTS_PATH = os.path.join(DATA_DIR, 'user_dictionary.txt')

END_POINT = os.getenv("END_POINT")
# ==============================================================================
# 🔐 API & SERVICE KEYS
# For accessing external services like Google Gemini and Unpaywall.
# ==============================================================================
# ⚠️ IMPORTANT: Replace placeholders with your actual keys.
# For better security, consider loading these from environment variables instead of hardcoding them.
LLM_API_KEY = os.getenv("LLM_API_KEY")
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL") # Required by the Unpaywall API
CORE_API_KEY = os.getenv("CORE_API_KEY")

# ==============================================================================
# 📊 DATASET & SAMPLING CONFIGURATION
# Defines how the dataset is handled.
# ==============================================================================
LABEL_COLUMN = 'idh'
PATIENT_ID_COLUMN = 'PatientID'
# Index of the patient sample to use for generating the report.
# For example, -1 means the last patient, -2 means the second to last.
SAMPLE_INDEX = -6


# ==============================================================================
# 🤖 MODEL TRAINING CONFIGURATION
# Parameters for the machine learning model training and evaluation process.
# ==============================================================================
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Define classifiers and their hyperparameter grids for GridSearchCV.
# The pipeline will train all of these and select the best one based on performance.
CLASSIFIERS = {
    'GradientBoosting': (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            'clf__max_depth': [5, 10],
            'clf__min_samples_split': [2, 5],
            'clf__min_samples_leaf': [1, 4],
            'clf__max_features': ['sqrt', 'log2']
        }
    ),
}

# ==============================================================================
# 🔍 LITERATURE RETRIEVER CONFIGURATION (RAG)
# Settings for fetching and searching through research papers.
# ==============================================================================
# This query is used to find relevant papers on PubMed.
PUBMED_QUERY = "(IDH OR isocitrate dehydrogenase) AND (glioma OR glioblastoma) AND (radiomics OR MGMT OR age)"
MAX_PUBMED_RESULTS = 1000 # Max number of paper abstracts to fetch.
# The pre-trained model used to create vector embeddings for the text.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2-main/"
# Number of relevant text chunks to retrieve from the literature for the LLM context.
RETRIEVER_TOP_K = 10
MAX_ATTEMPT = 3


# ==============================================================================
# 📝 REPORT GENERATION CONFIGURATION
# Settings for the final PDF report.
# ==============================================================================
# The LLM model to use for generating the clinical narrative.
LLM_MODEL_NAME = "Gemini-2.0-Flash-001"
# Filename for the SHAP waterfall plot image.
SHAP_PLOT_FILENAME = "shap_waterfall_plot.png"
# Final filename for the generated PDF report.
REPORT_FILENAME = "clinical_interpretation_report.pdf"